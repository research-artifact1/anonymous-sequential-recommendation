"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from trimrec.checkpoint import load_checkpoint
from trimrec.config import load_config
from trimrec.data import TrajectoryDataset
from trimrec.engine import create_loader, evaluate, train
from trimrec.experiment import build_experiment
from trimrec.runtime import (
    configure_logging,
    create_run_directory,
    resolve_device,
    set_seed,
    write_json,
)


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Dataset YAML configuration")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Base directory for relative data and output paths",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a configuration value; may be repeated",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trimrec",
        description="TRIMRec training and checkpoint evaluation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train with Val-only selection")
    _common_parser(train_parser)
    train_parser.add_argument("--resume", help="Resume optimizer and model state")
    train_parser.add_argument("--verbose", action="store_true")

    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate one frozen checkpoint explicitly"
    )
    _common_parser(eval_parser)
    eval_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint produced by a trusted TRIMRec training run",
    )
    eval_parser.add_argument("--split", required=True, choices=("validation", "test"))
    eval_parser.add_argument("--output", help="Optional JSON result path")

    inspect_parser = subparsers.add_parser(
        "inspect", help="Validate inputs and report dataset/model statistics"
    )
    _common_parser(inspect_parser)
    return parser


def _config_and_experiment(args: argparse.Namespace):
    config = load_config(args.config, args.overrides)
    set_seed(config.training.seed, config.training.deterministic)
    experiment = build_experiment(config, args.project_root)
    return config, experiment


def _train(args: argparse.Namespace) -> None:
    config, experiment = _config_and_experiment(args)
    device = resolve_device(args.device)
    output_root = Path(args.project_root) / config.output.root
    run_name = config.output.run_name or (
        f"{config.dataset.name}_seed{config.training.seed}"
    )
    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        run_directory = resume_path.parent
    else:
        run_directory = create_run_directory(output_root, run_name)
    logger = configure_logging(run_directory, args.verbose)
    if not args.resume:
        write_json(run_directory / "config.json", config.to_dict())
        write_json(run_directory / "inputs.json", experiment.fingerprints)

    logger.info("Dataset: %s", config.dataset.name)
    logger.info(
        "Users=%d Items=%d Interactions=%d Splits=%s",
        experiment.corpus.user_count,
        experiment.corpus.item_count,
        experiment.corpus.interaction_count,
        experiment.corpus.split_counts,
    )
    logger.info(
        "Parameters: total=%d trainable=%d",
        experiment.model.parameter_count(),
        experiment.model.parameter_count(trainable_only=True),
    )
    logger.info("Device: %s", device)
    result = train(
        experiment.model,
        experiment.corpus,
        config,
        device,
        run_directory,
        logger,
        experiment.fingerprints,
        resume=args.resume,
    )
    logger.info(
        "Best Val NDCG@20 %.6f at epoch %d",
        result.best_validation_metric,
        result.best_epoch,
    )
    logger.info("Test was not evaluated. Use the explicit evaluate command.")


def _evaluate(args: argparse.Namespace) -> None:
    config, experiment = _config_and_experiment(args)
    device = resolve_device(args.device)
    experiment.model.to(device)
    payload = load_checkpoint(
        args.checkpoint,
        experiment.model,
        map_location=device,
        expected_config=config.to_dict(),
        expected_input_fingerprints=experiment.fingerprints,
    )

    dataset = TrajectoryDataset(
        experiment.corpus,
        config.dataset.max_length,
        args.split,
        config.dataset.timestamp_unit,
        config.model.gap_buckets,
    )
    loader = create_loader(
        dataset,
        config.training.eval_batch_size,
        False,
        config.training.num_workers,
        config.training.seed + (202 if args.split == "validation" else 303),
    )
    metrics = evaluate(experiment.model, loader, device)
    result = {
        "dataset": config.dataset.name,
        "split": args.split,
        "seed": config.training.seed,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_epoch": payload["epoch"],
        "metrics": metrics,
    }
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def _inspect(args: argparse.Namespace) -> None:
    config, experiment = _config_and_experiment(args)
    report = {
        "dataset": config.dataset.name,
        "users": experiment.corpus.user_count,
        "items": experiment.corpus.item_count,
        "interactions": experiment.corpus.interaction_count,
        "splits": experiment.corpus.split_counts,
        "text_shape": list(experiment.features.text.shape),
        "image_shape": list(experiment.features.image.shape),
        "parameters": experiment.model.parameter_count(),
        "trainable_parameters": experiment.model.parameter_count(True),
        "input_fingerprints": experiment.fingerprints,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "train":
        _train(args)
    elif args.command == "evaluate":
        _evaluate(args)
    else:
        _inspect(args)


if __name__ == "__main__":
    main()
