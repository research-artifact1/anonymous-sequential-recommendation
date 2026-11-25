"""Command-line interface."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional

from trimrec.checkpoint import load_checkpoint
from trimrec.config import load_config
from trimrec.data import TrajectoryDataset
from trimrec.engine import create_loader, evaluate, train
from trimrec.experiment import build_experiment
from trimrec.ml1m import attach_test_targets, prepare
from trimrec.runtime import (
    configure_logging,
    create_run_directory,
    environment,
    resolve_device,
    set_seed,
    sha256_file,
    write_json,
)
from trimrec.variants import VARIANTS, apply_variant


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Dataset YAML configuration")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Base directory for relative data and output paths",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--variant", choices=VARIANTS, default="full")
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
    eval_parser.add_argument("--test-source", type=Path, help="Original ML-1M ZIP")
    eval_parser.add_argument("--output", help="Optional JSON result path")
    eval_parser.add_argument(
        "--compare-refinement",
        action="store_true",
        help="On validation only, report pre/post refinement rankings",
    )

    inspect_parser = subparsers.add_parser(
        "inspect", help="Validate inputs and report dataset/model statistics"
    )
    _common_parser(inspect_parser)
    data_parser = subparsers.add_parser("prepare-ml1m", help="Verify and prepare ML-1M")
    for name in ("ml1m", "ml10m", "text", "image", "output"):
        data_parser.add_argument(f"--{name}", required=True, type=Path)
    return parser


def _config_and_experiment(args: argparse.Namespace):
    config = apply_variant(load_config(args.config, args.overrides), args.variant)
    set_seed(config.training.seed, config.training.deterministic)
    experiment = build_experiment(config, args.project_root)
    return config, experiment


def _train(args: argparse.Namespace) -> None:
    config, experiment = _config_and_experiment(args)
    device = resolve_device(args.device)
    output_root = Path(args.project_root) / config.output.root
    run_name = config.output.run_name or (
        f"{config.dataset.name}_{args.variant}_seed{config.training.seed}"
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
        write_json(run_directory / "environment.json", environment(device))

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
    if args.compare_refinement and args.split != "validation":
        raise ValueError("Refinement diagnostics are Validation only")
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

    corpus = experiment.corpus
    if args.split == "test":
        if payload["epoch"] != payload["best_epoch"]:
            raise ValueError("Test requires the Val-selected best checkpoint")
        if corpus.train_val_only:
            if args.test_source is None:
                raise ValueError("ML-1M Test requires --test-source /path/to/ml-1m.zip")
            corpus = attach_test_targets(corpus, args.test_source)
        elif args.test_source is not None:
            raise ValueError("--test-source applies only to ML-1M Test")
    elif args.test_source is not None:
        raise ValueError("Validation does not accept a Test archive")

    dataset = TrajectoryDataset(
        corpus,
        config.dataset.max_length,
        args.split,
        config.dataset.timestamp_unit,
        config.model.gap_buckets,
        config.dataset.neutral_feedback,
    )
    loader = create_loader(
        dataset,
        config.training.eval_batch_size,
        False,
        config.training.num_workers,
        config.training.seed + (202 if args.split == "validation" else 303),
    )
    comparison = {} if args.compare_refinement else None
    metrics = evaluate(
        experiment.model,
        loader,
        device,
        amp=config.training.amp,
        refinement_report=comparison,
    )
    recipe = config.to_dict()
    recipe.pop("output")
    recipe["training"].pop("seed")
    recipe_bytes = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    result = {
        "dataset": config.dataset.name,
        "split": args.split,
        "seed": config.training.seed,
        "checkpoint": Path(args.checkpoint).name,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "input_sha256": experiment.fingerprints,
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "variant": args.variant,
        "protocol": "leave_two_out_full_catalog_mask_0_torch_sort_desc_v1",
        "environment": environment(device),
        "checkpoint_epoch": payload["epoch"],
        "metrics": metrics,
    }
    if comparison is not None:
        result["refinement"] = comparison
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def _inspect(args: argparse.Namespace) -> None:
    config, experiment = _config_and_experiment(args)
    corpus = experiment.corpus
    report = {
        "dataset": config.dataset.name,
        "users": experiment.corpus.user_count,
        "items": experiment.corpus.item_count,
        "interactions": corpus.interaction_count
        + (corpus.user_count if corpus.train_val_only else 0),
        "stored_interactions": corpus.interaction_count,
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
    elif args.command == "prepare-ml1m":
        print(
            json.dumps(
                prepare(args.ml1m, args.ml10m, args.text, args.image, args.output),
                indent=2,
            )
        )
    else:
        _inspect(args)


if __name__ == "__main__":
    main()
