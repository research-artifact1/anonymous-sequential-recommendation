"""Training and full-catalog evaluation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from trimrec.checkpoint import load_checkpoint, save_checkpoint
from trimrec.config import ExperimentConfig
from trimrec.data import Corpus, TrajectoryDataset
from trimrec.losses import compute_loss
from trimrec.metrics import RankingMetrics
from trimrec.model.trimrec import TRIMRec
from trimrec.runtime import JsonlWriter, seed_worker, write_json


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_validation_metric: float
    best_checkpoint: Path
    validation_metrics: Dict[str, float]


def move_batch(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def create_loader(
    dataset: TrajectoryDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    drop_last: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def evaluate(
    model: TRIMRec,
    loader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    show_progress: bool = True,
) -> Dict[str, float]:
    model.eval()
    metrics = RankingMetrics()
    retrieval_hits = 0.0
    examples = 0
    iterator = tqdm(loader, desc="Evaluate", leave=False, disable=not show_progress)
    with torch.inference_mode():
        for cpu_batch in iterator:
            batch = move_batch(cpu_batch, device)
            output = model(batch)
            batch_index = torch.arange(batch["states"].size(0), device=device)
            targets = batch["actions"][batch_index, output.final_positions]
            final_logits = output.logits[batch_index, output.final_positions]
            metrics.update(final_logits, targets)
            retrieval_hits += (
                output.candidate_ids.eq(targets[:, None]).any(1).sum().item()
            )
            examples += targets.numel()
    result = metrics.compute()
    result["RetrievalRecall@M"] = retrieval_hits / max(examples, 1)
    return result


def train(
    model: TRIMRec,
    corpus: Corpus,
    config: ExperimentConfig,
    device: torch.device,
    run_directory: Path,
    logger: logging.Logger,
    fingerprints: Dict[str, str],
    resume: Optional[Union[str, Path]] = None,
) -> TrainingResult:
    train_dataset = TrajectoryDataset(
        corpus,
        config.dataset.max_length,
        "train",
        config.dataset.timestamp_unit,
        config.model.gap_buckets,
    )
    validation_dataset = TrajectoryDataset(
        corpus,
        config.dataset.max_length,
        "validation",
        config.dataset.timestamp_unit,
        config.model.gap_buckets,
    )
    train_loader = create_loader(
        train_dataset,
        config.training.batch_size,
        True,
        config.training.num_workers,
        config.training.seed + 101,
        drop_last=True,
    )
    validation_loader = create_loader(
        validation_dataset,
        config.training.eval_batch_size,
        False,
        config.training.num_workers,
        config.training.seed + 202,
    )

    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    amp_enabled = config.training.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    first_epoch = 1
    best_metric = float("-inf")
    best_epoch = 0
    best_metrics: Dict[str, float] = {}
    checkpoint_path = run_directory / "best.pt"
    stale_epochs = 0

    if resume is not None:
        payload = load_checkpoint(
            resume,
            model,
            optimizer,
            map_location=device,
            expected_config=config.to_dict(),
            expected_input_fingerprints=fingerprints,
        )
        first_epoch = int(payload["epoch"]) + 1
        best_metric = float(payload["best_metric"])
        best_epoch = int(payload.get("best_epoch", payload["epoch"]))
        best_metrics = dict(payload.get("best_validation_metrics", {}))
        stale_epochs = int(payload.get("stale_epochs", 0))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "Cannot resume without best.pt in the checkpoint directory"
            )
        logger.info("Resumed from epoch %d", first_epoch - 1)

    history = JsonlWriter(run_directory / "history.jsonl")
    for epoch in range(first_epoch, config.training.epochs + 1):
        started = time.perf_counter()
        model.train()
        totals = {"loss": 0.0, "main": 0.0, "aux": 0.0, "recall": 0.0}
        batches = 0
        progress = tqdm(train_loader, desc=f"Train {epoch}", leave=False)
        for cpu_batch in progress:
            batch = move_batch(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = model(batch)
                losses = compute_loss(output, batch, config.objective, epoch)
            scaler.scale(losses.loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            totals["loss"] += losses.loss.detach().item()
            totals["main"] += losses.main_loss.detach().item()
            totals["aux"] += losses.reranker_aux_loss.detach().item()
            totals["recall"] += losses.candidate_recall.detach().item()
            batches += 1
            progress.set_postfix(loss=f"{totals['loss'] / batches:.4f}")

        if batches == 0:
            raise ValueError(
                "Training produced no batches. Reduce training.batch_size or add data."
            )

        validation = evaluate(model, validation_loader, device)
        selected = validation[config.training.selection_metric]
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "seconds": elapsed,
            "train_loss": totals["loss"] / max(batches, 1),
            "train_main_loss": totals["main"] / max(batches, 1),
            "train_reranker_aux_loss": totals["aux"] / max(batches, 1),
            "train_retrieval_recall": totals["recall"] / max(batches, 1),
            "recency_decay": losses.recency_decay,
            "validation": validation,
        }
        history.append(row)
        logger.info(
            "Epoch %d | loss %.5f | Val NDCG@20 %.6f | %.1fs",
            epoch,
            row["train_loss"],
            validation["NDCG@20"],
            elapsed,
        )

        improved = selected > best_metric
        if improved:
            best_metric = selected
            best_epoch = epoch
            best_metrics = validation
            stale_epochs = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                best_metric,
                config.to_dict(),
                fingerprints,
                best_epoch=best_epoch,
                best_validation_metrics=best_metrics,
                stale_epochs=stale_epochs,
            )
            logger.info("Saved best checkpoint: %s", checkpoint_path)
        else:
            stale_epochs += 1

        save_checkpoint(
            run_directory / "last.pt",
            model,
            optimizer,
            epoch,
            best_metric,
            config.to_dict(),
            fingerprints,
            best_epoch=best_epoch,
            best_validation_metrics=best_metrics,
            stale_epochs=stale_epochs,
        )
        if stale_epochs >= config.training.patience:
            logger.info("Early stopping after %d stale epochs", stale_epochs)
            break

    if best_epoch == 0:
        raise RuntimeError("training finished without a valid checkpoint")
    summary = {
        "best_epoch": best_epoch,
        "selection_metric": config.training.selection_metric,
        "best_validation_metric": best_metric,
        "validation_metrics": best_metrics,
        "checkpoint": str(checkpoint_path),
        "test_evaluated": False,
    }
    write_json(run_directory / "summary.json", summary)
    return TrainingResult(best_epoch, best_metric, checkpoint_path, best_metrics)
