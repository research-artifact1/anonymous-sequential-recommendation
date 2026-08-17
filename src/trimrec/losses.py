"""Training objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from trimrec.config import ObjectiveConfig
from trimrec.model.trimrec import TRIMRecOutput


@dataclass(frozen=True)
class LossOutput:
    loss: torch.Tensor
    main_loss: torch.Tensor
    reranker_aux_loss: torch.Tensor
    candidate_recall: torch.Tensor
    recency_decay: float


def scheduled_recency_decay(config: ObjectiveConfig, epoch: int) -> float:
    if config.recency_anneal_epochs <= 1:
        return config.recency_final
    progress = min(max((epoch - 1) / float(config.recency_anneal_epochs - 1), 0.0), 1.0)
    return math.exp(
        (1.0 - progress) * math.log(config.recency_start)
        + progress * math.log(config.recency_final)
    )


def recency_weighted_cross_entropy(
    logits: torch.Tensor,
    actions: torch.Tensor,
    attention_mask: torch.Tensor,
    decay: float,
) -> torch.Tensor:
    batch_size, sequence_length, catalog_size = logits.shape
    token_loss = F.cross_entropy(
        logits.reshape(-1, catalog_size),
        actions.reshape(-1),
        ignore_index=0,
        reduction="none",
    ).reshape(batch_size, sequence_length)
    valid = attention_mask.bool() & (actions > 0)
    final_positions = (
        (attention_mask.sum(1) - 1).long().clamp(min=0, max=sequence_length - 1)
    )
    positions = torch.arange(sequence_length, device=logits.device)[None]
    distance = (final_positions[:, None] - positions).clamp(min=0)
    base = torch.as_tensor(decay, dtype=token_loss.dtype, device=logits.device)
    weights = torch.pow(base, distance.to(token_loss.dtype))
    weights *= valid.to(token_loss.dtype)
    return (token_loss * weights).sum() / weights.sum().clamp_min(1e-8)


def candidate_topk_cross_entropy(
    output: TRIMRecOutput,
    actions: torch.Tensor,
    top_k: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_index = torch.arange(actions.size(0), device=actions.device)
    target = actions[batch_index, output.final_positions]
    match = output.candidate_ids.eq(target[:, None])
    in_pool = (target > 0) & match.any(dim=1)
    candidate_recall = in_pool.float().mean().detach()
    if not in_pool.any() or output.candidate_ids.size(1) <= 1:
        return output.reranker_scores.sum() * 0.0, candidate_recall

    target_column = match[in_pool].float().argmax(dim=1)
    final_logits = output.logits[batch_index, output.final_positions].float()
    pool_logits = final_logits.gather(1, output.candidate_ids)[in_pool]
    positive = pool_logits.gather(1, target_column[:, None])
    negatives = pool_logits.masked_fill(match[in_pool], -float("inf"))
    hard_k = max(1, min(int(top_k), negatives.size(1) - 1))
    hard_indices = torch.topk(negatives.detach(), k=hard_k, dim=1).indices
    hard_negatives = negatives.gather(1, hard_indices)
    scores = torch.cat((positive, hard_negatives), dim=1)
    scores /= max(float(temperature), 1e-6)
    labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
    return F.cross_entropy(scores, labels), candidate_recall


def compute_loss(
    output: TRIMRecOutput,
    batch: dict[str, torch.Tensor],
    config: ObjectiveConfig,
    epoch: int,
) -> LossOutput:
    decay = scheduled_recency_decay(config, epoch)
    main = recency_weighted_cross_entropy(
        output.logits,
        batch["actions"],
        batch["attention_mask"],
        decay,
    )
    auxiliary, recall = candidate_topk_cross_entropy(
        output,
        batch["actions"],
        config.reranker_aux_top_k,
        config.reranker_aux_temperature,
    )
    total = main + config.reranker_aux_weight * auxiliary
    return LossOutput(total, main, auxiliary, recall, decay)
