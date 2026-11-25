"""Full-catalog ranking metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable

import torch


def catalog_ranks(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = logits.clone()
    logits[:, 0] = -float("inf")
    if not bool(((targets > 0) & (targets < logits.size(1))).all()):
        raise ValueError("ranking targets must be real catalog items")
    if not bool(torch.isfinite(logits[:, 1:]).all()):
        raise ValueError("nonfinite real-item score")
    # Keep the historical evaluator's ordering, including its sort tie policy.
    ranking = torch.sort(logits, dim=1, descending=True).indices
    return ranking.eq(targets[:, None]).nonzero(as_tuple=True)[1] + 1


class RankingMetrics:
    def __init__(self, cutoffs: Iterable[int] = (5, 10, 20, 50, 100, 200)) -> None:
        self.cutoffs = tuple(sorted(set(int(k) for k in cutoffs)))
        self.sums: Dict[str, float] = defaultdict(float)
        self.count = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        self.update_ranks(catalog_ranks(logits, targets))

    def update_ranks(self, rank: torch.Tensor) -> None:
        for cutoff in self.cutoffs:
            hit = rank <= cutoff
            self.sums[f"HR@{cutoff}"] += hit.float().sum().item()
            discount = torch.where(
                hit,
                1.0 / torch.log2(rank.float() + 1.0),
                torch.zeros_like(rank, dtype=torch.float32),
            )
            self.sums[f"NDCG@{cutoff}"] += discount.sum().item()
        reciprocal = torch.where(
            rank <= 20,
            rank.float().reciprocal(),
            torch.zeros_like(rank, dtype=torch.float32),
        )
        self.sums["MRR@20"] += reciprocal.sum().item()
        self.count += rank.numel()

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            raise ValueError("no examples were added")
        return {key: value / self.count for key, value in sorted(self.sums.items())}
