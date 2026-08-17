"""Full-catalog ranking metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable

import torch


class RankingMetrics:
    def __init__(self, cutoffs: Iterable[int] = (5, 10, 20, 50, 100, 200)) -> None:
        self.cutoffs = tuple(sorted(set(int(k) for k in cutoffs)))
        self.sums: Dict[str, float] = defaultdict(float)
        self.count = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        logits = logits.float().clone()
        logits[:, 0] = -float("inf")
        maximum = min(max(self.cutoffs), logits.size(1) - 1)
        ranking = torch.topk(logits, k=maximum, dim=1).indices
        matches = ranking.eq(targets[:, None])
        found = matches.any(dim=1)
        rank = matches.float().argmax(dim=1) + 1
        rank = torch.where(found, rank, torch.zeros_like(rank))

        for cutoff in self.cutoffs:
            hit = found & (rank <= cutoff)
            self.sums[f"HR@{cutoff}"] += hit.float().sum().item()
            discount = torch.where(
                hit,
                1.0 / torch.log2(rank.float() + 1.0),
                torch.zeros_like(rank, dtype=torch.float32),
            )
            self.sums[f"NDCG@{cutoff}"] += discount.sum().item()
        reciprocal = torch.where(
            found & (rank <= 20),
            rank.float().reciprocal(),
            torch.zeros_like(rank, dtype=torch.float32),
        )
        self.sums["MRR@20"] += reciprocal.sum().item()
        self.count += targets.numel()

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            raise ValueError("no examples were added")
        return {key: value / self.count for key, value in sorted(self.sums.items())}
