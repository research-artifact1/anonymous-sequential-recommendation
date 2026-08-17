import pytest
import torch

from trimrec.metrics import RankingMetrics


def test_ranking_metrics():
    logits = torch.tensor(
        [
            [-9.0, 0.1, 0.4, 0.3],
            [-9.0, 0.7, 0.6, 0.5],
        ]
    )
    targets = torch.tensor([2, 3])
    metrics = RankingMetrics((1, 2, 3))
    metrics.update(logits, targets)
    result = metrics.compute()
    assert result["HR@1"] == pytest.approx(0.5)
    assert result["HR@3"] == pytest.approx(1.0)
    assert result["NDCG@3"] > result["NDCG@1"]
