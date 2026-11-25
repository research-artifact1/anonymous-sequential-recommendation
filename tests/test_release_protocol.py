import json
import pickle
from pathlib import Path

import pytest
import torch

from trimrec.cli import main
from trimrec.config import load_config
from trimrec.data import TrajectoryDataset, load_corpus
from trimrec.metrics import catalog_ranks
from trimrec.variants import VARIANTS, apply_variant


def test_catalog_sort_ties_and_padding():
    scores = torch.ones(2, 3707)
    scores[:, 0] = 1000
    items = torch.tensor([463, 1124])
    masked = scores.clone()
    masked[:, 0] = -torch.inf
    expected = masked.sort(dim=1, descending=True).indices
    expected = expected.eq(items[:, None]).nonzero(as_tuple=True)[1] + 1
    assert torch.equal(catalog_ranks(scores, items), expected)
    assert scores[0, 0] == 1000
    with pytest.raises(ValueError, match="real catalog"):
        catalog_ranks(scores, torch.zeros_like(items))


def test_train_val_only_matches_original_prefix(sequence_file, tmp_path):
    payload = pickle.loads(sequence_file.read_bytes())
    full = load_corpus(sequence_file)
    payload["split"] = "train_val_only"
    payload["sequences"] = {k: seq[:-1] for k, seq in payload["sequences"].items()}
    p = tmp_path / "train_val.pkl"
    p.write_bytes(pickle.dumps(payload))
    packed = load_corpus(p, train_val_only=True)
    assert packed.split_counts == full.split_counts
    for split in ("train", "validation"):
        actual, expected = (
            TrajectoryDataset(corpus, 6, split) for corpus in (packed, full)
        )
        for row, other in zip(actual, expected):
            assert all(torch.equal(row[k], other[k]) for k in row)
    with pytest.raises(ValueError, match="held-out targets"):
        TrajectoryDataset(packed, 6, "test")


def test_heldout_reward_anchor_and_neutral_control(sequence_file):
    payload = pickle.loads(sequence_file.read_bytes())
    corpus = load_corpus(sequence_file)
    before = TrajectoryDataset(corpus, 6, "test")[0]
    val_before = TrajectoryDataset(corpus, 6, "validation")[0]
    payload["sequences"][1][-1][2] = 1
    sequence_file.write_bytes(pickle.dumps(payload))
    changed = load_corpus(sequence_file)
    assert torch.equal(
        TrajectoryDataset(changed, 6, "test")[0]["returns_to_go"],
        before["returns_to_go"],
    )
    payload["sequences"][1][-2][2] = 3
    sequence_file.write_bytes(pickle.dumps(payload))
    changed = load_corpus(sequence_file)
    after = TrajectoryDataset(changed, 6, "test")[0]["returns_to_go"]
    assert torch.equal(after[:3], before["returns_to_go"][:3])
    assert float(after[3]) == pytest.approx((2.0 - 0.5) / 6.0)
    val_after = TrajectoryDataset(changed, 6, "validation")[0]["returns_to_go"]
    assert torch.equal(val_after, val_before["returns_to_go"])
    neutral = TrajectoryDataset(changed, 6, "test", neutral_feedback=True)[0]
    assert torch.allclose(neutral["returns_to_go"][:4], torch.full((4,), 2 / 6))


def _cli_args(tmp_path, sequence_file, feature_tensors):
    config = Path(__file__).resolve().parents[1] / "configs/beauty.yaml"
    text, image = feature_tensors
    args = ["--config", str(config), "--project-root", str(tmp_path), "--device", "cpu"]
    for key, value in {
        "dataset.sequence_path": sequence_file.as_posix(),
        "dataset.text_features_path": text.as_posix(),
        "dataset.image_features_path": image.as_posix(),
        "model.hidden_dim": 16,
        "model.num_layers": 1,
        "model.text_dim": 8,
        "model.image_dim": 8,
        "model.reranker_attention_dim": 8,
        "model.reranker_hidden_dim": 8,
        "model.reranker_top_m": 4,
        "dataset.max_length": 6,
        "training.epochs": 1,
        "training.batch_size": 2,
        "training.eval_batch_size": 2,
        "training.num_workers": 0,
    }.items():
        args.extend(("--set", f"{key}={value}"))
    return args


def test_cli_train_validate_and_test(tmp_path, sequence_file, feature_tensors, capsys):
    args = _cli_args(tmp_path, sequence_file, feature_tensors)
    main(["train", *args])
    (run,) = list((tmp_path / "runs").iterdir())
    assert not json.loads((run / "summary.json").read_bytes())["test_evaluated"]
    main(
        [
            "evaluate",
            *args,
            "--split",
            "validation",
            "--checkpoint",
            str(run / "best.pt"),
            "--compare-refinement",
        ]
    )
    val = json.loads(capsys.readouterr().out)
    assert val["refinement"]["users"] == 2
    main(["evaluate", *args, "--split", "test", "--checkpoint", str(run / "best.pt")])
    test = json.loads(capsys.readouterr().out)
    assert test["split"] == "test" and len(test["checkpoint_sha256"]) == 64
    assert test["protocol"] == val["protocol"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_forward(tmp_path, sequence_file, feature_tensors, variant):
    from trimrec.experiment import build_experiment
    from trimrec.losses import compute_loss

    args = _cli_args(tmp_path, sequence_file, feature_tensors)
    config = apply_variant(load_config(args[1], args[7::2]), variant)
    ex = build_experiment(config, tmp_path)
    ds = TrajectoryDataset(
        ex.corpus, 6, "train", neutral_feedback=config.dataset.neutral_feedback
    )
    batch = {k: v.unsqueeze(0) for k, v in ds[0].items()}
    out = ex.model(batch)
    loss = compute_loss(out, batch, config.objective, 1)
    assert torch.isfinite(loss.loss)
    loss.loss.backward()
    if variant == "similarity_only":
        assert ex.model.reranker.scorer[0].in_features == 3 * 8
        assert not hasattr(ex.model.reranker, "query")
    if variant == "no_candidate_modal_evidence":
        assert ex.model.reranker.scorer[0].in_features == 4 * 8
        assert ex.model.text_projections
    if variant in {"no_refinement", "early_fusion"}:
        assert not out.candidate_ids.numel()
