from pathlib import Path

import pytest

from trimrec.config import load_config


def test_dataset_config_inheritance_and_override():
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root / "configs" / "sports.yaml",
        ("training.seed=123", "training.batch_size=128"),
    )
    assert config.dataset.name == "Sports_and_Outdoors_2014"
    assert config.model.dropout == 0.3
    assert config.training.seed == 123
    assert config.training.batch_size == 128


def test_unknown_override_is_rejected():
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(KeyError):
        load_config(root / "configs" / "beauty.yaml", ("model.unknown=1",))
