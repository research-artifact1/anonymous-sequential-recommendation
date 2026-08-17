from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import torch


@pytest.fixture()
def sequence_file(tmp_path: Path) -> Path:
    payload = {
        "sequences": {
            1: [
                [0, 1, 5.0],
                [86400, 2, 4.0],
                [172800, 3, 3.0],
                [259200, 4, 5.0],
                [345600, 5, 4.0],
            ],
            2: [
                [0, 2, 5.0],
                [86400, 3, 2.0],
                [172800, 4, 4.0],
                [259200, 5, 5.0],
            ],
        },
        "itemmap": {str(index): index for index in range(1, 6)},
        "usermap": {"u1": 1, "u2": 2},
    }
    path = tmp_path / "sequences.pkl"
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


@pytest.fixture()
def feature_tensors(tmp_path: Path):
    generator = torch.Generator().manual_seed(7)
    text = torch.randn(6, 8, generator=generator)
    image = torch.randn(6, 8, generator=generator)
    text[0] = 0
    image[0] = 0
    text_path = tmp_path / "text.pt"
    image_path = tmp_path / "image.pt"
    torch.save(text, text_path)
    torch.save(image, image_path)
    return text_path, image_path
