"""Loading and validation for frozen multimodal item features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch


@dataclass(frozen=True)
class FrozenFeatures:
    text: torch.Tensor
    image: torch.Tensor

    def to(self, device: torch.device) -> "FrozenFeatures":
        return FrozenFeatures(self.text.to(device), self.image.to(device))


def _load_tensor(path: Union[str, Path]) -> torch.Tensor:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    value = torch.load(source, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        tensor_values = [item for item in value.values() if torch.is_tensor(item)]
        if len(tensor_values) != 1:
            raise ValueError(f"expected one tensor in {source}")
        value = tensor_values[0]
    if not torch.is_tensor(value):
        raise TypeError(f"feature file does not contain a tensor: {source}")
    return value.detach().float().contiguous()


def load_frozen_features(
    text_path: Union[str, Path],
    image_path: Union[str, Path],
    item_count: int,
    text_dim: int = 512,
    image_dim: int = 512,
    text_streams: int = 1,
) -> FrozenFeatures:
    text = _load_tensor(text_path)
    image = _load_tensor(image_path)
    expected_rows = item_count + 1

    if text.ndim == 2:
        text = text.unsqueeze(1)
    if text.shape != (expected_rows, text_streams, text_dim):
        raise ValueError(
            "text features must have shape "
            f"{(expected_rows, text_streams, text_dim)}, got {tuple(text.shape)}"
        )
    if image.shape != (expected_rows, image_dim):
        raise ValueError(
            f"image features must have shape {(expected_rows, image_dim)}, "
            f"got {tuple(image.shape)}"
        )
    if torch.count_nonzero(text[0]) or torch.count_nonzero(image[0]):
        raise ValueError("feature row 0 must be reserved for padding")
    if not torch.isfinite(text).all() or not torch.isfinite(image).all():
        raise ValueError("features contain NaN or infinite values")
    return FrozenFeatures(text=text, image=image)
