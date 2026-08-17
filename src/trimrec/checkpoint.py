"""Checkpoint serialization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from trimrec.model.trimrec import TRIMRec


def save_checkpoint(
    path: Union[str, Path],
    model: TRIMRec,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    config: Dict[str, Any],
    input_fingerprints: Dict[str, str],
    best_epoch: Optional[int] = None,
    best_validation_metrics: Optional[Dict[str, float]] = None,
    stale_epochs: int = 0,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "schema": "trimrec_checkpoint_v1",
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch if best_epoch is not None else epoch),
        "best_validation_metrics": best_validation_metrics or {},
        "stale_epochs": int(stale_epochs),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "input_fingerprints": input_fingerprints,
    }
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_checkpoint(
    path: Union[str, Path],
    model: TRIMRec,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Union[str, torch.device] = "cpu",
    expected_config: Optional[Dict[str, Any]] = None,
    expected_input_fingerprints: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location=map_location, weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "trimrec_checkpoint_v1"
    ):
        raise ValueError(f"unsupported checkpoint format: {source}")
    if (
        expected_input_fingerprints is not None
        and payload.get("input_fingerprints") != expected_input_fingerprints
    ):
        raise ValueError("checkpoint input fingerprints do not match current inputs")
    if expected_config is not None:
        saved_config = payload.get("config", {})
        contract_sections = ("dataset", "model", "structure", "objective")
        for section in contract_sections:
            if saved_config.get(section) != expected_config.get(section):
                raise ValueError(
                    f"checkpoint configuration mismatch in section: {section}"
                )
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload
