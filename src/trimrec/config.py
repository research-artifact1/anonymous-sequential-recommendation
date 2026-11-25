"""Configuration loading and command-line overrides."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Union

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    sequence_path: str
    text_features_path: str
    image_features_path: str
    max_length: int = 50
    timestamp_unit: str = "seconds"
    neutral_feedback: bool = False
    train_val_only: bool = False


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 128
    num_heads: int = 2
    num_layers: int = 2
    max_timestep: int = 4096
    dropout: float = 0.2
    gap_buckets: int = 128
    text_dim: int = 512
    text_streams: int = 1
    image_dim: int = 512
    graph_layers: int = 1
    graph_gamma_max: float = 0.5
    num_interests: int = 4
    interest_gate_max: float = 0.5
    transition_beta_max: float = 3.0
    reranker_top_m: int = 200
    reranker_history_length: int = 8
    reranker_attention_dim: int = 64
    reranker_hidden_dim: int = 64
    reranker_beta_max: float = 3.0
    use_return_token: bool = True
    use_time_gap: bool = True
    use_graph: bool = True
    use_transition: bool = True
    refinement: str = "cross_attention"
    candidate_modalities: bool = True
    mask_missing_text_residual: bool = False
    cpu_graph_eval: bool = False
    fp32_scorer: bool = False
    early_image_fusion: bool = False


@dataclass(frozen=True)
class StructureConfig:
    exclude_last: int = 2
    graph_min_count: int = 2
    graph_top_k: int = 20
    transition_min_count: int = 2
    transition_top_k: int = 20


@dataclass(frozen=True)
class ObjectiveConfig:
    recency_start: float = 0.8
    recency_final: float = 0.5
    recency_anneal_epochs: int = 10
    reranker_aux_weight: float = 0.001
    reranker_aux_top_k: int = 20
    reranker_aux_temperature: float = 1.0


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 2026
    batch_size: int = 256
    eval_batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 200
    patience: int = 20
    gradient_clip_norm: float = 5.0
    num_workers: int = 4
    amp: bool = True
    deterministic: bool = True
    selection_metric: str = "NDCG@20"


@dataclass(frozen=True)
class OutputConfig:
    root: str = "runs"
    run_name: Optional[str] = None


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetConfig
    model: ModelConfig
    structure: StructureConfig
    objective: ObjectiveConfig
    training: TrainingConfig
    output: OutputConfig

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if not self.dataset.name:
            raise ValueError("dataset.name must be set")
        for field_name in (
            "sequence_path",
            "text_features_path",
            "image_features_path",
        ):
            if not getattr(self.dataset, field_name):
                raise ValueError(f"dataset.{field_name} must be set")
        if self.dataset.max_length <= 0:
            raise ValueError("dataset.max_length must be positive")
        if self.dataset.timestamp_unit not in {"seconds", "days"}:
            raise ValueError("dataset.timestamp_unit must be 'seconds' or 'days'")
        if self.model.hidden_dim <= 0 or self.model.num_heads <= 0:
            raise ValueError("model hidden size and head count must be positive")
        if self.model.hidden_dim % self.model.num_heads != 0:
            raise ValueError("model.hidden_dim must be divisible by model.num_heads")
        if self.model.num_layers <= 0 or self.model.graph_layers <= 0:
            raise ValueError("model layer counts must be positive")
        if not 0.0 <= self.model.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        if self.model.gap_buckets <= 1:
            raise ValueError("model.gap_buckets must be greater than 1")
        if self.model.num_interests < 1:
            raise ValueError("model.num_interests must be positive")
        if self.model.refinement not in {"cross_attention", "similarity", "none"}:
            raise ValueError("unknown candidate refinement mode")
        if self.model.refinement == "none" and self.objective.reranker_aux_weight:
            raise ValueError("candidate auxiliary loss requires refinement")
        if self.model.reranker_top_m <= 1:
            raise ValueError("model.reranker_top_m must be greater than 1")
        if self.structure.exclude_last != 2:
            raise ValueError("strict leave-two-out requires structure.exclude_last=2")
        if (
            min(
                self.structure.graph_min_count,
                self.structure.graph_top_k,
                self.structure.transition_min_count,
                self.structure.transition_top_k,
            )
            <= 0
        ):
            raise ValueError("structure counts and Top-K widths must be positive")
        if (
            min(
                self.training.batch_size,
                self.training.eval_batch_size,
                self.training.epochs,
                self.training.patience,
            )
            <= 0
        ):
            raise ValueError(
                "training batch sizes, epochs, and patience must be positive"
            )
        if self.training.num_workers < 0:
            raise ValueError("training.num_workers cannot be negative")
        if self.training.selection_metric != "NDCG@20":
            raise ValueError("the paper protocol selects checkpoints by Val NDCG@20")
        if not 0.0 < self.objective.recency_final <= 1.0:
            raise ValueError("objective.recency_final must be in (0, 1]")
        if not 0.0 < self.objective.recency_start <= 1.0:
            raise ValueError("objective.recency_start must be in (0, 1]")
        if self.objective.reranker_aux_top_k <= 0:
            raise ValueError("objective.reranker_aux_top_k must be positive")


def _deep_merge(
    base: MutableMapping[str, Any], update: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    for key, value in update.items():
        if (
            key in base
            and isinstance(base[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _load_mapping(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    base_name = payload.pop("_base_", None)
    if base_name is None:
        return payload
    base_path = (path.parent / str(base_name)).resolve()
    return dict(_deep_merge(_load_mapping(base_path), payload))


def _parse_scalar(raw: str) -> Any:
    return yaml.safe_load(raw)


def _apply_overrides(payload: Dict[str, Any], overrides: Iterable[str]) -> None:
    for expression in overrides:
        if "=" not in expression:
            raise ValueError(f"override must use key=value syntax: {expression}")
        dotted_key, raw_value = expression.split("=", 1)
        keys = dotted_key.split(".")
        target: MutableMapping[str, Any] = payload
        for key in keys[:-1]:
            value = target.get(key)
            if not isinstance(value, MutableMapping):
                raise KeyError(f"unknown configuration path: {dotted_key}")
            target = value
        leaf = keys[-1]
        classes = {
            "dataset": DatasetConfig,
            "model": ModelConfig,
            "structure": StructureConfig,
            "objective": ObjectiveConfig,
            "training": TrainingConfig,
            "output": OutputConfig,
        }
        section = classes.get(keys[0])
        if leaf not in target and not (
            len(keys) == 2
            and section is not None
            and leaf in section.__dataclass_fields__
        ):
            raise KeyError(f"unknown configuration key: {dotted_key}")
        target[leaf] = _parse_scalar(raw_value)


def load_config(
    path: Union[str, Path], overrides: Iterable[str] = ()
) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    payload = _load_mapping(config_path)
    _apply_overrides(payload, overrides)
    try:
        config = ExperimentConfig(
            dataset=DatasetConfig(**payload["dataset"]),
            model=ModelConfig(**payload["model"]),
            structure=StructureConfig(**payload["structure"]),
            objective=ObjectiveConfig(**payload["objective"]),
            training=TrainingConfig(**payload["training"]),
            output=OutputConfig(**payload["output"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid configuration in {config_path}: {exc}") from exc
    config.validate()
    return config
