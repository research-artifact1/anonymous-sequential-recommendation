"""Experiment assembly shared by training and evaluation commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

from trimrec.config import ExperimentConfig
from trimrec.data import Corpus, load_corpus
from trimrec.features import FrozenFeatures, load_frozen_features
from trimrec.model.trimrec import TRIMRec
from trimrec.runtime import sha256_file
from trimrec.structure import build_item_graph, build_transition_table


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    corpus: Corpus
    features: FrozenFeatures
    model: TRIMRec
    fingerprints: Dict[str, str]


def resolve_input_path(project_root: Union[str, Path], value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(project_root).expanduser() / path
    return path.resolve()


def build_experiment(
    config: ExperimentConfig, project_root: Union[str, Path] = "."
) -> Experiment:
    sequence_path = resolve_input_path(project_root, config.dataset.sequence_path)
    text_path = resolve_input_path(project_root, config.dataset.text_features_path)
    image_path = resolve_input_path(project_root, config.dataset.image_features_path)
    corpus = load_corpus(sequence_path)
    features = load_frozen_features(
        text_path,
        image_path,
        item_count=corpus.item_count,
        text_dim=config.model.text_dim,
        image_dim=config.model.image_dim,
        text_streams=config.model.text_streams,
    )
    graph = build_item_graph(
        corpus,
        exclude_last=config.structure.exclude_last,
        min_count=config.structure.graph_min_count,
        top_k=config.structure.graph_top_k,
    )
    transition_indices, transition_values = build_transition_table(
        corpus,
        exclude_last=config.structure.exclude_last,
        min_count=config.structure.transition_min_count,
        top_k=config.structure.transition_top_k,
    )
    model = TRIMRec(
        config.model,
        corpus.item_count,
        features,
        graph,
        transition_indices,
        transition_values,
    )
    fingerprints = {
        "sequences": sha256_file(sequence_path),
        "text_features": sha256_file(text_path),
        "image_features": sha256_file(image_path),
    }
    return Experiment(config, corpus, features, model, fingerprints)
