"""Training-prefix graph and directed transition prior construction."""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Tuple

import torch

from trimrec.data import Corpus


def _training_items(corpus: Corpus, exclude_last: int) -> Iterable[List[int]]:
    for sequence in corpus.sequences.values():
        omitted = exclude_last - int(corpus.train_val_only)
        visible = sequence[:-omitted] if omitted > 0 else sequence
        yield [event.item_id for event in visible]


def build_item_graph(
    corpus: Corpus,
    exclude_last: int = 2,
    min_count: int = 2,
    top_k: int = 20,
) -> torch.Tensor:
    """Build a symmetric degree-normalized graph from visible transitions."""

    edge_counts: DefaultDict[Tuple[int, int], float] = defaultdict(float)
    for items in _training_items(corpus, exclude_last):
        for left, right in zip(items, items[1:]):
            if left == right:
                continue
            edge = (left, right) if left < right else (right, left)
            edge_counts[edge] += 1.0

    retained = {
        edge: count for edge, count in edge_counts.items() if count >= min_count
    }
    if top_k > 0 and retained:
        neighbors: DefaultDict[int, List[Tuple[int, float]]] = defaultdict(list)
        for (left, right), count in retained.items():
            neighbors[left].append((right, count))
            neighbors[right].append((left, count))
        selected_edges = set()
        for source, candidates in neighbors.items():
            selected_neighbors = sorted(candidates, key=lambda row: (-row[1], row[0]))[
                :top_k
            ]
            for target, _ in selected_neighbors:
                edge = (source, target) if source < target else (target, source)
                selected_edges.add(edge)
        retained = {edge: retained[edge] for edge in selected_edges}

    size = corpus.item_count + 1
    if not retained:
        diagonal = torch.arange(size)
        return torch.sparse_coo_tensor(
            torch.stack([diagonal, diagonal]),
            torch.ones(size),
            (size, size),
            check_invariants=False,
        ).coalesce()

    rows: List[int] = []
    columns: List[int] = []
    weights: List[float] = []
    for (left, right), count in retained.items():
        rows.extend((left, right))
        columns.extend((right, left))
        weights.extend((count, count))

    indices = torch.tensor([rows, columns], dtype=torch.long)
    values = torch.tensor(weights, dtype=torch.float32)
    degree = torch.zeros(size, dtype=torch.float32)
    degree.index_add_(0, indices[0], values)
    normalized = values / torch.sqrt(
        degree[indices[0]].clamp_min(1e-8) * degree[indices[1]].clamp_min(1e-8)
    )
    return torch.sparse_coo_tensor(
        indices,
        normalized,
        (size, size),
        check_invariants=False,
    ).coalesce()


def build_transition_table(
    corpus: Corpus,
    exclude_last: int = 2,
    min_count: int = 2,
    top_k: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build fixed-width directed successor IDs and row-normalized weights."""

    outgoing: DefaultDict[int, Dict[int, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for items in _training_items(corpus, exclude_last):
        for source, target in zip(items, items[1:]):
            if source != target:
                outgoing[source][target] += 1.0

    indices = torch.zeros(corpus.item_count + 1, top_k, dtype=torch.long)
    values = torch.zeros(corpus.item_count + 1, top_k, dtype=torch.float32)
    for source, row in outgoing.items():
        retained = [
            (target, count) for target, count in row.items() if count >= min_count
        ]
        retained.sort(key=lambda pair: (-pair[1], pair[0]))
        retained = retained[:top_k]
        if not retained:
            continue
        total = sum(count for _, count in retained)
        for column, (target, count) in enumerate(retained):
            indices[source, column] = target
            values[source, column] = count / total
    return indices, values
