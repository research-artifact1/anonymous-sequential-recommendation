"""Strict leave-two-out trajectory data used by TRIMRec."""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Hashable, List, Mapping, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Event:
    timestamp: float
    item_id: int
    rating: float


@dataclass(frozen=True)
class Corpus:
    sequences: Mapping[Hashable, Tuple[Event, ...]]
    item_count: int
    interaction_count: int

    @property
    def user_count(self) -> int:
        return len(self.sequences)

    @property
    def split_counts(self) -> Dict[str, int]:
        valid_users = sum(len(sequence) >= 3 for sequence in self.sequences.values())
        train = sum(max(len(sequence) - 3, 0) for sequence in self.sequences.values())
        return {"train": train, "validation": valid_users, "test": valid_users}


def _parse_event(raw: object) -> Event:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("events must contain timestamp, item ID, and rating")
    if len(raw) < 3:
        raise ValueError("events must contain timestamp, item ID, and rating")
    event = Event(float(raw[0]), int(raw[1]), float(raw[2]))
    if event.item_id <= 0:
        raise ValueError("real item IDs must start at 1; item ID 0 is padding")
    return event


def load_corpus(path: Union[str, Path]) -> Corpus:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    required = {"sequences", "itemmap", "usermap"}
    if not isinstance(payload, Mapping) or not required.issubset(payload):
        raise ValueError(f"sequence file must contain mappings: {sorted(required)}")
    if not all(isinstance(payload[key], Mapping) for key in required):
        raise ValueError("sequences, itemmap, and usermap must be mappings")
    item_ids = {int(value) for value in payload["itemmap"].values()}
    expected_ids = set(range(1, len(item_ids) + 1))
    if item_ids != expected_ids:
        raise ValueError("itemmap IDs must be contiguous and start at 1")

    parsed: Dict[Hashable, Tuple[Event, ...]] = {}
    item_count = len(item_ids)
    interactions = 0
    for user_id, raw_sequence in payload["sequences"].items():
        events = tuple(_parse_event(raw) for raw in raw_sequence)
        if len(events) < 3:
            continue
        if any(
            left.timestamp > right.timestamp for left, right in zip(events, events[1:])
        ):
            raise ValueError(f"timestamps are not monotonic for user {user_id!r}")
        parsed[user_id] = events
        interactions += len(events)
        if any(event.item_id > item_count for event in events):
            raise ValueError(f"sequence contains an unmapped item for user {user_id!r}")
    if not parsed:
        raise ValueError("no sequence with at least three interactions was found")
    return Corpus(parsed, item_count, interactions)


def rating_to_reward(rating: float) -> float:
    if rating >= 4.0:
        return 1.0
    if rating == 3.0:
        return 0.5
    if rating == 2.0:
        return 0.0
    return -0.2


def logarithmic_gap_bucket(
    previous: float,
    current: float,
    timestamp_unit: str,
    bucket_count: int,
) -> int:
    difference = max(0.0, current - previous)
    days = difference / 86400.0 if timestamp_unit == "seconds" else difference
    return min(int(math.log2(days + 1.0)) + 1, bucket_count - 1)


class TrajectoryDataset(Dataset[Dict[str, torch.Tensor]]):
    """Dense training prefixes and one target per validation/test user.

    Validation uses the penultimate interaction and test uses the final
    interaction. The final two events never contribute to train samples,
    graph edges, or transition counts.
    """

    MODES = {"train", "validation", "test"}

    def __init__(
        self,
        corpus: Corpus,
        max_length: int,
        mode: str,
        timestamp_unit: str = "seconds",
        gap_buckets: int = 128,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        self.corpus = corpus
        self.max_length = int(max_length)
        self.mode = mode
        self.timestamp_unit = timestamp_unit
        self.gap_buckets = int(gap_buckets)
        self._records: List[Tuple[Tuple[Event, ...], int]] = []
        for sequence in corpus.sequences.values():
            train_length = len(sequence) - 2
            if mode == "train":
                targets = range(1, train_length)
            elif mode == "validation":
                targets = (train_length,)
            else:
                targets = (train_length + 1,)
            self._records.extend((sequence, target) for target in targets)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sequence, target_index = self._records[index]
        items = [event.item_id for event in sequence]
        ratings = [event.rating for event in sequence]
        rewards = [rating_to_reward(rating) for rating in ratings]

        gaps = [1]
        for previous, current in zip(sequence, sequence[1:]):
            gaps.append(
                logarithmic_gap_bucket(
                    previous.timestamp,
                    current.timestamp,
                    self.timestamp_unit,
                    self.gap_buckets,
                )
            )

        train_length = len(sequence) - 2
        running_return = sum(rewards[:train_length]) + 2.0
        returns = [running_return]
        for reward in rewards:
            running_return -= reward
            returns.append(running_return)

        start = max(0, target_index - self.max_length + 1)
        states = items[start:target_index]
        actions = items[start + 1 : target_index + 1]
        rtg = returns[start + 1 : target_index + 1]
        timesteps = list(range(start, target_index))
        gap_values = gaps[start:target_index]

        valid_length = len(states)
        padding = self.max_length - valid_length
        scale = float(self.max_length)

        def padded(
            values: Sequence[Union[float, int]], fill: Union[float, int]
        ) -> list:
            return list(values) + [fill] * padding

        return {
            "states": torch.tensor(padded(states, 0), dtype=torch.long),
            "actions": torch.tensor(padded(actions, 0), dtype=torch.long),
            "returns_to_go": torch.tensor(
                padded([value / scale for value in rtg], 0.0),
                dtype=torch.float32,
            ),
            "timesteps": torch.tensor(padded(timesteps, 0), dtype=torch.long),
            "gap_buckets": torch.tensor(padded(gap_values, 0), dtype=torch.long),
            "attention_mask": torch.tensor(
                [1] * valid_length + [0] * padding, dtype=torch.bool
            ),
        }
