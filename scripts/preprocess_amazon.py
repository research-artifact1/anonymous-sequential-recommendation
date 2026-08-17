"""Convert Amazon review JSON lines into TRIMRec sequence files."""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_records(path: Path) -> Iterator[Mapping[str, object]]:
    with open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                value = ast.literal_eval(line)
            if isinstance(value, Mapping):
                yield value


def first(record: Mapping[str, object], keys: Iterable[str]):
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-interactions", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output}")

    user_counts: Counter[str] = Counter()
    item_counts: Counter[str] = Counter()
    for record in iter_records(args.reviews):
        user = first(record, ("reviewerID", "user_id"))
        item = first(record, ("asin", "parent_asin"))
        if user is not None and item is not None:
            user_counts[str(user)] += 1
            item_counts[str(item)] += 1

    usermap: Dict[str, int] = {}
    itemmap: Dict[str, int] = {}
    sequences = defaultdict(list)
    for record in iter_records(args.reviews):
        user = first(record, ("reviewerID", "user_id"))
        item = first(record, ("asin", "parent_asin"))
        timestamp = first(record, ("unixReviewTime", "timestamp"))
        rating = first(record, ("overall", "rating"))
        if None in (user, item, timestamp, rating):
            continue
        user, item = str(user), str(item)
        if (
            user_counts[user] < args.min_interactions
            or item_counts[item] < args.min_interactions
        ):
            continue
        usermap.setdefault(user, len(usermap) + 1)
        itemmap.setdefault(item, len(itemmap) + 1)
        sequences[usermap[user]].append([int(timestamp), itemmap[item], float(rating)])

    for events in sequences.values():
        events.sort(key=lambda event: event[0])
    payload = {
        "sequences": dict(sequences),
        "itemmap": itemmap,
        "usermap": usermap,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, args.output)

    interactions = sum(map(len, sequences.values()))
    print(
        json.dumps(
            {
                "users": len(usermap),
                "items": len(itemmap),
                "interactions": interactions,
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
