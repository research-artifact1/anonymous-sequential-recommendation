"""Prepare MovieLens 1M and align the released M3L plot and poster vectors."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import struct
import unicodedata
from collections import defaultdict
from dataclasses import replace
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import torch

from trimrec.data import Corpus, Event
from trimrec.runtime import sha256_file

SOURCE_SHA256 = {
    "ml-1m.zip": "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20",
    "ml-10m.zip": "813c411ccb6122564edfe752e7f80c4dcc5aa25fa94c93622f6877a7ba252862",
    "TEXT_clip-text.zip": (
        "341298d0e70fc70435caf9bb81b5766076eb2fe0ee49a7c4a1fbbd213e426233"
    ),
    "IMG_clip-image.zip": (
        "5d5fac448d010d77643cb51ecc59ae725a4cd9d75e7c60a8dacb0c8b49338671"
    ),
}
TEST_TARGET_SHA256 = "4d3bff1a8c21751073cb2b160968b7f7f859854fde2ae89faa0bf69963d612ad"
COHORT = (6040, 3706, 1000209)


def _verify(path: Path, key: str) -> None:
    if sha256_file(path) != SOURCE_SHA256[key]:
        raise ValueError(f"Source checksum mismatch: {key}")


def _member(archive: ZipFile, name: str):
    matches = [
        x
        for x in archive.infolist()
        if not x.is_dir() and PurePosixPath(x.filename).name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Missing or ambiguous MovieLens member: {name}")
    return matches[0]


def _movies(archive: ZipFile, encoding: str) -> dict[int, str]:
    titles = {}
    with archive.open(_member(archive, "movies.dat")) as rows:
        for row in rows:
            prefix, _ = row.decode(encoding).rstrip("\r\n").rsplit("::", 1)
            raw_id, title = prefix.split("::", 1)
            movie = int(raw_id)
            if movie <= 0 or movie in titles or not title:
                raise ValueError("Malformed MovieLens catalog")
            titles[movie] = " ".join(
                unicodedata.normalize("NFC", title).casefold().split()
            )
    return titles


def _read_ratings(path: Path):
    users = defaultdict(list)
    items = set()
    with ZipFile(path) as archive:
        titles = _movies(archive, "latin-1")
        with archive.open(_member(archive, "ratings.dat")) as rows:
            for row in rows:
                user, movie, rating, time = map(int, row.split(b"::"))
                if (
                    user <= 0
                    or movie not in titles
                    or not 1 <= rating <= 5
                    or time <= 0
                ):
                    raise ValueError("Malformed MovieLens rating")
                users[user].append((time, movie, rating))
                items.add(movie)
    counts = (len(users), len(items), sum(map(len, users.values())))
    if counts != COHORT:
        raise ValueError(f"Unexpected ML-1M cohort: {counts}")
    for prefix in users.values():
        if len(prefix) < 3:
            raise ValueError("ML-1M leave-two-out requires three or more ratings")
        # Stable sort preserves original file order for timestamps that tie.
        prefix.sort(key=lambda event: event[0])
    return titles, users, {movie: i for i, movie in enumerate(sorted(items), 1)}


def _feature_table(path: Path, matches: set[int], itemmap: dict[int, int]):
    result = torch.zeros((len(itemmap) + 1, 512), dtype=torch.float32)
    seen = set()
    with ZipFile(path) as archive:
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            if name.suffix != ".json" or not name.stem.isdecimal():
                continue
            movie = int(name.stem)
            if movie not in matches:
                continue
            if movie in seen:
                raise ValueError("Duplicate MovieLens feature")
            seen.add(movie)
            vector = json.loads(archive.read(member))[str(movie)]
            if (
                not isinstance(vector, list)
                or len(vector) != 512
                or not all(type(v) in (float, int) and math.isfinite(v) for v in vector)
            ):
                raise ValueError("Invalid MovieLens feature")
            result[itemmap[movie]] = torch.tensor(vector, dtype=torch.float32)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite float32 MovieLens feature")
    return result


def prepare(ml1m: Path, ml10m: Path, text: Path, image: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    for path, name in zip((ml1m, ml10m, text, image), SOURCE_SHA256):
        _verify(path, name)
    titles, users, itemmap = _read_ratings(ml1m)
    with ZipFile(ml10m) as archive:
        reference = _movies(archive, "utf-8")
    matches = {movie for movie in itemmap if titles[movie] == reference.get(movie)}
    text_table = _feature_table(text, matches, itemmap)
    image_table = _feature_table(image, matches, itemmap)
    coverage = [int(table.ne(0).any(1).sum()) for table in (text_table, image_table)]
    if coverage != [3315, 3310]:
        raise ValueError(f"Unexpected CLIP coverage: {coverage}")
    usermap = {uid: i for i, uid in enumerate(sorted(users), 1)}
    sequences = {
        usermap[uid]: [
            (ts, itemmap[item], score) for ts, item, score in users[uid][:-1]
        ]
        for uid in usermap
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ml1m-", dir=output.parent) as work:
        stage = Path(work) / "prepared"
        stage.mkdir()
        with (stage / "train_val.pkl").open("wb") as stream:
            pickle.dump(
                dict(
                    split="train_val_only",
                    sequences=sequences,
                    usermap=usermap,
                    itemmap=itemmap,
                ),
                stream,
                protocol=4,
            )
        torch.save(text_table, stage / "text.pt")
        torch.save(image_table, stage / "image.pt")
        report = dict(
            users=COHORT[0],
            items=COHORT[1],
            interactions=COHORT[2],
            text_covered=coverage[0],
            image_covered=coverage[1],
            test_targets_written=False,
            source_sha256=SOURCE_SHA256,
            output_sha256={p.name: sha256_file(p) for p in stage.iterdir()},
        )
        (stage / "manifest.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    return report


def attach_test_targets(corpus: Corpus, source: Path) -> Corpus:
    """At explicit Test only, check the visible prefix and recover target IDs."""
    if not corpus.train_val_only:
        raise ValueError("Expected a Train+Val corpus")
    _verify(source, "ml-1m.zip")
    _, users, itemmap = _read_ratings(source)
    if len(itemmap) != corpus.item_count or len(users) != corpus.user_count:
        raise ValueError("ML-1M Test cohort differs")
    sequences = {}
    targets = hashlib.sha256()
    for index, user in enumerate(sorted(users), 1):
        prefix = corpus.sequences[index]
        raw = users[user]
        if len(prefix) != len(raw) - 1 or any(
            got != Event(ts, itemmap[movie], rating)
            for got, (ts, movie, rating) in zip(prefix, raw)
        ):
            raise ValueError("ML-1M visible history differs from the frozen prefix")
        target_id = itemmap[raw[-1][1]]
        targets.update(struct.pack("<q", target_id))
        sequences[index] = (*prefix, Event(prefix[-1].timestamp, target_id, 0.0))
    if targets.hexdigest() != TEST_TARGET_SHA256:
        raise ValueError("ML-1M target order differs from the frozen fingerprint")
    return replace(
        corpus,
        sequences=sequences,
        train_val_only=False,
        interaction_count=corpus.interaction_count + len(sequences),
    )
