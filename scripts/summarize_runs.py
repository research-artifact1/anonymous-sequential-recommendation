"""Summarize complete five-seed groups from public TRIMRec result JSONs."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

SEEDS = (1, 123, 1337, 2025, 2026)
METRICS = ("NDCG@20", "HR@20")
PROTOCOL = "leave_two_out_full_catalog_mask_0_torch_sort_desc_v1"


def summarize(paths, split):
    groups = defaultdict(dict)
    inputs = {}
    recipes = {}
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            row = json.load(stream)
        dataset, variant, seed = (row[k] for k in ("dataset", "variant", "seed"))
        versions = inputs.setdefault(dataset, row["input_sha256"])
        recipe = recipes.setdefault((dataset, variant), row["recipe_sha256"])
        if (
            row["split"] != split
            or row["protocol"] != PROTOCOL
            or row["input_sha256"] != versions
            or row["recipe_sha256"] != recipe
            or seed not in SEEDS
            or seed in groups[dataset, variant]
        ):
            raise ValueError(f"Conflicting split, input identity, or seed: {path.name}")
        values = {k: row["metrics"][k] for k in METRICS}
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in values.values()):
            raise ValueError(f"Invalid metric values: {path.name}")
        groups[dataset, variant][seed] = values

    for (dataset, variant), values in groups.items():
        if set(values) != set(SEEDS):
            raise ValueError(f"Incomplete five-seed group: {dataset}/{variant}")
    records = []
    for (dataset, variant), values in sorted(groups.items()):
        full = groups.get((dataset, "full"))
        summary = {}
        for metric in METRICS:
            samples = [values[seed][metric] for seed in SEEDS]
            summary[metric] = dict(mean=mean(samples), sample_sd=stdev(samples))
            if full and variant != "full":
                diffs = [full[seed][metric] - values[seed][metric] for seed in SEEDS]
                summary[metric]["full_minus_variant"] = dict(
                    mean=mean(diffs),
                    sample_sd=stdev(diffs),
                    wins=sum(d > 0 for d in diffs),
                    ties=sum(d == 0 for d in diffs),
                    losses=sum(d < 0 for d in diffs),
                )
        records.append(dict(dataset=dataset, variant=variant, metrics=summary))
    return dict(split=split, protocol=PROTOCOL, seeds=SEEDS, results=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.results, args.split)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
