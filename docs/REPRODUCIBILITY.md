# Reproducibility Protocol

## Experiment coverage

| Paper experiment | Coverage in this artifact |
|---|---|
| Table 2, TRIMRec | Fresh fit, Val selection and explicit Test on all five datasets. |
| Table 2, external baselines | Not bundled; their native/adapted runners and historical checkpoints are outside this release. |
| Table 3, component ablations | All ten interventions available via `--variant`. |
| Table 4, Early Fusion | Available for Beauty, Sports and Toys as `early_fusion`. |
| Table 4, recency transfer | The separate SASRec/adapted-IISAN training controls are not bundled. |
| Figures 2 and 3 | Val-only retrieval/refinement diagnostics on compatible checkpoints. Historical per-user scores and ranks are not distributed. |

The MIT license covers the project code, not external data and model weights.
No raw data, item-level features, private checkpoints, or run logs are bundled.

## Evaluation contract

1. Users with fewer than three retained interactions are excluded.
2. The penultimate interaction is the validation target and the final
   interaction is the test target.
3. Graph and transition structures use training prefixes only.
4. Ranking is over the complete item catalog with padding-only masking.
5. Hyperparameters and epochs are selected exclusively by validation NDCG@20.
6. Test evaluation is a separate checkpoint-only command.
7. The reported runs use seeds `1`, `123`, `1337`, `2025`, and `2026`.

Final ranking uses descending `torch.sort`, excluding only item ID 0. An item
in the observed history remains eligible. The terminal target Action is zeroed
before Val/Test model execution. Test uses the training-prefix Return anchor,
with a terminal return of `(2 - validation_reward) / max_length`. The test
target rating is not used.

## Frozen inputs

The training command records SHA-256 fingerprints for the sequence file and
both feature tensors. Evaluation refuses a checkpoint when these fingerprints
do not match. Frozen feature tensors are not serialized into checkpoints.

For Amazon, compare the reported fingerprints with
[AMAZON_ASSETS.json](AMAZON_ASSETS.json). The helper scripts cannot guarantee
the same encoded bytes across image snapshots, CLIP versions or serialization
environments.

ML-1M keeps all 3,706 rated items and 6,040 users. Its four source ZIPs are
pinned by SHA-256. The ML-10M ZIP is only an ID/title reference; no ML-10M
interactions are included. The exporter writes Train+Val data without the final
target. Explicit Test verifies the source archive, visible sequences, and
ordered target-ID fingerprint. Missing text features are masked after the
projection and normalization. The ML-1M config uses CPU COO propagation in
evaluation and an FP32 candidate MLP under CUDA AMP.

## Running the paper settings

Choose one domain config: `beauty`, `sports`, `toys`, `video_games`, or
`ml1m`. Train the five seeds; the following loop does not call Test.

```bash
domain=ml1m
variant=full
for seed in 1 123 1337 2025 2026; do
  trimrec train --config "configs/$domain.yaml" --variant "$variant" \
    --device cuda:0 --set "training.seed=$seed" || break
done
```

| Variant | Fixed intervention |
|---|---|
| `full` | Complete model. |
| `no_recency` | Uniform weighting at valid main-loss positions. |
| `no_refinement` | No candidate refinement or candidate auxiliary objective. |
| `no_item_graph` | No graph propagation; retain the transition prior. |
| `no_candidate_modal_evidence` | Remove the explicit candidate text and image similarities, retaining item-space attention and the shared text residual. |
| `neutral_rating` | Zero the shaped rewards; retain the Return token and its offset. |
| `no_time_gap` | Zero the State gap contribution; retain chronology. |
| `similarity_only` | MLP on item, text and image similarities without cross-attention. |
| `sa` | State-Action encoder without the Return token. |
| `no_transition` | Remove directed successor logits, retaining the graph. |
| `k1` | Single base preference; disable the multi-interest residual. |
| `early_fusion` | Add images to the shared table before graph propagation; omit candidate refinement and its auxiliary loss. |

Evaluate each selected checkpoint separately as shown in the README.
Aggregate per-seed NDCG@20 and HR@20 by arithmetic mean with sample standard
deviation. Match dataset and seed when computing paired differences. Do not
select hyperparameters from the Test outputs.

```bash
python scripts/summarize_runs.py --split test --output comparison.json \
  runs/selected/*/test.json
```

Supply five compatible files per dataset and variant. The script rejects
duplicate seeds, mixed splits, or mismatched dataset inputs and recipes.

For the validation diagnostic:

```bash
trimrec evaluate --config configs/beauty.yaml --device cuda:0 \
  --set training.seed=1 --checkpoint runs/<run>/best.pt \
  --split validation --compare-refinement --output runs/<run>/val_refinement.json
```

For each domain average post-minus-pre changes over five seeds at cutoffs
10, 20, 50, 100 and 200; show the per-seed min/max as the range. Sum promoted
and lost counts over all user-seed cases. For target-rank movement use the
natural candidate coverage count as the denominator.

## Determinism

Python, NumPy, CPU PyTorch, and all CUDA generators are seeded. Deterministic
algorithms are requested and cuDNN benchmarking is disabled. Exact
bit-for-bit agreement can still depend on the PyTorch, CUDA, driver, and GPU
versions; archive these versions with published results.

This is a standalone refactor: parameter names, training scaffolding and
metric reductions are not a byte-for-byte copy of the original runners.
Historical checkpoints cannot be loaded directly. In the reported ML-1M
repair, 45 affected Test runs reused frozen selections; 10 migrated runs used
exact local RTX 2080 Ti validation replay rather than exact reproduction of
archived RTX 4090 ranks. Neither hardware replay ledger is bundled here.
Resume restores model and optimizer at an epoch boundary; it does not
promise identical post-resume RNG.

## Run artifacts

Each training directory contains:

- `config.json`: fully resolved settings;
- `inputs.json`: input fingerprints;
- `environment.json`: Python, NumPy, PyTorch and device versions;
- `train.log`: human-readable events;
- `history.jsonl`: per-epoch losses, timing, and validation metrics;
- `best.pt`: checkpoint selected by Val NDCG@20;
- `last.pt`: optimizer-resumable latest state;
- `summary.json`: final validation selection record.

The test command writes a separate JSON result when `--output` is specified.

## Model size

Parameter counts include every learned tensor returned by `model.parameters()`.
They exclude frozen CLIP inputs and non-parametric graph/transition buffers. The
same policy must be applied to all methods in an efficiency comparison.

