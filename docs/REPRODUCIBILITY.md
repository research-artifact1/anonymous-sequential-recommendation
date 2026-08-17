# Reproducibility Protocol

## Evaluation contract

1. Users with fewer than three retained interactions are excluded.
2. The penultimate interaction is the validation target and the final
   interaction is the test target.
3. Graph and transition structures use training prefixes only.
4. Ranking is over the complete item catalog with padding-only masking.
5. Hyperparameters and epochs are selected exclusively by validation NDCG@20.
6. Test evaluation is a separate checkpoint-only command.
7. The reported runs use seeds `1`, `123`, `1337`, `2025`, and `2026`.

## Frozen inputs

The training command records SHA-256 fingerprints for the sequence file and
both feature tensors. Evaluation refuses a checkpoint when these fingerprints
do not match. Frozen feature tensors are not serialized into checkpoints.

## Determinism

Python, NumPy, CPU PyTorch, and all CUDA generators are seeded. Deterministic
algorithms are requested and cuDNN benchmarking is disabled. Exact
bit-for-bit agreement can still depend on the PyTorch, CUDA, driver, and GPU
versions; archive these versions with published results.

## Run artifacts

Each training directory contains:

- `config.json`: fully resolved settings;
- `inputs.json`: input fingerprints;
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

