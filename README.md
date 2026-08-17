# TRIMRec

TRIMRec is a sequential recommender that performs collaborative retrieval before
candidate-level multimodal refinement. It combines a Return-State-Action
trajectory encoder, training-prefix structural priors, multi-interest
full-catalog retrieval, and a candidate-conditioned reranker over frozen text
and image features.

This repository contains the final model and reproducibility pipeline. It does
not include baseline implementations, hyperparameter-search drivers, or
historical ablation branches.

## Method

TRIMRec has three stages:

1. **Trajectory-aware preference encoding.** Interleaved return, state, and
   action tokens are processed by a causal Transformer. Time-gap embeddings are
   added to state tokens.
2. **Interest-guided full-catalog retrieval.** A shared item table combines ID,
   frozen text, and train-prefix graph representations. Multiple preference
   directions and a directed transition prior produce the retrieval ranking.
3. **Candidate-conditioned multimodal refinement.** Each Top-200 candidate
   queries the recent history. ID, text, image, and cross-attention evidence
   produce a centered residual score for candidate reranking.

The objective is a recency-weighted next-item cross-entropy loss plus a
candidate-pool Top-K auxiliary loss. Validation NDCG@20 is the only checkpoint
selection signal.

## Installation

Python 3.9 or later and PyTorch 2.1 or later are required.

```bash
git clone <repository-url>
cd TRIMRec
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install the optional CLIP preprocessing dependencies when features must be
re-extracted:

```bash
pip install -e ".[clip]"
```

## Data

Each processed dataset uses:

```text
data/processed/<dataset>_sequences.pkl
data/processed/<dataset>_clip_vitb32_openai_text.pt
data/processed/<dataset>_clip_vitb32_openai_image.pt
```

The sequence file contains a `sequences` mapping. Every event is
`[timestamp, item_id, rating]`; item ID `0` is reserved for padding. Feature
tensors contain one padding row followed by item-aligned 512-dimensional CLIP
features. See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for the complete
contract.

Amazon review files can be prepared with:

```bash
python scripts/preprocess_amazon.py \
  --reviews /path/to/reviews_Beauty_5.json.gz \
  --output data/processed/Beauty_2014_sequences.pkl

python scripts/extract_clip_features.py \
  --sequences data/processed/Beauty_2014_sequences.pkl \
  --metadata /path/to/meta_Beauty.json.gz \
  --image-dir data/raw_images/Beauty_2014 \
  --text-output data/processed/Beauty_2014_clip_vitb32_openai_text.pt \
  --image-output data/processed/Beauty_2014_clip_vitb32_openai_image.pt
```

## Training

Run commands from the repository root. Training evaluates validation data only
and saves `best.pt` according to Val NDCG@20.

```bash
trimrec train --config configs/beauty.yaml --device cuda:0
```

Configuration values can be changed without editing YAML files:

```bash
trimrec train \
  --config configs/sports.yaml \
  --device cuda:1 \
  --set training.seed=123 \
  --set training.batch_size=128
```

Every run writes the resolved configuration, input SHA-256 fingerprints,
human-readable log, JSONL history, best checkpoint, resumable last checkpoint,
and validation summary to a timestamped directory under `runs/`.

## Evaluation

Test evaluation is an explicit checkpoint-only command. It is never performed
inside training. Load only checkpoints produced by a trusted training run;
PyTorch checkpoints use Python serialization.

```bash
trimrec evaluate \
  --config configs/beauty.yaml \
  --checkpoint runs/<run>/best.pt \
  --split test \
  --device cuda:0 \
  --output runs/<run>/test.json
```

Full-catalog HR and NDCG are reported at 5, 10, 20, 50, 100, and 200, together
with MRR@20. `RetrievalRecall@M` is the fraction of true next items present in
the pre-reranking Top-M pool.

## Final Hyperparameters

The architecture is fixed across datasets. Dataset-specific optimization
choices selected by validation NDCG@20 are:

| Dataset | Learning rate | Dropout | Batch size | Layers | Hidden size |
|---|---:|---:|---:|---:|---:|
| Beauty | 0.001 | 0.2 | 256 | 2 | 128 |
| Sports and Outdoors | 0.001 | 0.3 | 256 | 2 | 128 |
| Toys and Games | 0.001 | 0.3 | 256 | 2 | 128 |
| Video Games | 0.001 | 0.3 | 256 | 2 | 128 |

The paper reports five fixed seeds: `1`, `123`, `1337`, `2025`, and `2026`.
All remaining final parameters are recorded in
[docs/HYPERPARAMETERS.md](docs/HYPERPARAMETERS.md) and the YAML files.

## Parameter Counting

The reported model size is
`sum(parameter.numel() for parameter in model.parameters())`. Frozen CLIP
feature tensors, graph adjacency, and transition tables are external inputs and
are not model parameters. No checkpoint is needed to calculate the parameter
count; `trimrec inspect --config ...` constructs the architecture and reports
the count directly.

## Reproducibility

The strict protocol and artifact policy are documented in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). To run the test suite:

```bash
pip install -e ".[dev]"
pytest
```

## Citation

Please cite the accompanying paper when using this implementation. Publication
metadata can be updated in `CITATION.cff` after the archival record is issued.

## License

This project is released under the MIT License.
