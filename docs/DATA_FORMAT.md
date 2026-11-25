# Data Format

## Amazon sequence file

The pickle payload must contain:

```python
{
    "sequences": {
        user_id: [
            [timestamp, item_id, rating],
            ...
        ]
    },
    "itemmap": {raw_item_id: integer_item_id},
    "usermap": {raw_user_id: integer_user_id},
}
```

Events must be sorted by timestamp. Real item IDs are contiguous positive
integers; `0` is padding. Each retained user must have at least three events.

## MovieLens-1M

`trimrec prepare-ml1m` writes the same event schema plus
`split: "train_val_only"`. The last stored event is Val, not Test.
The original verified ML-1M ZIP is read by the explicit Test command to
reconstruct the held-out item IDs. Feature rows follow ascending rated movie
IDs; missing features do not remove items from the catalog.

## Strict leave-two-out

For a sequence of length `L`:

- training targets are positions `1` through `L-3`;
- validation predicts position `L-2`;
- test predicts position `L-1`.

The last two events are excluded when constructing graph edges and directed
transition statistics. During ranking, only item ID `0` is masked. Previously
observed items remain eligible because repeated consumption is valid.

## Reward and return-to-go

Ratings are mapped to rewards as follows:

| Rating | Reward |
|---:|---:|
| `>= 4` | `1.0` |
| `3` | `0.5` |
| `2` | `0.0` |
| otherwise | `-0.2` |

The initial return is the sum of training-prefix rewards plus `2.0`. Each
subsequent return subtracts the preceding reward. Returns are divided by the
maximum sequence length before being passed to the model. Neither held-out
rating belongs to the training sum. Test subtracts the observed Val reward
without re-anchoring. The Test rating is never a model input.

## Time gaps

Consecutive timestamp differences are converted to days and assigned to
`floor(log2(days + 1)) + 1`; bucket `0` is padding. Values are clipped to the
configured number of gap buckets. The first event in the original sequence
uses bucket 1; the first slot of a truncated window retains its original bucket.

## Frozen features

Text features may have shape `[N+1, 512]` or `[N+1, S, 512]`. The loader
canonicalizes the first form to a single stream. Image features have shape
`[N+1, 512]`. Row `0` must be all zeros. Missing images are represented by
zero-valued rows and remain explicitly distinguishable from encoded images.

The Amazon experiments use OpenAI CLIP ViT-B/32 embeddings. ML-1M uses
released M3L plot/poster CLIP vectors. Explicit candidate similarities use
cosine normalization.

