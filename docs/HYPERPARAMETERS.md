# Final Hyperparameters

## Dataset-specific settings

| Dataset | Learning rate | Dropout |
|---|---:|---:|
| Beauty | 0.001 | 0.2 |
| Sports and Outdoors | 0.001 | 0.3 |
| Toys and Games | 0.001 | 0.3 |
| Video Games | 0.001 | 0.3 |

## Shared settings

| Group | Parameter | Value |
|---|---|---:|
| Backbone | hidden size | 128 |
| Backbone | attention heads | 2 |
| Backbone | layers | 2 |
| Backbone | maximum sequence length | 50 |
| Backbone | token layout | Return-State-Action |
| Optimizer | optimizer | Adam |
| Optimizer | batch size | 256 |
| Optimizer | weight decay | 0 |
| Optimizer | maximum epochs | 200 |
| Optimizer | early-stop patience | 20 |
| Selection | metric | Val NDCG@20 |
| Graph | propagation layers | 1 |
| Graph | maximum residual coefficient | 0.5 |
| Graph | minimum edge count | 2 |
| Graph | neighbors per item | 20 |
| Multi-interest | interest count | 4 |
| Multi-interest | pooling | max |
| Multi-interest | maximum residual coefficient | 0.5 |
| Transition | successors per item | 20 |
| Transition | minimum count | 2 |
| Transition | maximum residual coefficient | 3.0 |
| Reranker | retrieval pool | 200 |
| Reranker | recent history length | 8 |
| Reranker | attention dimension | 64 |
| Reranker | scorer hidden dimension | 64 |
| Reranker | maximum residual coefficient | 3.0 |
| Objective | recency schedule | 0.8 to 0.5 over 10 epochs |
| Objective | candidate auxiliary weight | 0.001 |
| Objective | hard negatives in auxiliary loss | 20 |

Reported seeds: `1`, `123`, `1337`, `2025`, and `2026`.
