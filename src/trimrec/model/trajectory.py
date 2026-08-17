"""Return-State-Action trajectory encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryEncoder(nn.Module):
    """Causal encoder over interleaved return, state, and action tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        max_timestep: int,
        gap_buckets: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_timestep = max_timestep
        self.position_embedding = nn.Embedding(max_timestep, hidden_dim)
        self.return_projection = nn.Linear(1, hidden_dim)
        self.gap_embedding = nn.Embedding(gap_buckets, hidden_dim, padding_idx=0)
        nn.init.zeros_(self.gap_embedding.weight)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
        gap_buckets: torch.Tensor,
        item_table: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length = states.shape
        if timesteps.numel() and timesteps.max().item() >= self.max_timestep:
            raise ValueError("timestep exceeds the configured maximum")

        positions = self.position_embedding(timesteps)
        returns = self.return_projection(returns_to_go.unsqueeze(-1)) + positions
        state_tokens = F.embedding(states, item_table) + positions
        state_tokens = state_tokens + self.gap_embedding(gap_buckets)
        action_tokens = F.embedding(actions, item_table) + positions

        tokens = torch.stack((returns, state_tokens, action_tokens), dim=2)
        tokens = tokens.reshape(batch_size, 3 * sequence_length, -1)
        tokens = self.dropout(tokens)

        token_count = 3 * sequence_length
        causal_mask = torch.triu(
            torch.ones(
                token_count,
                token_count,
                dtype=torch.bool,
                device=states.device,
            ),
            diagonal=1,
        )
        padding_mask = (~attention_mask.bool()).repeat_interleave(3, dim=1)
        hidden = self.layers(
            tokens,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        return hidden[:, 1::3]
