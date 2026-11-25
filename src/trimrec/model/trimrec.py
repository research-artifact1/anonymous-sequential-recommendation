"""TRIMRec model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from trimrec.config import ModelConfig
from trimrec.features import FrozenFeatures
from trimrec.model.reranker import CandidateReranker
from trimrec.model.trajectory import TrajectoryEncoder


def bounded_residual(raw: torch.Tensor, maximum: float) -> torch.Tensor:
    return float(maximum) * torch.tanh(raw)


@dataclass(frozen=True)
class TRIMRecOutput:
    logits: torch.Tensor
    candidate_ids: torch.Tensor
    reranker_scores: torch.Tensor
    final_positions: torch.Tensor
    retrieval_logits: torch.Tensor | None = None


class TRIMRec(nn.Module):
    """Trajectory-aware retrieval and multimodal candidate refinement.

    Item ID 0 is padding and real item IDs are in ``[1, item_count]``.
    Frozen CLIP features and data-derived structures are non-persistent
    buffers: checkpoints contain learned parameters only and reconstruct
    external inputs from the experiment configuration.
    """

    def __init__(
        self,
        config: ModelConfig,
        item_count: int,
        features: FrozenFeatures,
        graph_adjacency: torch.Tensor,
        transition_indices: torch.Tensor,
        transition_values: torch.Tensor,
    ) -> None:
        super().__init__()
        self.config = config
        self.item_count = item_count
        catalog_size = item_count + 1
        hidden_dim = config.hidden_dim
        self._validate_inputs(
            features,
            graph_adjacency,
            transition_indices,
            transition_values,
            catalog_size,
        )

        self.register_buffer("text_features", features.text, persistent=False)
        self.register_buffer("image_features", features.image, persistent=False)
        self.register_buffer(
            "graph_adjacency", graph_adjacency.coalesce(), persistent=False
        )
        self._cpu_eval_graph = (
            self.graph_adjacency.cpu() if config.cpu_graph_eval else None
        )
        self.register_buffer(
            "transition_indices", transition_indices.long(), persistent=False
        )
        self.register_buffer(
            "transition_values", transition_values.float(), persistent=False
        )

        self.item_embedding = nn.Embedding(catalog_size, hidden_dim, padding_idx=0)
        self.output_bias = nn.Parameter(torch.zeros(catalog_size))

        self.text_id_norm = nn.LayerNorm(hidden_dim)
        self.text_projections = nn.ModuleList(
            nn.Linear(config.text_dim, hidden_dim) for _ in range(config.text_streams)
        )
        self.text_norms = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in range(config.text_streams)
        )
        self.text_alpha = nn.Parameter(torch.zeros(config.text_streams))

        self.graph_norm = nn.LayerNorm(hidden_dim)
        self.graph_gamma_raw = nn.Parameter(torch.zeros(1))

        self.interest_norm = nn.LayerNorm(hidden_dim)
        self.interest_dropout = nn.Dropout(config.dropout)
        self.interest_projection = (
            nn.Linear(hidden_dim, config.num_interests * hidden_dim)
            if config.num_interests > 1
            else None
        )
        self.interest_gate_raw = nn.Parameter(torch.zeros(1))
        self.transition_beta_raw = nn.Parameter(torch.zeros(1))

        self.reranker = (
            None
            if config.refinement == "none"
            else CandidateReranker(
                hidden_dim=hidden_dim,
                text_streams=config.text_streams,
                top_m=config.reranker_top_m,
                history_length=config.reranker_history_length,
                attention_dim=config.reranker_attention_dim,
                scorer_hidden_dim=config.reranker_hidden_dim,
                beta_max=config.reranker_beta_max,
                mode=config.refinement,
                candidate_modalities=config.candidate_modalities,
                fp32_scorer=config.fp32_scorer,
            )
        )

        # Keep the initialization order used by the reported experiments:
        # item/text modules, multi-interest projection, then RSA encoder.
        # The reranker uses an isolated RNG stream and does not alter this order.
        self.encoder = TrajectoryEncoder(
            hidden_dim=hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_timestep=config.max_timestep,
            gap_buckets=config.gap_buckets,
            dropout=config.dropout,
            use_return_token=config.use_return_token,
            use_time_gap=config.use_time_gap,
        )
        if config.early_image_fusion:
            with torch.random.fork_rng(devices=[]):
                self.image_projection = nn.Linear(config.image_dim, hidden_dim)
                self.image_norm = nn.LayerNorm(hidden_dim)
            self.image_gate = nn.Parameter(torch.zeros(1))

    def _validate_inputs(
        self,
        features: FrozenFeatures,
        graph: torch.Tensor,
        transition_indices: torch.Tensor,
        transition_values: torch.Tensor,
        catalog_size: int,
    ) -> None:
        expected_text = (
            catalog_size,
            self.config.text_streams,
            self.config.text_dim,
        )
        expected_image = (catalog_size, self.config.image_dim)
        if tuple(features.text.shape) != expected_text:
            raise ValueError(
                f"text feature shape must be {expected_text}, "
                f"got {tuple(features.text.shape)}"
            )
        if tuple(features.image.shape) != expected_image:
            raise ValueError(
                f"image feature shape must be {expected_image}, "
                f"got {tuple(features.image.shape)}"
            )
        if tuple(graph.shape) != (catalog_size, catalog_size):
            raise ValueError("graph shape does not match the catalog")
        if transition_indices.shape != transition_values.shape:
            raise ValueError("transition index and value tables must have equal shape")
        if transition_indices.size(0) != catalog_size:
            raise ValueError("transition tables do not match the catalog")

    def build_item_table(self) -> torch.Tensor:
        item_table = self.text_id_norm(self.item_embedding.weight)
        for stream in range(self.config.text_streams):
            source = self.text_features[:, stream]
            text = self.text_norms[stream](self.text_projections[stream](source))
            if self.config.mask_missing_text_residual:
                text = text * source.ne(0).any(dim=-1, keepdim=True)
            item_table = item_table + self.text_alpha[stream] * text
        if self.config.early_image_fusion:
            image = self.image_norm(self.image_projection(self.image_features))
            available = self.image_features.ne(0).any(dim=-1, keepdim=True)
            item_table = item_table + self.image_gate * image * available
        item_table = item_table.clone()
        item_table[0] = 0.0
        if not self.config.use_graph:
            return item_table

        output_dtype = item_table.dtype
        device_type = item_table.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            propagated = item_table.float()
            graph_outputs = []
            for _ in range(self.config.graph_layers):
                if not self.training and self._cpu_eval_graph is not None:
                    propagated = torch.sparse.mm(
                        self._cpu_eval_graph.float(), propagated.cpu()
                    ).to(item_table.device)
                else:
                    propagated = torch.sparse.mm(
                        self.graph_adjacency.float(), propagated
                    )
                graph_outputs.append(propagated)
            graph_context = self.graph_norm(torch.stack(graph_outputs).mean(0))
            gamma = bounded_residual(
                self.graph_gamma_raw.float(), self.config.graph_gamma_max
            )
            item_table = item_table.float() + gamma * graph_context
        item_table = item_table.to(output_dtype)
        item_table = item_table.clone()
        item_table[0] = 0.0
        return item_table

    def _apply_multi_interest(
        self,
        logits: torch.Tensor,
        state_hidden: torch.Tensor,
        item_table: torch.Tensor,
    ) -> torch.Tensor:
        if self.interest_projection is None:
            return logits
        batch_size, sequence_length, hidden_dim = state_hidden.shape
        deltas = self.interest_projection(
            self.interest_dropout(self.interest_norm(state_hidden))
        ).view(
            batch_size,
            sequence_length,
            self.config.num_interests,
            hidden_dim,
        )
        interests = state_hidden.unsqueeze(2) + deltas
        # Compute one catalog score tensor at a time. Materializing
        # [batch, length, interests, catalog] exceeds practical GPU memory on
        # the larger datasets and is unnecessary for max pooling.
        pooled = None
        for interest in range(self.config.num_interests):
            score = F.linear(interests[:, :, interest], item_table)
            pooled = score if pooled is None else torch.maximum(pooled, score)
        assert pooled is not None
        base_without_bias = logits - self.output_bias.view(1, 1, -1)
        gate = bounded_residual(self.interest_gate_raw, self.config.interest_gate_max)
        return logits + gate * (pooled - base_without_bias)

    def _apply_transition_prior(
        self, logits: torch.Tensor, states: torch.Tensor
    ) -> torch.Tensor:
        candidate_ids = self.transition_indices[states]
        candidate_values = self.transition_values[states]
        beta = bounded_residual(
            self.transition_beta_raw, self.config.transition_beta_max
        )
        return logits.scatter_add(
            2,
            candidate_ids,
            beta.to(logits.dtype) * candidate_values.to(logits.dtype),
        )

    def forward(
        self, batch: dict[str, torch.Tensor], *, include_retrieval: bool = False
    ) -> TRIMRecOutput:
        item_table = self.build_item_table()
        state_hidden = self.encoder(
            states=batch["states"],
            actions=batch["actions"],
            returns_to_go=batch["returns_to_go"],
            timesteps=batch["timesteps"],
            attention_mask=batch["attention_mask"],
            gap_buckets=batch["gap_buckets"],
            item_table=item_table,
        )
        logits = F.linear(state_hidden, item_table, self.output_bias)
        logits = self._apply_multi_interest(logits, state_hidden, item_table)
        if self.config.use_transition:
            logits = self._apply_transition_prior(logits, batch["states"])
        if self.reranker is None:
            terminal = batch["attention_mask"].sum(1).long() - 1
            empty_ids = batch["states"].new_empty((logits.size(0), 0))
            return TRIMRecOutput(
                logits, empty_ids, logits.new_empty(empty_ids.shape), terminal
            )
        reranked = self.reranker(
            logits=logits,
            states=batch["states"],
            attention_mask=batch["attention_mask"],
            item_table=item_table,
            text_features=self.text_features,
            image_features=self.image_features,
        )
        return TRIMRecOutput(
            logits=reranked.logits,
            candidate_ids=reranked.candidate_ids,
            reranker_scores=reranked.scores,
            final_positions=reranked.final_positions,
            retrieval_logits=(
                logits[
                    torch.arange(logits.size(0), device=logits.device),
                    reranked.final_positions,
                ]
                if include_retrieval
                else None
            ),
        )

    def parameter_count(self, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (
                parameter for parameter in parameters if parameter.requires_grad
            )
        return sum(parameter.numel() for parameter in parameters)
