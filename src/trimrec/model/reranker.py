"""Candidate-conditioned multimodal refinement."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RerankerOutput:
    logits: torch.Tensor
    candidate_ids: torch.Tensor
    scores: torch.Tensor
    final_positions: torch.Tensor


class CandidateReranker(nn.Module):
    """Refine Top-M candidates with recent ID, text, and image evidence."""

    def __init__(
        self,
        hidden_dim: int,
        text_streams: int,
        top_m: int,
        history_length: int,
        attention_dim: int,
        scorer_hidden_dim: int,
        beta_max: float,
        mode: str = "cross_attention",
        candidate_modalities: bool = True,
        fp32_scorer: bool = False,
    ) -> None:
        super().__init__()
        self.top_m = top_m
        self.history_length = history_length
        self.beta_max = beta_max
        self.mode = mode
        self.candidate_modalities = candidate_modalities
        self.fp32_scorer = fp32_scorer
        feature_dim = history_length * (text_streams + 2 if candidate_modalities else 1)

        # A fixed local initialization stream keeps the refinement head
        # comparable across backbone seeds without changing global randomness.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17017)
            self.item_norm = nn.LayerNorm(hidden_dim)
            if mode == "cross_attention":
                self.query = nn.Linear(hidden_dim, attention_dim)
                self.key = nn.Linear(hidden_dim, attention_dim)
                self.value = nn.Linear(hidden_dim, attention_dim)
                self.context_norm = nn.LayerNorm(attention_dim)
                feature_dim += 3 * attention_dim
            self.scorer = nn.Sequential(
                nn.Linear(feature_dim, scorer_hidden_dim),
                nn.GELU(),
                nn.Linear(scorer_hidden_dim, 1),
            )
        self.beta_raw = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        logits: torch.Tensor,
        states: torch.Tensor,
        attention_mask: torch.Tensor,
        item_table: torch.Tensor,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
    ) -> RerankerOutput:
        batch_size, sequence_length = states.shape
        device = states.device
        batch_index = torch.arange(batch_size, device=device)
        final_positions = (
            (attention_mask.sum(1) - 1).long().clamp(min=0, max=sequence_length - 1)
        )

        retrieval_logits = logits[batch_index, final_positions].float().clone()
        retrieval_logits[:, 0] = -float("inf")
        pool_size = min(self.top_m, retrieval_logits.size(1) - 1)
        candidate_ids = torch.topk(
            retrieval_logits.detach(), k=pool_size, dim=1
        ).indices

        candidates = F.normalize(
            self.item_norm(F.embedding(candidate_ids, item_table)), dim=-1
        )
        offsets = torch.arange(self.history_length, device=device).view(1, -1)
        history_positions = final_positions[:, None] - offsets
        history_valid = history_positions >= 0
        history_positions = history_positions.clamp(min=0)
        history_ids = states.gather(1, history_positions)
        history_valid &= attention_mask.gather(1, history_positions).bool()
        history_valid &= history_ids > 0

        history = F.normalize(
            self.item_norm(F.embedding(history_ids, item_table)), dim=-1
        )
        history = history * history_valid.unsqueeze(-1).to(history.dtype)

        id_similarity = torch.einsum("bmd,bhd->bmh", candidates, history)
        id_similarity *= history_valid[:, None].to(id_similarity.dtype)

        features = [id_similarity]
        if self.candidate_modalities:
            candidate_text = F.normalize(text_features[candidate_ids], dim=-1)
            history_text = F.normalize(text_features[history_ids], dim=-1)
            text_sim = torch.einsum("bmsd,bhsd->bmhs", candidate_text, history_text)
            text_sim *= history_valid[:, None, :, None].to(text_sim.dtype)
            candidate_image = F.normalize(image_features[candidate_ids], dim=-1)
            history_image = F.normalize(image_features[history_ids], dim=-1)
            image_sim = torch.einsum("bmd,bhd->bmh", candidate_image, history_image)
            image_sim *= history_valid[:, None].to(image_sim.dtype)
            features.extend((text_sim.flatten(start_dim=2), image_sim))

        if self.mode == "cross_attention":
            query = self.query(candidates)
            key = self.key(history)
            value = self.value(history) * history_valid.unsqueeze(-1)
            attention_logits = torch.einsum("bma,bha->bmh", query, key)
            attention_logits = attention_logits / math.sqrt(float(query.size(-1)))
            attention_logits = attention_logits.masked_fill(
                ~history_valid[:, None], -1e4
            )
            attention = F.softmax(attention_logits, dim=-1)
            context = self.context_norm(torch.einsum("bmh,bha->bma", attention, value))
            features = [query, context, query * context, *features]

        matching = torch.cat(features, dim=-1)
        if self.fp32_scorer:
            with torch.autocast(device_type=matching.device.type, enabled=False):
                scores = self.scorer(matching.float()).squeeze(-1)
        else:
            scores = self.scorer(matching).squeeze(-1)
        scores = scores - scores.mean(dim=1, keepdim=True)

        beta = self.beta_max * torch.tanh(self.beta_raw)
        residual = torch.zeros_like(retrieval_logits)
        residual.scatter_add_(1, candidate_ids, beta * scores)
        refined = logits.clone()
        refined[batch_index, final_positions] += residual.to(refined.dtype)
        return RerankerOutput(refined, candidate_ids, scores, final_positions)
