"""Transformer history encoder with random time-window masking.

Drop-in replacement for the score-weighted mean pool in `TwoTowerModel.pool_history`.
The encoder consumes per-item embeddings from `AnimeEncoder` plus auxiliary
signals (history score, completion_fraction, time-delta bucket) and returns a
single user-history embedding via a learnable [CLS] token.

The "random time-window mask" trick (Pinterest 2022) drops the most-recent K
events during training with probability `p_mask_recent`. Without it the model
collapses onto the last few interactions and loses diversity at inference time.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Day cut-points for the bucketed time-delta embedding. Index i corresponds to
# "delta is between edges[i-1] and edges[i]" (with edge -inf at i=0 and +inf at last).
_TIME_BUCKET_EDGES_DAYS = (1.0, 3.0, 7.0, 14.0, 30.0, 90.0, 180.0, 365.0, 730.0)


def _bucketize_time_delta_days(days: torch.Tensor) -> torch.Tensor:
    """Map a float days tensor to bucket indices in [0, len(edges)]."""
    edges = torch.tensor(_TIME_BUCKET_EDGES_DAYS, device=days.device, dtype=days.dtype)
    return torch.bucketize(days, edges)


def random_time_window_mask(
    history_mask: torch.Tensor,
    days_ago: torch.Tensor,
    *,
    p_mask: float,
    window_min: int,
    window_max: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a new mask with the most-recent k events zeroed in each row.

    For each row independently with probability `p_mask`, sample
    k ~ Uniform[window_min, window_max] and zero out the k most-recent events
    (lowest `days_ago`). Rows shorter than `window_min` are not modified --
    aggressive masking on short histories starves the model.

    Args:
        history_mask: (B, L) float mask in {0, 1}.
        days_ago: (B, L) float days since each event relative to the target.
        p_mask: probability of applying the mask to a given row.
        window_min, window_max: bounds for the random window size (inclusive).
        generator: optional torch.Generator for reproducibility.
    """
    if p_mask <= 0.0:
        return history_mask
    b, length = history_mask.shape
    device = history_mask.device
    apply = torch.rand(b, generator=generator, device=device) < p_mask
    # Random window per row, clamped to history length.
    window = torch.randint(
        window_min, window_max + 1, (b,), generator=generator, device=device
    )
    # Each row's history is variable-length; days_ago for padding positions is
    # arbitrary, so we rely on history_mask to keep masked-pad positions zero.
    inf_filled = torch.where(history_mask > 0, days_ago, torch.full_like(days_ago, float("inf")))
    # rank by days_ago ascending -> position 0 is the most-recent real event
    order = inf_filled.argsort(dim=1)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(length, device=device).expand_as(order))

    drop = (rank < window.unsqueeze(1)) & (history_mask > 0)
    drop = drop & apply.unsqueeze(1)

    # Don't drop everything: if a row would lose all its events, keep them.
    remaining = (history_mask > 0).sum(dim=1) - drop.sum(dim=1)
    keep_row = remaining > 0
    drop = drop & keep_row.unsqueeze(1)

    new_mask = history_mask.clone()
    new_mask[drop] = 0.0
    return new_mask


class SequenceEncoder(nn.Module):
    """Small Transformer over the user's item-embedding history.

    Inputs (all (B, L) except `item_emb` which is (B, L, D_item)):
        item_emb       per-item embeddings from AnimeEncoder (already L2-normalized)
        history_mask   1 = real event, 0 = padding
        history_scores raw rating in [0, 10], 0 if unrated
        history_completion completion_fraction in [0, 1]
        days_ago       float days between the event and the target

    Output: (B, d_model) aggregated user-history embedding (NOT L2-normalized;
    that happens after concatenation with side features in UserTower).
    """

    def __init__(
        self,
        item_emb_dim: int,
        d_model: int,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_mult: int = 2,
        dropout: float = 0.1,
        max_history: int = 128,
        n_time_buckets: int = len(_TIME_BUCKET_EDGES_DAYS) + 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_history = max_history
        # We project the item embedding + 2 scalar signals (score, completion)
        # into d_model, then add learned positional and time-bucket embeddings.
        self.input_proj = nn.Linear(item_emb_dim + 2, d_model)
        self.pos_emb = nn.Embedding(max_history + 1, d_model)
        self.time_emb = nn.Embedding(n_time_buckets, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        item_emb: torch.Tensor,                  # (B, L, D_item)
        history_mask: torch.Tensor,              # (B, L) float {0, 1}
        history_scores: torch.Tensor,            # (B, L) raw scores
        history_completion: torch.Tensor,        # (B, L) in [0, 1]
        days_ago: torch.Tensor,                  # (B, L) float days
    ) -> torch.Tensor:
        b, length, _ = item_emb.shape
        # Standardize score to ~[-1, 1] -- 0 means unrated and stays 0.
        scores_n = (history_scores - 7.0) / 3.0
        scalars = torch.stack([scores_n, history_completion], dim=-1)
        x = torch.cat([item_emb, scalars], dim=-1)
        x = self.input_proj(x)

        # Positional indices follow the chronological order (already sorted by
        # the dataset). We additionally bucketize days_ago so the model can tell
        # "yesterday" from "a year ago" -- absolute position alone is ambiguous
        # because users have wildly different activity densities.
        positions = torch.arange(length, device=x.device).clamp(max=self.max_history - 1)
        x = x + self.pos_emb(positions).unsqueeze(0)
        time_buckets = _bucketize_time_delta_days(days_ago)
        x = x + self.time_emb(time_buckets)

        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, L+1, d_model)

        cls_mask = torch.ones((b, 1), device=history_mask.device, dtype=history_mask.dtype)
        full_mask = torch.cat([cls_mask, history_mask], dim=1)
        key_padding_mask = full_mask <= 0  # True where padded

        # If a row has no real events the encoder would output garbage; we force
        # the CLS to attend only to itself in that case (and the caller can
        # still gate on mask sum).
        empty_row = (full_mask.sum(dim=1) <= 1).unsqueeze(1)
        key_padding_mask = key_padding_mask & ~empty_row

        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        cls_out = self.out_norm(h[:, 0])
        return cls_out
