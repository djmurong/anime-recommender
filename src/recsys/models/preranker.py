"""Lightweight cross-encoder used as the second stage of the cascade.

After FAISS retrieves ~1000 candidates the pre-ranker scores all of them with a
2-layer MLP over [user_emb, item_emb, user_emb * item_emb, |user_emb - item_emb|]
in one batched forward pass. It is intentionally cheap so we can afford to run
it over the full retrieve pool, even though only the top `pool_prerank`
survivors reach the heavy MMoE ranker.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreRanker(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        in_dim = embedding_dim * 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _pairwise_features(user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """Pack (u, i, u*i, |u-i|) along the last dim for the MLP."""
        if user_emb.dim() == 2:
            user_emb = user_emb.unsqueeze(1).expand_as(item_emb)
        return torch.cat(
            [user_emb, item_emb, user_emb * item_emb, (user_emb - item_emb).abs()],
            dim=-1,
        )

    def forward(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """Score (B, K) candidates per user.

        Args:
            user_emb: (B, D) or (B, K, D)
            item_emb: (B, K, D)
        Returns:
            (B, K) logits.
        """
        x = self._pairwise_features(user_emb, item_emb)
        return self.net(x).squeeze(-1)


def distill_targets_from_dot(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Soft targets for distilling the two-tower dot product into the PreRanker.

    Returns the per-row softmax over candidate scores. Use with KL divergence
    loss to train the PreRanker to approximate (and ideally refine) retrieval
    ordering before any expensive heavy ranker exists.
    """
    if user_emb.dim() == 2:
        user_emb = user_emb.unsqueeze(1)
    logits = (user_emb * item_emb).sum(dim=-1) / max(temperature, 1e-6)
    return F.softmax(logits, dim=-1)
