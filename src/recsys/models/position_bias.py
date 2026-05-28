"""Position-bias side tower (YouTube 2019, ByteDance 2020).

When a recommender is trained on its own logged feedback the model conflates
item quality with the *position* at which the item was shown -- top-slot
items get more engagement regardless of relevance. A shallow side tower learns
the position-conditioned bias, gets ADDED to the ranker logit at training
time, and is DETACHED at serve so production scoring is bias-free.

This pipeline is offline-only today; we don't yet have logged exposure
positions. As an interim approximation we use **rank in the user's own
history when sorted by `my_last_updated`**: a row about to be ranked sees its
position context as "how recent is it among the user's interactions". Once
the system is deployed and we collect impression logs, swap the surrogate for
the actual displayed position without retraining the ranker head.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PositionBiasTower(nn.Module):
    """Maps position-context features -> additive logit shift.

    Inputs (all per-row):
        position_in_list: (B,) int -- 0-based rank in the slate this row was
            shown at (or surrogate during cold-start training).
        slate_size: (B,) int -- total slate length, log-normalized internally.
        surface_id: (B,) int -- which UI surface (home feed, search, etc.).
            Use 0 if you only have one surface today.

    The tower is intentionally tiny so it cannot model the full ranker problem
    -- if you give it features other than position it will leak quality signal
    and you'll over-debias.
    """

    def __init__(
        self,
        max_position: int = 128,
        n_surfaces: int = 4,
        hidden: int = 16,
    ):
        super().__init__()
        self.position_emb = nn.Embedding(max_position + 1, hidden)
        self.surface_emb = nn.Embedding(n_surfaces, hidden)
        self.net = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.max_position = max_position
        # Initialize to zero output so the tower doesn't shift logits until it
        # has seen data; this keeps early epochs stable.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        position_in_list: torch.Tensor,    # (B,) int
        slate_size: torch.Tensor,          # (B,) int
        surface_id: torch.Tensor,          # (B,) int
        *,
        training: bool = True,
    ) -> torch.Tensor:
        """Returns (B,) additive logit. Zero (no-op) when `training=False`.

        Keep `training=False` at inference; the bias has done its job during
        training and we serve a bias-free score.
        """
        if not training:
            return torch.zeros_like(position_in_list, dtype=torch.float32)
        pos = position_in_list.clamp(min=0, max=self.max_position)
        slate = (slate_size.clamp(min=1).float() + 1.0).log()
        x = torch.cat(
            [self.position_emb(pos), self.surface_emb(surface_id), slate.unsqueeze(-1)],
            dim=-1,
        )
        return self.net(x).squeeze(-1)
