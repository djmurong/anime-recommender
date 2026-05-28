"""Multi-gate Mixture-of-Experts ranker.

Stage 3 of the cascade. The two-tower retriever already cut the candidate set
to ~200 with PreRank; the MMoE ranker spends real compute on each survivor and
predicts three anime-native targets:

  * `completion`   — sigmoid, predicts completion_fraction in [0, 1].
  * `rating`       — linear, predicts the user-centered rating (rating_z).
  * `drop`         — sigmoid, predicts P(my_status == 4 | u, i).

A 4-head MMoE bottom (Ma et al., 2018) lets the heads share a representation
while each gets its own gate over experts. The final serve score blends head
outputs with weights from `RetrievalConfig.mmoe_w_*` so we can tune the
exploitation/safety tradeoff (negative drop-weight = penalize likely drops)
without retraining.

The Phase 2c "season 3 reach" head was dropped because the anime CSV has no
sequel graph; that head can be added later if a related-anime feed is wired in.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MMoEConfig:
    embedding_dim: int = 128
    bottom_hidden: int = 256
    expert_hidden: int = 128
    n_experts: int = 4
    n_heads: int = 3
    side_dim: int = 0
    dropout: float = 0.1


class _Expert(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MMoERanker(nn.Module):
    """Shared-bottom + experts + per-task gates + 3 task heads.

    Input wiring (concatenated along the last dim):
        user_emb           (B, D)
        item_emb           (B, D)
        user_emb * item_emb (B, D)
        |user_emb - item_emb| (B, D)
        side_feats         (B, S)  -- optional (genre overlap, popularity, etc.)
    """

    def __init__(self, cfg: MMoEConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.embedding_dim * 4 + cfg.side_dim
        self.bottom = nn.Sequential(
            nn.Linear(in_dim, cfg.bottom_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        self.experts = nn.ModuleList(
            [
                _Expert(cfg.bottom_hidden, cfg.expert_hidden, cfg.expert_hidden, cfg.dropout)
                for _ in range(cfg.n_experts)
            ]
        )
        self.gates = nn.ModuleList(
            [nn.Linear(cfg.bottom_hidden, cfg.n_experts) for _ in range(cfg.n_heads)]
        )

        self.completion_head = nn.Linear(cfg.expert_hidden, 1)
        self.rating_head = nn.Linear(cfg.expert_hidden, 1)
        self.drop_head = nn.Linear(cfg.expert_hidden, 1)

        # Learnable task uncertainties (Kendall & Gal 2018). Stored as log-variance
        # so they stay strictly positive; the loss adds the log-variance term so
        # the model is penalized for inflating uncertainty to dodge a hard task.
        self.log_var_completion = nn.Parameter(torch.zeros(1))
        self.log_var_rating = nn.Parameter(torch.zeros(1))
        self.log_var_drop = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _pairwise(user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [user_emb, item_emb, user_emb * item_emb, (user_emb - item_emb).abs()],
            dim=-1,
        )

    def _forward_features(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        side_feats: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if user_emb.dim() == 2 and item_emb.dim() == 3:
            user_emb = user_emb.unsqueeze(1).expand_as(item_emb)
        pair = self._pairwise(user_emb, item_emb)
        if side_feats is not None:
            pair = torch.cat([pair, side_feats], dim=-1)
        bottom = self.bottom(pair)

        # All-experts forward, then gate-weighted mix per head.
        # Stack experts: (B, [K,] n_experts, expert_hidden).
        exp_out = torch.stack([e(bottom) for e in self.experts], dim=-2)
        head_outputs: list[torch.Tensor] = []
        for gate in self.gates:
            g = F.softmax(gate(bottom), dim=-1)
            mixed = (exp_out * g.unsqueeze(-1)).sum(dim=-2)
            head_outputs.append(mixed)
        return head_outputs

    def forward(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        side_feats: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Returns a dict of head logits/scores.

        Keys: 'completion' (sigmoid logit), 'rating' (linear), 'drop' (sigmoid logit).
        Shapes are (B,) or (B, K) depending on item_emb dimensionality.
        """
        completion_feat, rating_feat, drop_feat = self._forward_features(
            user_emb, item_emb, side_feats
        )
        completion = self.completion_head(completion_feat).squeeze(-1)
        rating = self.rating_head(rating_feat).squeeze(-1)
        drop = self.drop_head(drop_feat).squeeze(-1)
        return {"completion": completion, "rating": rating, "drop": drop}

    def loss(
        self,
        preds: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        weights: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Multi-task loss with Kendall-Gal uncertainty weighting.

        targets keys:
            completion: (B,) in [0, 1]
            rating:     (B,) centered ratings (any real)
            drop:       (B,) in {0, 1}
        weights keys (optional): same names. If provided, each per-row loss is
        multiplied by the corresponding weight before averaging.
        """
        comp_t = targets["completion"]
        rating_t = targets["rating"]
        drop_t = targets["drop"]

        def _w(name: str, per_row: torch.Tensor) -> torch.Tensor:
            if weights is None or name not in weights:
                return per_row.mean()
            w = weights[name].to(per_row.dtype)
            denom = w.sum().clamp(min=1e-6)
            return (per_row * w).sum() / denom

        comp_per = F.binary_cross_entropy_with_logits(
            preds["completion"], comp_t, reduction="none"
        )
        rating_per = F.mse_loss(preds["rating"], rating_t, reduction="none")
        drop_per = F.binary_cross_entropy_with_logits(
            preds["drop"], drop_t, reduction="none"
        )

        comp_loss = _w("completion", comp_per)
        rating_loss = _w("rating", rating_per)
        drop_loss = _w("drop", drop_per)

        precision = lambda lv: torch.exp(-lv)
        total = (
            precision(self.log_var_completion) * comp_loss
            + 0.5 * self.log_var_completion
            + precision(self.log_var_rating) * rating_loss
            + 0.5 * self.log_var_rating
            + precision(self.log_var_drop) * drop_loss
            + 0.5 * self.log_var_drop
        ).squeeze()

        report = {
            "loss/completion": float(comp_loss.detach()),
            "loss/rating": float(rating_loss.detach()),
            "loss/drop": float(drop_loss.detach()),
            "loss/total": float(total.detach()),
        }
        return total, report

    @torch.no_grad()
    def serve_score(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        side_feats: torch.Tensor | None = None,
        w_completion: float = 0.5,
        w_rating: float = 0.3,
        w_drop: float = -0.2,
    ) -> torch.Tensor:
        """Blend heads into a single per-candidate score.

        Default weights match the plan: reward predicted completion + rating,
        penalize predicted drop probability. Probabilities are squashed via
        sigmoid so the magnitudes are comparable across heads.
        """
        out = self.forward(user_emb, item_emb, side_feats)
        return (
            w_completion * torch.sigmoid(out["completion"])
            + w_rating * out["rating"]
            + w_drop * torch.sigmoid(out["drop"])
        )


class MMoEServeFn:
    """Callable wrapper matching `Cascade.RankerProtocol`."""

    def __init__(
        self,
        model: MMoERanker,
        side_feats_fn=None,
        w_completion: float = 0.5,
        w_rating: float = 0.3,
        w_drop: float = -0.2,
    ):
        self.model = model
        self.side_feats_fn = side_feats_fn
        self.w_completion = w_completion
        self.w_rating = w_rating
        self.w_drop = w_drop

    def __call__(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        side = self.side_feats_fn(user_emb, item_emb) if self.side_feats_fn is not None else None
        return self.model.serve_score(
            user_emb,
            item_emb,
            side_feats=side,
            w_completion=self.w_completion,
            w_rating=self.w_rating,
            w_drop=self.w_drop,
        )
