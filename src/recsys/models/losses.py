from __future__ import annotations

import torch
import torch.nn.functional as F


def _weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None,
) -> torch.Tensor:
    if weights is None:
        return F.cross_entropy(logits, targets)
    per_row = F.cross_entropy(logits, targets, reduction="none")
    w = weights.to(per_row.dtype)
    denom = w.sum().clamp(min=1e-6)
    return (per_row * w).sum() / denom


def sampled_softmax_loss(
    user_emb: torch.Tensor,
    pos_anime_emb: torch.Tensor,
    log_q: torch.Tensor | None,
    temperature: float,
    extra_neg_emb: torch.Tensor | None = None,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """In-batch sampled softmax with optional log-Q correction and per-row hard negatives.

    `pos_weight` (B,) up-weights rows representing high-completion / high-rating
    positives so we optimize for *good* watches, not just any click.
    """
    logits = (user_emb @ pos_anime_emb.t()) / temperature

    if log_q is not None:
        logits = logits - log_q.unsqueeze(0)

    if extra_neg_emb is not None and extra_neg_emb.numel() > 0:
        hard_logits = torch.einsum("bd,bkd->bk", user_emb, extra_neg_emb) / temperature
        logits = torch.cat([logits, hard_logits], dim=1)

    targets = torch.arange(user_emb.size(0), device=user_emb.device)
    return _weighted_cross_entropy(logits, targets, pos_weight)


def combined_ranking_loss(
    user_emb: torch.Tensor,
    pos_anime_emb: torch.Tensor,
    log_q: torch.Tensor | None,
    temperature: float,
    catalog_neg_emb: torch.Tensor | None = None,
    catalog_neg_weight: float = 1.0,
    extra_neg_emb: torch.Tensor | None = None,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """In-batch softmax + optional catalog negatives + hard negatives.

    Completion weighting: when `pos_weight` is supplied (B,) the per-row
    cross-entropy is multiplied by it before reduction. Plan-to-watch positives
    pass weight 0 here so they only act as in-batch negatives for other users
    -- the model never gets a reward for ranking them highly. High-completion +
    high-rating positives get the largest weights, which is the anime-native
    equivalent of YouTube's watch-time-weighted objective.

    log-Q correction (`log_q`) is also applied: subtracting log P(item appears
    in batch) from the diagonal logits cancels the popularity bias introduced
    by in-batch sampled softmax.
    """
    logits = (user_emb @ pos_anime_emb.t()) / temperature
    if log_q is not None:
        logits = logits - log_q.unsqueeze(0)

    parts = [logits]
    if catalog_neg_emb is not None and catalog_neg_emb.numel() > 0:
        cat_logits = torch.einsum("bd,bkd->bk", user_emb, catalog_neg_emb) / temperature
        parts.append(cat_logits * catalog_neg_weight)
    if extra_neg_emb is not None and extra_neg_emb.numel() > 0:
        hard_logits = torch.einsum("bd,bkd->bk", user_emb, extra_neg_emb) / temperature
        parts.append(hard_logits)

    logits = torch.cat(parts, dim=1)
    targets = torch.arange(user_emb.size(0), device=user_emb.device)
    return _weighted_cross_entropy(logits, targets, pos_weight)


def bpr_loss(
    user_emb: torch.Tensor,
    pos_anime_emb: torch.Tensor,
    neg_anime_emb: torch.Tensor,
) -> torch.Tensor:
    """Bayesian Personalized Ranking — kept as an alternative to sampled softmax."""
    pos = (user_emb * pos_anime_emb).sum(dim=-1)
    neg = (user_emb * neg_anime_emb).sum(dim=-1)
    return -F.logsigmoid(pos - neg).mean()
