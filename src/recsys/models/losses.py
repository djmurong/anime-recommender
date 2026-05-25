from __future__ import annotations

import torch
import torch.nn.functional as F


def sampled_softmax_loss(
    user_emb: torch.Tensor,
    pos_anime_emb: torch.Tensor,
    log_q: torch.Tensor | None,
    temperature: float,
    extra_neg_emb: torch.Tensor | None = None,
) -> torch.Tensor:
    """In-batch sampled softmax with optional log-Q correction and per-row hard negatives."""
    logits = (user_emb @ pos_anime_emb.t()) / temperature

    if log_q is not None:
        logits = logits - log_q.unsqueeze(0)

    if extra_neg_emb is not None and extra_neg_emb.numel() > 0:
        hard_logits = torch.einsum("bd,bkd->bk", user_emb, extra_neg_emb) / temperature
        logits = torch.cat([logits, hard_logits], dim=1)

    targets = torch.arange(user_emb.size(0), device=user_emb.device)
    return F.cross_entropy(logits, targets)


def combined_ranking_loss(
    user_emb: torch.Tensor,
    pos_anime_emb: torch.Tensor,
    log_q: torch.Tensor | None,
    temperature: float,
    catalog_neg_emb: torch.Tensor | None = None,
    catalog_neg_weight: float = 1.0,
    extra_neg_emb: torch.Tensor | None = None,
) -> torch.Tensor:
    """In-batch softmax + optional catalog negatives + hard negatives."""
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
    return F.cross_entropy(logits, targets)


def bpr_loss(
    user_emb: torch.Tensor,
    pos_anime_emb: torch.Tensor,
    neg_anime_emb: torch.Tensor,
) -> torch.Tensor:
    """Bayesian Personalized Ranking — kept as an alternative to sampled softmax."""
    pos = (user_emb * pos_anime_emb).sum(dim=-1)
    neg = (user_emb * neg_anime_emb).sum(dim=-1)
    return -F.logsigmoid(pos - neg).mean()
