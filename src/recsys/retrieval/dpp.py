"""Determinantal Point Process re-ranker (Chen et al., 2018 "Fast Greedy MAP").

DPPs model the probability of selecting a *set* of items jointly: items that
are individually relevant but mutually similar pay a determinant penalty. The
greedy MAP algorithm picks one item at a time, each time maximizing the
marginal log-determinant gain, in O(k * pool_size) per user.

For anime this matters because MMR's greedy max-similarity penalty
deduplicates against the single most-similar already-picked item, while DPPs
penalize against the whole picked set -- empirically that avoids the
"you watched one shonen, here are 30 shonen" failure mode much more reliably.
"""
from __future__ import annotations

import numpy as np


def _build_kernel(
    candidate_emb: np.ndarray,
    candidate_quality: np.ndarray,
    theta: float = 0.5,
) -> np.ndarray:
    """Construct an L-ensemble kernel L = diag(q) S diag(q).

    `theta` blends quality and diversity:
        theta = 0  -> pure diversity (S unchanged)
        theta = 1  -> pure quality (q dominates; behavior approaches argmax)
    We pass `theta` as the "quality temperature" in the exp.
    """
    n = candidate_emb.shape[0]
    # Cosine similarity (candidates are already L2-normalized).
    s = candidate_emb @ candidate_emb.T
    # Clamp to a sane range so the kernel stays PSD after scaling.
    np.fill_diagonal(s, 1.0)
    s = np.clip(s, -1.0, 1.0)
    # Quality vector: rescale by theta so larger theta widens the dynamic range.
    q = np.exp(theta * candidate_quality.astype(np.float64))
    return (q[:, None] * s.astype(np.float64) * q[None, :]).astype(np.float64), q, s


def fast_greedy_map(
    L: np.ndarray,
    k: int,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """Return the indices of the greedy-MAP DPP selection of size k.

    Standard incremental Cholesky update (Chen et al., Algorithm 1).
    """
    n = L.shape[0]
    k = min(k, n)
    if k <= 0:
        return np.zeros(0, dtype=np.int64)

    cis = np.zeros((k, n), dtype=np.float64)
    di2 = np.copy(np.diag(L))
    selected: list[int] = []
    j = int(np.argmax(di2))
    selected.append(j)

    for it in range(1, k):
        di2j = di2[j]
        if di2j <= epsilon:
            break
        # Update conditional variances of remaining items.
        ei = (L[j] - cis[:it, j] @ cis[:it]) / max(np.sqrt(di2j), epsilon)
        cis[it] = ei
        di2 = di2 - ei * ei
        di2[selected] = -1.0  # never re-select
        j = int(np.argmax(di2))
        if di2[j] <= epsilon:
            break
        selected.append(j)
    return np.array(selected, dtype=np.int64)


def dpp_rerank(
    candidate_idxs: np.ndarray,           # [N] (catalog indices, used only by caller)
    candidate_scores: np.ndarray,         # [N] quality (e.g. ranker output)
    candidate_emb: np.ndarray,            # [N, D] L2-normalized
    k: int,
    theta: float = 0.5,
) -> np.ndarray:
    """Drop-in replacement for `mmr_rerank` using greedy-MAP DPP.

    Returns the top-k positions *into the candidate arrays* (NOT catalog
    indices), same contract as `mmr_rerank`.
    """
    n = len(candidate_idxs)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    # Min-max normalize quality to [0, 1] so theta is comparable across runs.
    q = candidate_scores.astype(np.float64)
    lo, hi = float(q.min()), float(q.max())
    q_norm = (q - lo) / (hi - lo + 1e-12)
    L, _, _ = _build_kernel(candidate_emb, q_norm, theta=theta)
    return fast_greedy_map(L, k=k)


def make_dpp_reranker(theta: float = 0.5):
    """Factory matching the `RerankerProtocol` used by `Cascade`."""

    def _fn(candidate_idxs, candidate_scores, candidate_emb, k):
        return dpp_rerank(
            candidate_idxs=candidate_idxs,
            candidate_scores=candidate_scores,
            candidate_emb=candidate_emb,
            k=k,
            theta=theta,
        )

    return _fn
