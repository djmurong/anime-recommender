from __future__ import annotations

import numpy as np

from recsys.retrieval.dpp import dpp_rerank


def mmr_rerank(
    candidate_idxs: np.ndarray,       # [N]
    candidate_scores: np.ndarray,     # [N]
    candidate_emb: np.ndarray,        # [N, D] L2-normalized
    k: int,
    lambda_: float = 0.7,
) -> np.ndarray:
    """Maximal Marginal Relevance: balance relevance vs intra-list diversity.

    Returns the top-k indices into the candidate arrays (NOT into the catalog).
    """
    n = len(candidate_idxs)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    k = min(k, n)
    sim = candidate_emb @ candidate_emb.T  # [N, N]
    selected = [int(np.argmax(candidate_scores))]
    remaining = set(range(n)) - set(selected)
    for _ in range(k - 1):
        if not remaining:
            break
        best = None
        best_score = -np.inf
        for j in remaining:
            redund = max(sim[j, s] for s in selected)
            score = lambda_ * candidate_scores[j] - (1 - lambda_) * redund
            if score > best_score:
                best_score = score
                best = j
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
    return np.array(selected, dtype=np.int64)


def epsilon_greedy_inject(
    final_idxs: np.ndarray,
    explore_pool: np.ndarray,
    k: int,
    epsilon: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace ~epsilon fraction of slots with random items from explore_pool.

    Returns (new_idxs[k], is_exploratory_mask[k]).
    """
    final_idxs = final_idxs[:k].copy()
    mask = np.zeros(len(final_idxs), dtype=bool)
    if epsilon <= 0 or len(explore_pool) == 0:
        return final_idxs, mask

    n_explore = max(1, int(round(k * epsilon))) if epsilon > 0 else 0
    if n_explore == 0 or n_explore > k:
        return final_idxs, mask

    slots = rng.choice(k, size=n_explore, replace=False)
    chosen = rng.choice(explore_pool, size=n_explore, replace=False)
    for s, c in zip(slots, chosen):
        final_idxs[s] = c
        mask[s] = True
    return final_idxs, mask
