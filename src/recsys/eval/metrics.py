from __future__ import annotations

import numpy as np


def recall_at_k(predicted: np.ndarray, ground_truth: list[int] | np.ndarray, k: int) -> float:
    pred = predicted[:k]
    gt = set(int(x) for x in ground_truth)
    if not gt:
        return 0.0
    hits = sum(1 for p in pred if int(p) in gt)
    return hits / len(gt)


def precision_at_k(predicted: np.ndarray, ground_truth: list[int] | np.ndarray, k: int) -> float:
    pred = predicted[:k]
    gt = set(int(x) for x in ground_truth)
    if k <= 0:
        return 0.0
    hits = sum(1 for p in pred if int(p) in gt)
    return hits / k


def ndcg_at_k(predicted: np.ndarray, ground_truth: list[int] | np.ndarray, k: int) -> float:
    pred = predicted[:k]
    gt = set(int(x) for x in ground_truth)
    if not gt:
        return 0.0
    dcg = 0.0
    for i, p in enumerate(pred):
        if int(p) in gt:
            dcg += 1.0 / np.log2(i + 2)
    ideal_hits = min(len(gt), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return float(dcg / idcg) if idcg > 0 else 0.0


def intra_list_diversity(rec_idxs: np.ndarray, item_features: np.ndarray) -> float:
    """Average pairwise (1 - cosine) over the recommended items' feature vectors."""
    if len(rec_idxs) < 2:
        return 0.0
    feats = item_features[rec_idxs]
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
    feats = feats / norms
    sim = feats @ feats.T
    n = len(rec_idxs)
    iu = np.triu_indices(n, k=1)
    return float(1.0 - sim[iu].mean())


def catalog_coverage(recommended_per_user: list[np.ndarray], n_items: int) -> float:
    seen: set[int] = set()
    for rec in recommended_per_user:
        seen.update(int(x) for x in rec)
    return len(seen) / max(n_items, 1)


def serendipity(
    predicted: np.ndarray,
    ground_truth: list[int] | np.ndarray,
    popularity_rank: np.ndarray,           # [n_items], lower rank = more popular
    k: int,
    pop_threshold: int = 500,
) -> float:
    """Recall on items that are NOT in the global top `pop_threshold`.

    A surprising-yet-relevant hit. Returns 0 if no such items in the holdout.
    """
    pred = predicted[:k]
    gt = set(int(x) for x in ground_truth)
    long_tail = {x for x in gt if popularity_rank[int(x)] >= pop_threshold}
    if not long_tail:
        return 0.0
    hits = sum(1 for p in pred if int(p) in long_tail)
    return hits / len(long_tail)
