from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


RECENCY_BUCKETS = 5


@dataclass
class UserFeaturePack:
    """Per-user precomputed features needed by the user tower at training time."""
    history: dict[int, np.ndarray]            # user_idx -> int64 array of anime_idx (positives in train)
    history_scores: dict[int, np.ndarray]     # user_idx -> float32 array of scores
    genre_affinity: np.ndarray                # [n_users, n_genres] float32
    centered_avg_score: np.ndarray            # [n_users] float32
    recency: np.ndarray                       # [n_users, RECENCY_BUCKETS] float32 one-hot


def _bucketize_recency(days_since: np.ndarray, n_buckets: int = RECENCY_BUCKETS) -> np.ndarray:
    """Bucketize 'days since last activity' into a one-hot. Edges at log-scale."""
    edges = np.array([7, 30, 90, 365], dtype=np.float32)  # 5 buckets
    idx = np.digitize(days_since, edges, right=False).clip(0, n_buckets - 1)
    out = np.zeros((len(days_since), n_buckets), dtype=np.float32)
    out[np.arange(len(days_since)), idx] = 1.0
    return out


def build_user_features(
    train: pd.DataFrame,
    user_map: dict[str, int],
    anime_map: dict[int, int],
    genre_matrix: np.ndarray,  # [n_anime, n_genres]
) -> UserFeaturePack:
    n_users = len(user_map)
    n_genres = genre_matrix.shape[1]

    df = train.copy()
    df["user_idx"] = df["username"].map(user_map).astype("int64")
    df["anime_idx"] = df["anime_id"].astype(int).map(anime_map).astype("Int64")
    df = df.dropna(subset=["anime_idx"])
    df["anime_idx"] = df["anime_idx"].astype("int64")

    history: dict[int, np.ndarray] = {}
    history_scores: dict[int, np.ndarray] = {}
    centered = np.zeros(n_users, dtype=np.float32)
    affinity = np.zeros((n_users, n_genres), dtype=np.float32)

    df = df.sort_values(["user_idx", "my_last_updated"], kind="stable")

    for u_idx, group in df.groupby("user_idx", sort=False, observed=True):
        scores = group["my_score"].astype(np.float32).to_numpy()
        animes = group["anime_idx"].to_numpy()
        mu = float(scores.mean()) if len(scores) else 0.0
        centered[u_idx] = mu - 7.0  # MAL global mean ~7
        keep = scores >= mu
        if not keep.any():
            keep = np.ones_like(keep, dtype=bool)
        positives = animes[keep]
        positive_scores = scores[keep]
        history[int(u_idx)] = positives.astype(np.int64)
        history_scores[int(u_idx)] = positive_scores
        affinity[u_idx] = genre_matrix[positives].sum(axis=0)
        norm = affinity[u_idx].sum()
        if norm > 0:
            affinity[u_idx] /= norm

    last_ts = df.groupby("user_idx", observed=True)["my_last_updated"].max()
    if pd.api.types.is_datetime64_any_dtype(last_ts):
        ref = last_ts.max()
        days_since = (ref - last_ts).dt.days.astype(np.float32).to_numpy()
    else:
        days_since = np.zeros(len(last_ts), dtype=np.float32)
    recency_per_user = np.zeros((n_users, RECENCY_BUCKETS), dtype=np.float32)
    rec_buckets = _bucketize_recency(days_since)
    user_idxs = last_ts.index.to_numpy().astype(np.int64)
    recency_per_user[user_idxs] = rec_buckets

    return UserFeaturePack(
        history=history,
        history_scores=history_scores,
        genre_affinity=affinity,
        centered_avg_score=centered,
        recency=recency_per_user,
    )
