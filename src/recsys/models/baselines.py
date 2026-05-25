from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize


class Recommender(ABC):
    name: str = "base"

    @abstractmethod
    def recommend(
        self,
        user_idx: int,
        k: int,
        exclude: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (anime_idx, score) arrays of length up to k."""

    def batch_recommend(
        self,
        user_indices: np.ndarray,
        k: int,
        exclude_per_user: dict[int, np.ndarray] | None = None,
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for u in user_indices:
            ex = None if exclude_per_user is None else exclude_per_user.get(int(u))
            out[int(u)] = self.recommend(int(u), k, ex)
        return out


def _topk(scores: np.ndarray, k: int, exclude: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if exclude is not None and len(exclude):
        scores = scores.copy()
        scores[exclude] = -np.inf
    if k >= len(scores):
        order = np.argsort(-scores)
    else:
        idx = np.argpartition(-scores, k)[:k]
        order = idx[np.argsort(-scores[idx])]
    return order, scores[order]


@dataclass
class PopularityRec(Recommender):
    """Bayesian-shrunk popularity: (members * score) with smoothing toward global mean."""
    name: str = "Popularity"
    scores_: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    def fit(self, anime_df: pd.DataFrame, anime_map: dict[int, int]) -> "PopularityRec":
        n = len(anime_map)
        scores = np.zeros(n, dtype=np.float32)
        members = pd.to_numeric(anime_df["members"], errors="coerce").fillna(0.0).to_numpy()
        score_col = pd.to_numeric(anime_df["score"], errors="coerce").fillna(0.0).to_numpy()
        global_mean = float(np.nanmean(score_col[score_col > 0])) if (score_col > 0).any() else 6.5
        c = float(np.median(members[members > 0])) if (members > 0).any() else 1000.0
        bayes = (members * score_col + c * global_mean) / (members + c + 1e-9)
        bayes = bayes * np.log1p(members)
        anime_ids = anime_df["anime_id"].astype(int).to_numpy()
        for row, raw_id in enumerate(anime_ids):
            idx = anime_map.get(int(raw_id))
            if idx is not None:
                scores[idx] = float(bayes[row])
        self.scores_ = scores.astype(np.float32)
        return self

    def recommend(self, user_idx: int, k: int, exclude: np.ndarray | None = None):
        return _topk(self.scores_.copy(), k, exclude)


@dataclass
class ContentCosineRec(Recommender):
    """Cosine similarity over a content feature matrix; user vector = mean of train positives."""
    name: str = "ContentCosine"
    item_matrix_: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    user_vec_: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))

    def fit(
        self,
        feature_matrix: np.ndarray,
        train: pd.DataFrame,
        user_map: dict[str, int],
        anime_map: dict[int, int],
    ) -> "ContentCosineRec":
        item_matrix = normalize(feature_matrix.astype(np.float32))
        n_users = len(user_map)
        user_vec = np.zeros((n_users, item_matrix.shape[1]), dtype=np.float32)

        df = train.copy()
        df["user_idx"] = df["username"].map(user_map).astype("int64")
        df["anime_idx"] = df["anime_id"].astype(int).map(anime_map).astype("Int64")
        df = df.dropna(subset=["anime_idx"])
        df["anime_idx"] = df["anime_idx"].astype("int64")

        for u_idx, group in df.groupby("user_idx", sort=False, observed=True):
            mu = float(group["my_score"].mean())
            pos = group[group["my_score"] >= mu]
            if pos.empty:
                pos = group
            user_vec[u_idx] = item_matrix[pos["anime_idx"].to_numpy()].mean(axis=0)

        self.item_matrix_ = item_matrix
        self.user_vec_ = normalize(user_vec)
        return self

    def recommend(self, user_idx: int, k: int, exclude: np.ndarray | None = None):
        scores = self.item_matrix_ @ self.user_vec_[user_idx]
        return _topk(scores, k, exclude)


@dataclass
class ImplicitMFRec(Recommender):
    """Truncated-SVD matrix factorization on a (centered) interaction matrix.

    Equivalent in spirit to implicit-feedback MF: factorizes the user-item matrix into
    low-rank user/item factors, then scores items by their factor dot product. We use
    sklearn's TruncatedSVD (LAPACK-backed) so no native build tools are required.
    """
    name: str = "ImplicitMF"
    factors: int = 64
    iterations: int = 15
    user_factors_: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    item_factors_: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))

    def fit(
        self,
        train: pd.DataFrame,
        user_map: dict[str, int],
        anime_map: dict[int, int],
    ) -> "ImplicitMFRec":
        from sklearn.decomposition import TruncatedSVD

        df = train.copy()
        df["user_idx"] = df["username"].map(user_map).astype("int64")
        df["anime_idx"] = df["anime_id"].astype(int).map(anime_map).astype("Int64")
        df = df.dropna(subset=["anime_idx"])
        df["anime_idx"] = df["anime_idx"].astype("int64")

        rows = df["user_idx"].to_numpy()
        cols = df["anime_idx"].to_numpy()
        vals = df["my_score"].astype(np.float32).to_numpy()

        n_users, n_items = len(user_map), len(anime_map)
        mat = sp.coo_matrix((vals, (rows, cols)), shape=(n_users, n_items)).tocsr()

        k = min(self.factors, max(1, min(n_users, n_items) - 1))
        svd = TruncatedSVD(n_components=k, n_iter=self.iterations, random_state=0)
        user_factors = svd.fit_transform(mat).astype(np.float32)   # [n_users, k]
        item_factors = svd.components_.T.astype(np.float32)        # [n_items, k]

        self.user_factors_ = user_factors
        self.item_factors_ = item_factors
        return self

    def recommend(self, user_idx: int, k: int, exclude: np.ndarray | None = None):
        scores = self.item_factors_ @ self.user_factors_[user_idx]
        return _topk(scores.astype(np.float32), k, exclude)
