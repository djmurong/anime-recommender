"""Training dataset for the MMoE ranker.

The two-tower retriever is trained only on strong positives (completed, high
completion_fraction). The MMoE ranker needs the full multi-task signal, so
this dataset emits (user, candidate) pairs sampled from every status in the
expanded preprocess output:

  * positives are real interactions with their measured completion_fraction,
    drop label (status == 4), and rating_z.
  * negatives are random catalog draws with completion=0, drop=0, rating_z=0.

The pairs are kept distinct from `InteractionDataset` because the ranker
trains a *separate* model after the two-tower retriever has converged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from recsys.config import CFG, COMPLETED_STATUS, SEED
from recsys.data.features_user import UserFeaturePack


@dataclass
class _UserPositionInfo:
    """Per-user (anime_idx, position_in_user_history) for the position-bias surrogate."""

    anime_idx: np.ndarray
    position: np.ndarray
    slate_size: int


class RankerDataset(Dataset):
    """One example per labelled interaction (positive) plus k negatives per row.

    Position surrogate: position_in_list = rank of the anime in the user's
    *most-recent* slice (newest = 0). This is the position-bias-tower input
    until real impression logs are available.
    """

    def __init__(
        self,
        train: pd.DataFrame,
        user_map: dict[str, int],
        anime_map: dict[int, int],
        user_features: UserFeaturePack,
        n_anime: int,
        neg_per_pos: int = 4,
        max_position: int = 128,
        seed: int = SEED,
    ):
        df = train.copy()
        df["user_idx"] = df["username"].map(user_map).astype("int64")
        df["anime_idx"] = df["anime_id"].astype(int).map(anime_map).astype("Int64")
        df = df.dropna(subset=["anime_idx"])
        df["anime_idx"] = df["anime_idx"].astype("int64")
        df["ts_ns"] = pd.to_datetime(df["my_last_updated"], errors="coerce").astype("int64")
        if "completion_fraction" not in df.columns:
            df["completion_fraction"] = (
                df["my_status"].astype("int8") == COMPLETED_STATUS
            ).astype("float32")
        df = df.sort_values(["user_idx", "ts_ns"], kind="stable").reset_index(drop=True)

        self.user_idx = df["user_idx"].to_numpy(dtype=np.int64)
        self.anime_idx = df["anime_idx"].to_numpy(dtype=np.int64)
        self.status = df["my_status"].astype("int8").to_numpy()
        self.completion = df["completion_fraction"].astype(np.float32).to_numpy()
        self.score = df["my_score"].astype(np.float32).to_numpy()
        self.ts_ns = df["ts_ns"].to_numpy(dtype=np.int64)
        self.user_mean = user_features.centered_avg_score + 7.0
        self.user_features = user_features
        self.n_anime = n_anime
        self.neg_per_pos = neg_per_pos
        self.max_position = max_position
        self.seed = seed

        # Per-user position surrogate: index newest -> 0, oldest -> len-1.
        position_per_row = np.zeros(len(df), dtype=np.int64)
        slate_size_per_row = np.zeros(len(df), dtype=np.int64)
        for u_idx, group in df.groupby("user_idx", sort=False, observed=True):
            n = len(group)
            order_desc = group.sort_values("ts_ns", ascending=False).index.to_numpy()
            for new_rank, orig_idx in enumerate(order_desc):
                position_per_row[orig_idx] = min(new_rank, max_position)
                slate_size_per_row[orig_idx] = n
        self.position = position_per_row
        self.slate_size = slate_size_per_row

        # Positive set per user (any non-zero engagement) -> avoid sampling them
        # as random negatives.
        positives_lookup: dict[int, set[int]] = {}
        for u, a in zip(self.user_idx, self.anime_idx):
            positives_lookup.setdefault(int(u), set()).add(int(a))
        self._positives_lookup = positives_lookup

        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.user_idx)

    def _sample_negatives(self, u: int, pos_idx: int) -> np.ndarray:
        excluded = self._positives_lookup.get(u, set())
        out: list[int] = []
        tries = 0
        while len(out) < self.neg_per_pos and tries < self.neg_per_pos * 50:
            cand = int(self._rng.integers(0, self.n_anime))
            if cand == pos_idx or cand in excluded:
                tries += 1
                continue
            out.append(cand)
            tries += 1
        while len(out) < self.neg_per_pos:
            out.append(int(self._rng.integers(0, self.n_anime)))
        return np.array(out, dtype=np.int64)

    def __getitem__(self, i: int) -> dict:
        u = int(self.user_idx[i])
        a = int(self.anime_idx[i])
        score = float(self.score[i])
        cf = float(self.completion[i])
        status = int(self.status[i])
        rating_z = (score - float(self.user_mean[u])) / 3.0 if score > 0 else 0.0
        drop = float(status == 4)

        negs = self._sample_negatives(u, a)
        return {
            "user_idx": u,
            "pos_anime_idx": a,
            "neg_anime_idx": negs,
            "pos_completion": cf,
            "pos_rating_z": float(rating_z),
            "pos_drop": drop,
            "position": int(self.position[i]),
            "slate_size": int(self.slate_size[i]),
        }


def make_ranker_collate(user_features: UserFeaturePack):
    """Collate function for RankerDataset.

    Returns a flat tensor of (B*(1+neg_per_pos)) rows so the ranker forward
    pass scores positives and negatives in one shot. Labels are
    `1`/`completion_value` for positives and zeros for negatives.
    """
    affinity = torch.from_numpy(user_features.genre_affinity)
    centered = torch.from_numpy(user_features.centered_avg_score).unsqueeze(-1)
    recency = torch.from_numpy(user_features.recency)

    def collate(batch: list[dict]) -> dict:
        b = len(batch)
        neg_k = len(batch[0]["neg_anime_idx"])
        u_arr = np.empty(b, dtype=np.int64)
        pos_anime = np.empty(b, dtype=np.int64)
        neg_anime = np.empty((b, neg_k), dtype=np.int64)
        pos_cf = np.empty(b, dtype=np.float32)
        pos_rz = np.empty(b, dtype=np.float32)
        pos_drop = np.empty(b, dtype=np.float32)
        pos_position = np.empty(b, dtype=np.int64)
        pos_slate = np.empty(b, dtype=np.int64)
        for i, x in enumerate(batch):
            u_arr[i] = x["user_idx"]
            pos_anime[i] = x["pos_anime_idx"]
            neg_anime[i] = x["neg_anime_idx"]
            pos_cf[i] = x["pos_completion"]
            pos_rz[i] = x["pos_rating_z"]
            pos_drop[i] = x["pos_drop"]
            pos_position[i] = x["position"]
            pos_slate[i] = x["slate_size"]
        u_t = torch.from_numpy(u_arr)
        pos_t = torch.from_numpy(pos_anime)
        neg_t = torch.from_numpy(neg_anime)
        return {
            "user_idx": u_t,
            "pos_anime_idx": pos_t,
            "neg_anime_idx": neg_t,
            "pos_completion": torch.from_numpy(pos_cf),
            "pos_rating_z": torch.from_numpy(pos_rz),
            "pos_drop": torch.from_numpy(pos_drop),
            "pos_position": torch.from_numpy(pos_position),
            "pos_slate_size": torch.from_numpy(pos_slate),
            "genre_affinity": affinity[u_t],
            "centered_score": centered[u_t],
            "recency": recency[u_t],
        }

    return collate
