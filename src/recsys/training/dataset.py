from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler

from recsys.config import CFG, SEED

# torch.multinomial (used by WeightedRandomSampler) supports at most 2**24 categories
_MAX_TORCH_MULTINOMIAL = 2**24
from recsys.data.features_user import RECENCY_BUCKETS, UserFeaturePack


@dataclass
class _UserTimeline:
    ts_ns: np.ndarray
    anime_idx: np.ndarray
    scores: np.ndarray


def _to_ts_ns(series: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(series, errors="coerce")
    return ts.astype("int64").to_numpy(dtype=np.int64)


class InteractionDataset(Dataset):
    """One example per (user, positive anime) pair from the train split.

    History is causal: only events strictly before the target timestamp.
    """

    def __init__(
        self,
        train: pd.DataFrame,
        user_map: dict[str, int],
        anime_map: dict[int, int],
        user_features: UserFeaturePack,
        max_history: int | None = None,
        recency_tau_days: float | None = None,
    ):
        max_history = max_history if max_history is not None else CFG.train.max_history_len
        recency_tau_days = (
            recency_tau_days if recency_tau_days is not None else CFG.train.recency_sample_tau_days
        )

        df = train.copy()
        df["user_idx"] = df["username"].map(user_map).astype("int64")
        df["anime_idx"] = df["anime_id"].astype(int).map(anime_map).astype("Int64")
        df = df.dropna(subset=["anime_idx"])
        df["anime_idx"] = df["anime_idx"].astype("int64")
        df["ts_ns"] = _to_ts_ns(df["my_last_updated"])
        df = df.sort_values(["user_idx", "ts_ns"], kind="stable").reset_index(drop=True)

        self.user_idx = df["user_idx"].to_numpy(dtype=np.int64)
        self.pos_idx = df["anime_idx"].to_numpy(dtype=np.int64)
        self.target_ts_ns = df["ts_ns"].to_numpy(dtype=np.int64)
        self.user_features = user_features
        self.max_history = max_history
        self._user_mean_score = user_features.centered_avg_score + 7.0

        timelines: dict[int, _UserTimeline] = {}
        for u_idx, group in df.groupby("user_idx", sort=False, observed=True):
            timelines[int(u_idx)] = _UserTimeline(
                ts_ns=group["ts_ns"].to_numpy(dtype=np.int64),
                anime_idx=group["anime_idx"].to_numpy(dtype=np.int64),
                scores=group["my_score"].astype(np.float32).to_numpy(),
            )
        self._timelines = timelines

        counts = np.bincount(self.pos_idx, minlength=int(self.pos_idx.max()) + 1)
        total = counts.sum()
        prob = counts / max(total, 1)
        self.log_q = np.log(prob.clip(min=1.0 / max(total, 1)))

        ref_ts = int(self.target_ts_ns.max())
        tau_ns = float(recency_tau_days) * 86400.0 * 1e9
        days_ago = (ref_ts - self.target_ts_ns) / max(tau_ns, 1.0)
        self.sample_weights = np.exp(-days_ago).astype(np.float64)

    def __len__(self) -> int:
        return len(self.user_idx)

    def _causal_prefix(self, u: int, target_ts: int, exclude_anime: int) -> tuple[np.ndarray, np.ndarray]:
        tl = self._timelines.get(u)
        if tl is None or len(tl.ts_ns) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

        end = int(np.searchsorted(tl.ts_ns, target_ts, side="left"))
        if end <= 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

        animes = tl.anime_idx[:end]
        scores = tl.scores[:end]
        mask = animes != exclude_anime
        animes = animes[mask]
        scores = scores[mask]
        if len(animes) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

        if len(animes) > self.max_history:
            animes = animes[-self.max_history :]
            scores = scores[-self.max_history :]

        mu = float(self._user_mean_score[u])
        weights = np.maximum(scores - mu, 0.1).astype(np.float32)
        return animes.astype(np.int64), weights

    def __getitem__(self, i: int) -> dict:
        u = int(self.user_idx[i])
        a = int(self.pos_idx[i])
        ts = int(self.target_ts_ns[i])
        history, hist_weights = self._causal_prefix(u, ts, a)
        return {
            "user_idx": u,
            "pos_anime_idx": a,
            "history": history,
            "history_weights": hist_weights,
        }


def make_collate(user_features: UserFeaturePack, n_anime: int, log_q: np.ndarray):
    """Returns a closure that batches examples into padded tensors."""

    affinity = torch.from_numpy(user_features.genre_affinity)
    centered = torch.from_numpy(user_features.centered_avg_score).unsqueeze(-1)
    recency = torch.from_numpy(user_features.recency)
    log_q_t = torch.from_numpy(log_q.astype(np.float32))

    def collate(batch: list[dict]) -> dict:
        b = len(batch)
        max_h = max((len(x["history"]) for x in batch), default=1)
        max_h = max(max_h, 1)
        hist = np.zeros((b, max_h), dtype=np.int64)
        mask = np.zeros((b, max_h), dtype=np.float32)
        weights = np.zeros((b, max_h), dtype=np.float32)
        u_arr = np.empty(b, dtype=np.int64)
        a_arr = np.empty(b, dtype=np.int64)
        for i, x in enumerate(batch):
            u_arr[i] = x["user_idx"]
            a_arr[i] = x["pos_anime_idx"]
            h = x["history"]
            w = x["history_weights"]
            if len(h):
                hist[i, : len(h)] = h
                mask[i, : len(h)] = 1.0
                weights[i, : len(h)] = w
        u_t = torch.from_numpy(u_arr)
        a_t = torch.from_numpy(a_arr)
        return {
            "user_idx": u_t,
            "pos_anime_idx": a_t,
            "history_idx": torch.from_numpy(hist),
            "history_mask": torch.from_numpy(mask),
            "history_weights": torch.from_numpy(weights),
            "genre_affinity": affinity[u_t],
            "centered_score": centered[u_t],
            "recency": recency[u_t],
            "pos_log_q": log_q_t[a_t],
        }

    return collate


class RecencyWeightedSampler(Sampler[int]):
    """Weighted sampling via numpy — works when len(dataset) > 2**24."""

    def __init__(
        self,
        weights: np.ndarray,
        num_samples: int | None = None,
        seed: int = SEED,
    ):
        w = np.asarray(weights, dtype=np.float64)
        total = w.sum()
        self.probs = (w / total) if total > 0 else np.full(len(w), 1.0 / max(len(w), 1))
        self.num_samples = num_samples if num_samples is not None else len(w)
        self.n = len(w)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = rng.choice(self.n, size=self.num_samples, replace=True, p=self.probs)
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


def make_train_sampler(dataset: InteractionDataset, seed: int = SEED) -> Sampler[int]:
    """Recency-weighted sampler; falls back to numpy for very large train sets."""
    n = len(dataset)
    weights = dataset.sample_weights
    if n <= _MAX_TORCH_MULTINOMIAL:
        return WeightedRandomSampler(
            weights=torch.from_numpy(weights.astype(np.float64)),
            num_samples=n,
            replacement=True,
        )
    return RecencyWeightedSampler(weights=weights, num_samples=n, seed=seed)
