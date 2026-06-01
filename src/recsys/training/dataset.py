from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler

from recsys.config import CFG, COMPLETED_STATUS, MIN_COMPLETION_FOR_POSITIVE, SEED

# torch.multinomial (used by WeightedRandomSampler) supports at most 2**24 categories
_MAX_TORCH_MULTINOMIAL = 2**24
from recsys.data.features_user import RECENCY_BUCKETS, UserFeaturePack


@dataclass
class _UserTimeline:
    ts_ns: np.ndarray
    anime_idx: np.ndarray
    scores: np.ndarray
    completion: np.ndarray


def _to_ts_ns(series: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(series, errors="coerce")
    return ts.astype("int64").to_numpy(dtype=np.int64)


_NS_PER_DAY = 86400.0 * 1e9


class InteractionDataset(Dataset):
    """One example per strong-positive (user, anime) pair from the train split.

    A "strong positive" is a completed watch with completion_fraction >=
    MIN_COMPLETION_FOR_POSITIVE -- this matches the test/val target definition
    in `data.split.leave_one_out_split`. Causal history is every interaction
    (including weak ones: drops, on-hold, plan-to-watch, partial watches)
    strictly before the target timestamp, so the Transformer encoder gets the
    full signal even though only strong rows produce training positives.
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

        if "completion_fraction" in df.columns:
            df["completion_fraction"] = df["completion_fraction"].astype("float32")
        else:
            status = df["my_status"].astype("int8").to_numpy()
            df["completion_fraction"] = (status == COMPLETED_STATUS).astype("float32")

        # Per-user timelines: every interaction, used to build causal history.
        timelines: dict[int, _UserTimeline] = {}
        for u_idx, group in df.groupby("user_idx", sort=False, observed=True):
            timelines[int(u_idx)] = _UserTimeline(
                ts_ns=group["ts_ns"].to_numpy(dtype=np.int64),
                anime_idx=group["anime_idx"].to_numpy(dtype=np.int64),
                scores=group["my_score"].astype(np.float32).to_numpy(),
                completion=group["completion_fraction"].to_numpy(dtype=np.float32),
            )
        self._timelines = timelines

        # Positives feed only strong rows into the in-batch softmax / catalog
        # negatives. Weak rows still live in `_timelines` for history.
        status_arr = df["my_status"].astype("int8").to_numpy()
        cf_arr = df["completion_fraction"].astype(np.float32).to_numpy()
        strong = (status_arr == COMPLETED_STATUS) & (cf_arr >= MIN_COMPLETION_FOR_POSITIVE)
        if not strong.any():
            strong = status_arr == COMPLETED_STATUS  # fallback for legacy preprocess

        strong_df = df[strong].reset_index(drop=True)
        self.user_idx = strong_df["user_idx"].to_numpy(dtype=np.int64)
        self.pos_idx = strong_df["anime_idx"].to_numpy(dtype=np.int64)
        self.target_ts_ns = strong_df["ts_ns"].to_numpy(dtype=np.int64)
        self.pos_score = strong_df["my_score"].astype(np.float32).to_numpy()
        self.pos_completion = strong_df["completion_fraction"].astype(np.float32).to_numpy()

        self.user_features = user_features
        self.max_history = max_history
        self._user_mean_score = user_features.centered_avg_score + 7.0

        # log-Q correction terms: probability of seeing each anime among strong
        # positives. Subtracted from sampled-softmax logits to debias popular
        # items inside the in-batch negatives.
        n_items = max(int(self.pos_idx.max()) + 1, len(anime_map))
        counts = np.bincount(self.pos_idx, minlength=n_items)
        total = counts.sum()
        prob = counts / max(total, 1)
        self.log_q = np.log(prob.clip(min=1.0 / max(total, 1)))

        ref_ts = int(self.target_ts_ns.max()) if len(self.target_ts_ns) else 0
        tau_ns = float(recency_tau_days) * _NS_PER_DAY
        days_ago = (ref_ts - self.target_ts_ns) / max(tau_ns, 1.0)
        self.sample_weights = np.exp(-days_ago).astype(np.float64)

    def __len__(self) -> int:
        return len(self.user_idx)

    def _causal_prefix(
        self, u: int, target_ts: int, exclude_anime: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (anime_idx, weights, scores, completion, days_ago) for causal history."""
        tl = self._timelines.get(u)
        if tl is None or len(tl.ts_ns) == 0:
            return (
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )

        end = int(np.searchsorted(tl.ts_ns, target_ts, side="left"))
        if end <= 0:
            return (
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )

        animes = tl.anime_idx[:end]
        scores = tl.scores[:end]
        completion = tl.completion[:end]
        ts = tl.ts_ns[:end]
        mask = animes != exclude_anime
        animes = animes[mask]
        scores = scores[mask]
        completion = completion[mask]
        ts = ts[mask]
        if len(animes) == 0:
            return (
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )

        if len(animes) > self.max_history:
            animes = animes[-self.max_history :]
            scores = scores[-self.max_history :]
            completion = completion[-self.max_history :]
            ts = ts[-self.max_history :]

        mu = float(self._user_mean_score[u])
        weights = np.maximum(scores - mu, 0.1).astype(np.float32)
        days_ago = ((target_ts - ts) / _NS_PER_DAY).astype(np.float32)
        return (
            animes.astype(np.int64),
            weights,
            scores.astype(np.float32),
            completion.astype(np.float32),
            days_ago,
        )

    def __getitem__(self, i: int) -> dict:
        u = int(self.user_idx[i])
        a = int(self.pos_idx[i])
        ts = int(self.target_ts_ns[i])
        history, hist_weights, hist_scores, hist_completion, days_ago = self._causal_prefix(
            u, ts, a
        )
        return {
            "user_idx": u,
            "pos_anime_idx": a,
            "pos_score": float(self.pos_score[i]),
            "pos_completion": float(self.pos_completion[i]),
            "history": history,
            "history_weights": hist_weights,
            "history_scores": hist_scores,
            "history_completion": hist_completion,
            "history_days_ago": days_ago,
        }


def make_collate(
    user_features: UserFeaturePack,
    n_anime: int,
    log_q: np.ndarray,
    completion_floor: float = 0.1,
):
    """Returns a closure that batches examples into padded tensors.

    `pos_weight` for completion-weighted loss is computed here so the trainer
    stays light. weight = max(pos_completion, floor) * (1 + max(score_z, 0)).
    """

    affinity = torch.from_numpy(user_features.genre_affinity)
    centered = torch.from_numpy(user_features.centered_avg_score).unsqueeze(-1)
    recency = torch.from_numpy(user_features.recency)
    log_q_t = torch.from_numpy(log_q.astype(np.float32))
    user_mean = torch.from_numpy(user_features.centered_avg_score + 7.0)

    def collate(batch: list[dict]) -> dict:
        b = len(batch)
        max_h = max((len(x["history"]) for x in batch), default=1)
        max_h = max(max_h, 1)
        hist = np.zeros((b, max_h), dtype=np.int64)
        mask = np.zeros((b, max_h), dtype=np.float32)
        weights = np.zeros((b, max_h), dtype=np.float32)
        scores_h = np.zeros((b, max_h), dtype=np.float32)
        comp_h = np.zeros((b, max_h), dtype=np.float32)
        days_h = np.zeros((b, max_h), dtype=np.float32)
        u_arr = np.empty(b, dtype=np.int64)
        a_arr = np.empty(b, dtype=np.int64)
        pos_score = np.empty(b, dtype=np.float32)
        pos_completion = np.empty(b, dtype=np.float32)
        for i, x in enumerate(batch):
            u_arr[i] = x["user_idx"]
            a_arr[i] = x["pos_anime_idx"]
            pos_score[i] = x["pos_score"]
            pos_completion[i] = x["pos_completion"]
            h = x["history"]
            if len(h):
                hist[i, : len(h)] = h
                mask[i, : len(h)] = 1.0
                weights[i, : len(h)] = x["history_weights"]
                scores_h[i, : len(h)] = x["history_scores"]
                comp_h[i, : len(h)] = x["history_completion"]
                days_h[i, : len(h)] = x["history_days_ago"]
        u_t = torch.from_numpy(u_arr)
        a_t = torch.from_numpy(a_arr)
        pos_score_t = torch.from_numpy(pos_score)
        pos_comp_t = torch.from_numpy(pos_completion)
        # Centered score relative to the user's mean -> rating_z in [-1, +1] ish.
        score_z = (pos_score_t - user_mean[u_t]) / 3.0
        pos_weight = pos_comp_t.clamp(min=completion_floor) * (1.0 + score_z.clamp(min=0.0))

        return {
            "user_idx": u_t,
            "pos_anime_idx": a_t,
            "pos_score": pos_score_t,
            "pos_completion": pos_comp_t,
            "pos_weight": pos_weight,
            "history_idx": torch.from_numpy(hist),
            "history_mask": torch.from_numpy(mask),
            "history_weights": torch.from_numpy(weights),
            "history_scores": torch.from_numpy(scores_h),
            "history_completion": torch.from_numpy(comp_h),
            "history_days_ago": torch.from_numpy(days_h),
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
        # Avoid .tolist() on multi-million-row sets (huge RAM + slow epoch start).
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples


def make_train_sampler(dataset: InteractionDataset, seed: int = SEED) -> Sampler[int]:
    """Recency-weighted sampler; falls back to numpy for very large train sets."""
    n = len(dataset)
    weights = dataset.sample_weights
    if n <= _MAX_TORCH_MULTINOMIAL:
        gen = torch.Generator()
        gen.manual_seed(seed)
        return WeightedRandomSampler(
            weights=torch.from_numpy(weights.astype(np.float64)),
            num_samples=n,
            replacement=True,
            generator=gen,
        )
    return RecencyWeightedSampler(weights=weights, num_samples=n, seed=seed)
