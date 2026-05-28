from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.config import (
    COMPLETED_STATUS,
    KEEP_STATUSES,
    SEED,
    SUBSET_PRESETS,
    SubsetConfig,
)
from recsys.data.load import iter_ratings_chunks


def filter_completed(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy filter: keep only fully completed rows with a positive score.

    Retained because the baselines (Popularity, ContentCosine, ImplicitMF) are still
    trained on the completed-only slice so their numbers stay comparable across the
    cascade refactor.
    """
    return df[(df["my_status"] == COMPLETED_STATUS) & (df["my_score"] > 0)].copy()


def filter_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Keep watching / completed / on-hold / dropped / plan-to-watch rows.

    We deliberately do NOT require my_score > 0 anymore -- watch-time and drop
    signal carry information for the MMoE heads even when the user never rated.
    Only rows that are KEEP_STATUSES with either a score or an episode count are
    retained so unmarked entries don't pollute training.
    """
    keep_status = df["my_status"].isin(KEEP_STATUSES)
    has_signal = (df["my_score"] > 0) | (df["my_watched_episodes"].fillna(0) > 0) | (df["my_status"] == 6)
    return df[keep_status & has_signal].copy()


def attach_completion(df: pd.DataFrame, anime_episodes: dict[int, float]) -> pd.DataFrame:
    """Compute completion_fraction in [0, 1] per interaction.

    status 2 (completed) -> 1.0 regardless of recorded episodes (MAL truncates).
    status 6 (plan-to-watch) -> 0.0.
    otherwise -> my_watched_episodes / episodes, clipped to [0, 1]. If the anime
    has no episode count (movies, ongoing) we fall back to a heuristic: 1.0 if
    watched_episodes > 0 else 0.0.
    """
    df = df.copy()
    eps = df["anime_id"].astype(int).map(anime_episodes).astype("float32")
    watched = df["my_watched_episodes"].fillna(0).astype("float32")
    raw = np.where(
        eps.notna().to_numpy() & (eps.fillna(0).to_numpy() > 0),
        watched / eps.fillna(1).to_numpy(),
        np.where(watched > 0, 1.0, 0.0),
    ).clip(0.0, 1.0)

    status = df["my_status"].astype("int8").to_numpy()
    raw = np.where(status == 2, 1.0, raw)
    raw = np.where(status == 6, 0.0, raw)
    df["completion_fraction"] = raw.astype("float32")
    return df


def build_subset(
    subset: SubsetConfig,
    valid_anime_ids: set[int],
    seed: int = SEED,
    chunksize: int = 2_000_000,
    anime_episodes: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Stream the full ratings CSV, filter to KEEP_STATUSES, then sample users.

    Returns a DataFrame with columns:
      username, anime_id, my_score, my_status, my_watched_episodes,
      my_last_updated, completion_fraction
    """
    counter: dict[str, int] = {}
    kept_chunks: list[pd.DataFrame] = []

    for chunk in iter_ratings_chunks(chunksize=chunksize):
        chunk = filter_interactions(chunk)
        chunk = chunk[chunk["anime_id"].isin(valid_anime_ids)]
        if chunk.empty:
            continue
        # Sampling-quota counter uses "strong" interactions (completed or rated),
        # so users with hundreds of plan-to-watch entries but no real engagement
        # don't dominate the subset.
        strong = chunk[(chunk["my_status"] == COMPLETED_STATUS) | (chunk["my_score"] > 0)]
        for u, n in strong["username"].value_counts().items():
            counter[u] = counter.get(u, 0) + int(n)
        kept_chunks.append(chunk)

    if not kept_chunks:
        raise RuntimeError("No interactions found after filtering.")

    counts = pd.Series(counter, name="n").astype("int32")
    eligible = counts[
        (counts >= subset.min_user_ratings) & (counts <= subset.max_user_ratings)
    ]

    if len(eligible) > subset.max_users:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(eligible.index.to_numpy(), size=subset.max_users, replace=False)
        keep_users = set(chosen.tolist())
    else:
        keep_users = set(eligible.index.tolist())

    out = pd.concat(kept_chunks, ignore_index=True)
    out = out[out["username"].isin(keep_users)].reset_index(drop=True)
    if anime_episodes is not None:
        out = attach_completion(out, anime_episodes)
    else:
        out["completion_fraction"] = np.where(
            out["my_status"].astype("int8").to_numpy() == COMPLETED_STATUS, 1.0, 0.0
        ).astype("float32")
    return out


def get_subset(name: str) -> SubsetConfig:
    if name not in SUBSET_PRESETS:
        raise ValueError(f"Unknown subset '{name}'. Choose from {list(SUBSET_PRESETS)}")
    return SUBSET_PRESETS[name]
