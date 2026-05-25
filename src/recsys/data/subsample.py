from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.config import COMPLETED_STATUS, SUBSET_PRESETS, SubsetConfig
from recsys.data.load import iter_ratings_chunks


def filter_completed(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows representing a completed watch with a positive score."""
    return df[(df["my_status"] == COMPLETED_STATUS) & (df["my_score"] > 0)].copy()


def build_subset(
    subset: SubsetConfig,
    valid_anime_ids: set[int],
    seed: int = 42,
    chunksize: int = 2_000_000,
) -> pd.DataFrame:
    """Stream the full ratings CSV, filter, then sample users by interaction count.

    Returns a DataFrame with columns:
      username, anime_id, my_score, my_status, my_last_updated
    """
    counter: dict[str, int] = {}
    kept_chunks: list[pd.DataFrame] = []

    for chunk in iter_ratings_chunks(chunksize=chunksize):
        chunk = filter_completed(chunk)
        chunk = chunk[chunk["anime_id"].isin(valid_anime_ids)]
        if chunk.empty:
            continue
        for u, n in chunk["username"].value_counts().items():
            counter[u] = counter.get(u, 0) + int(n)
        kept_chunks.append(chunk)

    if not kept_chunks:
        raise RuntimeError("No completed ratings found after filtering.")

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
    return out


def get_subset(name: str) -> SubsetConfig:
    if name not in SUBSET_PRESETS:
        raise ValueError(f"Unknown subset '{name}'. Choose from {list(SUBSET_PRESETS)}")
    return SUBSET_PRESETS[name]
