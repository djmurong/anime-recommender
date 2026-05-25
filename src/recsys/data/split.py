from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from recsys.config import MIN_USER_RATINGS, SPLITS_DIR


def leave_one_out_split(
    ratings: pd.DataFrame,
    seed: int = 42,
    val_frac: float = 0.1,
    min_ratings: int = MIN_USER_RATINGS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hold out the most recent positive per user as the test target.

    A second-most-recent positive is held out for `val_frac` of users as a validation set.
    Users with fewer than `min_ratings` are dropped entirely (cold-start path at serve time).

    Returns (train, val, test) frames.
    """
    counts = ratings.groupby("username", observed=True).size()
    keep = set(counts[counts >= min_ratings].index)
    df = ratings[ratings["username"].isin(keep)].copy()

    df = df.sort_values(["username", "my_last_updated"], kind="stable")
    df["rank_desc"] = df.groupby("username", observed=True).cumcount(ascending=False)

    test = df[df["rank_desc"] == 0].copy()

    rng = np.random.default_rng(seed)
    val_user_mask = rng.random(len(test)) < val_frac
    val_users = set(test.loc[val_user_mask, "username"].tolist())

    val = df[(df["rank_desc"] == 1) & (df["username"].isin(val_users))].copy()
    train = df[
        (df["rank_desc"] >= 1)
        & ~((df["rank_desc"] == 1) & (df["username"].isin(val_users)))
    ].copy()

    for d in (train, val, test):
        d.drop(columns=["rank_desc"], inplace=True)
    return train, val, test


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, out_dir: Path = SPLITS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)


def load_splits(in_dir: Path = SPLITS_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(in_dir / "train.parquet")
    val = pd.read_parquet(in_dir / "val.parquet")
    test = pd.read_parquet(in_dir / "test.parquet")
    return train, val, test


def build_id_maps(
    train: pd.DataFrame, anime_df: pd.DataFrame
) -> tuple[dict[str, int], dict[int, int]]:
    """Stable string-username -> dense int, MAL anime_id -> dense int."""
    users_sorted = sorted(train["username"].unique().tolist())
    user_map = {u: i for i, u in enumerate(users_sorted)}
    anime_sorted = sorted(anime_df["anime_id"].astype(int).unique().tolist())
    anime_map = {int(a): i for i, a in enumerate(anime_sorted)}
    return user_map, anime_map
