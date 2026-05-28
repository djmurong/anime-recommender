from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from recsys.config import (
    COMPLETED_STATUS,
    MIN_COMPLETION_FOR_POSITIVE,
    MIN_USER_RATINGS,
    SEED,
    SPLITS_DIR,
)


def _strong_positive_mask(df: pd.DataFrame) -> np.ndarray:
    """True where a row is a 'strong' positive eligible as the held-out target.

    A strong positive is a completed watch (status 2) with completion_fraction
    >= MIN_COMPLETION_FOR_POSITIVE. Falls back to status == 2 if the column is
    missing (older artifacts).
    """
    status = df["my_status"].astype("int8").to_numpy()
    if "completion_fraction" in df.columns:
        cf = df["completion_fraction"].astype("float32").to_numpy()
        return (status == COMPLETED_STATUS) & (cf >= MIN_COMPLETION_FOR_POSITIVE)
    return status == COMPLETED_STATUS


def leave_one_out_split(
    ratings: pd.DataFrame,
    seed: int = SEED,
    val_frac: float = 0.1,
    min_ratings: int = MIN_USER_RATINGS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hold out the most recent STRONG positive per user as the test target.

    A strong positive is a completed (status == 2) watch with high
    completion_fraction; anything weaker (dropped, plan-to-watch, partial watch)
    stays in train as multi-task signal for the MMoE ranker rather than being a
    target. A second-most-recent strong positive is held out for val_frac of
    users. Users with fewer than min_ratings strong positives are dropped.
    """
    df = ratings.copy()
    strong = _strong_positive_mask(df)

    # Eligibility is computed against strong positives only, so a user with 200
    # plan-to-watches but no completed shows is treated as cold-start.
    strong_counts = df[strong].groupby("username", observed=True).size()
    keep = set(strong_counts[strong_counts >= min_ratings].index)
    df = df[df["username"].isin(keep)].reset_index(drop=True)

    df = df.sort_values(["username", "my_last_updated"], kind="stable").reset_index(drop=True)

    # rank_strong_desc: 0 = most recent strong positive for that user, 1 = next, ...
    # rows that are not strong positives get NaN here.
    df["_is_strong"] = _strong_positive_mask(df)
    strong_idx = df.index[df["_is_strong"].to_numpy()]
    strong_df = df.loc[strong_idx, ["username"]].copy()
    strong_df["rank_strong_desc"] = (
        strong_df.groupby("username", observed=True).cumcount(ascending=False)
    )
    df["rank_strong_desc"] = pd.Series(
        strong_df["rank_strong_desc"].to_numpy(), index=strong_idx
    )

    is_test = df["rank_strong_desc"] == 0
    test = df[is_test].copy()

    rng = np.random.default_rng(seed)
    val_user_mask = rng.random(len(test)) < val_frac
    val_users = set(test.loc[val_user_mask, "username"].tolist())

    is_val = (df["rank_strong_desc"] == 1) & (df["username"].isin(val_users))
    val = df[is_val].copy()

    # Train = everything else, including weak interactions (status 1/3/4/6, partial
    # completion) so the MMoE ranker can learn drop/start/completion signals.
    train = df[~(is_test | is_val)].copy()

    for d in (train, val, test):
        d.drop(columns=["rank_strong_desc", "_is_strong"], inplace=True, errors="ignore")
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
