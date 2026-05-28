"""End-to-end preprocessing: filter ratings -> splits -> id maps -> anime features.

Run:
    python -m recsys.cli.preprocess --subset iter
"""
from __future__ import annotations

import argparse
import json
import pickle

import pandas as pd

from recsys.config import (
    ARTIFACTS_DIR,
    CACHE_DIR,
    CFG,
    SPLITS_DIR,
    SUBSET_PRESETS,
    set_random_seed,
    set_thread_env,
)
from recsys.data.features_anime import build_anime_features
from recsys.data.features_user import build_user_features
from recsys.data.load import load_anime
from recsys.data.split import build_id_maps, leave_one_out_split, save_splits
from recsys.data.subsample import build_subset, get_subset


def main() -> None:
    set_thread_env()
    p = argparse.ArgumentParser()
    p.add_argument("--subset", choices=list(SUBSET_PRESETS), default="iter")
    p.add_argument("--seed", type=int, default=CFG.seed, help="RNG for subset + LOO split")
    args = p.parse_args()
    set_random_seed(seed=args.seed)

    print("Loading anime catalog...")
    anime_df = load_anime()
    anime_df = anime_df.sort_values("anime_id", kind="stable").reset_index(drop=True)
    valid_ids = set(anime_df["anime_id"].astype(int).tolist())
    print(f"  {len(anime_df):,} anime")

    anime_episodes = {
        int(a): float(e) if pd.notna(e) and float(e) > 0 else float("nan")
        for a, e in zip(anime_df["anime_id"].astype(int), anime_df["episodes"])
    }

    print(f"Building subset='{args.subset}' from ratings (streaming)...")
    subset = get_subset(args.subset)
    ratings = build_subset(
        subset,
        valid_anime_ids=valid_ids,
        seed=args.seed,
        anime_episodes=anime_episodes,
    )
    print(f"  {len(ratings):,} interactions, {ratings['username'].nunique():,} users")
    if "completion_fraction" in ratings.columns:
        cf = ratings["completion_fraction"].astype(float)
        print(
            f"  status mix: "
            + ", ".join(
                f"s{int(k)}={int(v)}" for k, v in ratings["my_status"].value_counts().items()
            )
        )
        print(
            f"  completion_fraction: mean={cf.mean():.3f} "
            f"p50={cf.median():.3f} >=0.8 frac={(cf >= 0.8).mean():.3f}"
        )

    print("Computing leave-one-out split...")
    train, val, test = leave_one_out_split(ratings, seed=args.seed)
    print(f"  train={len(train):,} val={len(val):,} test={len(test):,}")
    save_splits(train, val, test)

    print("Building id maps...")
    user_map, anime_map = build_id_maps(train, anime_df)
    (ARTIFACTS_DIR / "user_map.json").write_text(json.dumps(user_map))
    (ARTIFACTS_DIR / "anime_map.json").write_text(
        json.dumps({str(k): v for k, v in anime_map.items()})
    )
    print(f"  {len(user_map):,} users, {len(anime_map):,} anime")

    print("Building anime features (this downloads the sentence-transformer on first run)...")
    feats = build_anime_features(anime_df)

    (ARTIFACTS_DIR / "genre_vocab.json").write_text(
        json.dumps({"genres": feats["genre_vocab"]})
    )

    print("Building user features...")
    user_features = build_user_features(train, user_map, anime_map, feats["genres"])
    with open(CACHE_DIR / "user_features.pkl", "wb") as f:
        pickle.dump(user_features, f)

    print("Done.")
    print(f"Artifacts under {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
