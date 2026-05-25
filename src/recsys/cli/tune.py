"""Run Optuna hyperparameter search.

Run:
    python -m recsys.cli.tune --trials 30
"""
from __future__ import annotations

import argparse
import json
import pickle

from recsys.config import ARTIFACTS_DIR, CACHE_DIR, CFG, set_thread_env
from recsys.data.features_anime import build_anime_features
from recsys.data.load import load_anime
from recsys.data.split import load_splits
from recsys.models.popularity_bias import build_popularity_bias_vector
from recsys.training.tune import run_study


def main() -> None:
    set_thread_env()
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=CFG.tune.n_trials)
    p.add_argument(
        "--max-train-rows",
        type=int,
        default=2_000_000,
        help="Subsample train rows during search (full train uses all rows)",
    )
    args = p.parse_args()

    train_df, val_df, _ = load_splits()
    user_map = json.loads((ARTIFACTS_DIR / "user_map.json").read_text())
    anime_map = {int(k): v for k, v in json.loads((ARTIFACTS_DIR / "anime_map.json").read_text()).items()}
    anime_df = load_anime().sort_values("anime_id", kind="stable").reset_index(drop=True)
    feats = build_anime_features(anime_df)
    with open(CACHE_DIR / "user_features.pkl", "rb") as f:
        user_features = pickle.load(f)

    pop_bias = build_popularity_bias_vector(anime_df, anime_map)
    best = run_study(
        train_df=train_df,
        val_df=val_df,
        feats=feats,
        user_features=user_features,
        user_map=user_map,
        anime_map=anime_map,
        popularity_bias=pop_bias,
        n_trials=args.trials,
        max_train_rows=args.max_train_rows,
    )
    print(f"Best value: {best['value']:.4f}")
    print(f"Best params: {best['params']}")


if __name__ == "__main__":
    main()
