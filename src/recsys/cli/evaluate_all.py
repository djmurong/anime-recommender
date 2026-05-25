"""Run all baselines + the two-tower model on the same leave-one-out test split.

Run:
    python -m recsys.cli.evaluate_all
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import torch

from recsys.config import ARTIFACTS_DIR, CACHE_DIR, CFG, MODELS_DIR, set_thread_env
from recsys.data.features_anime import build_anime_features
from recsys.data.features_user import RECENCY_BUCKETS
from recsys.data.load import load_anime
from recsys.data.split import load_splits
from recsys.eval.harness import (
    evaluate_baseline,
    evaluate_two_tower,
    format_table,
    write_report,
)
from recsys.models.popularity_bias import build_popularity_bias_vector
from recsys.models.two_tower import load_two_tower_from_checkpoint


def _popularity_rank(feats: dict) -> np.ndarray:
    """Rank index 0 = most popular. Falls back to anime row order if numerical missing."""
    n = feats["numerical"].shape[0]
    pop_col = None
    cols = list(getattr(CFG.features, "numerical_cols", []))
    if "popularity" in cols:
        pop_col = feats["numerical"][:, cols.index("popularity")]
    if pop_col is None:
        pop_col = -np.arange(n, dtype=np.float32)  # arbitrary stable order
    order = np.argsort(pop_col)  # smaller "popularity" number = more popular on MAL
    rank = np.empty_like(order)
    rank[order] = np.arange(n)
    return rank


def main() -> None:
    set_thread_env()
    train, val, test = load_splits()
    user_map = json.loads((ARTIFACTS_DIR / "user_map.json").read_text())
    anime_map = {int(k): v for k, v in json.loads((ARTIFACTS_DIR / "anime_map.json").read_text()).items()}
    anime_df = load_anime().sort_values("anime_id", kind="stable").reset_index(drop=True)
    feats = build_anime_features(anime_df)
    with open(CACHE_DIR / "user_features.pkl", "rb") as f:
        user_features = pickle.load(f)

    item_features_for_ild = np.concatenate([feats["genres"], feats["numerical"]], axis=1).astype(np.float32)
    pop_rank = _popularity_rank(feats)

    with open(MODELS_DIR / "baselines.pkl", "rb") as f:
        bls = pickle.load(f)

    results = []
    for key in ("popularity", "content", "mf"):
        rec = bls[key]
        print(f"Evaluating {rec.name}...")
        results.append(
            evaluate_baseline(
                rec=rec,
                test_df=test,
                user_features=user_features,
                user_map=user_map,
                anime_map=anime_map,
                item_features_for_ild=item_features_for_ild,
                popularity_rank=pop_rank,
            )
        )

    ckpt_path = MODELS_DIR / "two_tower.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        pop_bias = build_popularity_bias_vector(anime_df, anime_map)
        model = load_two_tower_from_checkpoint(
            ckpt,
            recency_dim=RECENCY_BUCKETS,
            n_anime=feats["numerical"].shape[0],
            popularity_bias=pop_bias,
        )
        print("Evaluating TwoTower...")
        results.append(
            evaluate_two_tower(
                model=model,
                feats=feats,
                test_df=test,
                user_features=user_features,
                user_map=user_map,
                anime_map=anime_map,
                item_features_for_ild=item_features_for_ild,
                popularity_rank=pop_rank,
            )
        )
    else:
        print(f"No two-tower checkpoint at {ckpt_path}; skipping. Run cli.train_two_tower first.")

    table = format_table(results)
    print()
    print(table)
    out = write_report(results)
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
