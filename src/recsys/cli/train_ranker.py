"""Train the MMoE ranker on top of a trained two-tower retriever.

Run (after preprocess + train_two_tower):
    python -m recsys.cli.train_ranker --epochs 5
    python -m recsys.cli.train_ranker --device cuda --epochs 5 --batch-size 1024
    python -m recsys.cli.train_ranker --no-position-bias  # ablation
"""
from __future__ import annotations

import argparse
import json
import pickle

from recsys.config import (
    ARTIFACTS_DIR,
    CACHE_DIR,
    CFG,
    MODELS_DIR,
    TrainConfig,
    resolve_device,
    set_random_seed,
    set_thread_env,
)
from recsys.data.features_anime import build_anime_features
from recsys.data.features_user import RECENCY_BUCKETS
from recsys.data.load import load_anime
from recsys.data.split import load_splits
from recsys.models.popularity_bias import build_popularity_bias_vector
from recsys.models.two_tower import load_two_tower_from_checkpoint
from recsys.training.ranker_trainer import train_ranker


def _load_id_maps() -> tuple[dict[str, int], dict[int, int]]:
    user_map = json.loads((ARTIFACTS_DIR / "user_map.json").read_text())
    anime_map_raw = json.loads((ARTIFACTS_DIR / "anime_map.json").read_text())
    return user_map, {int(k): v for k, v in anime_map_raw.items()}


def main() -> None:
    set_thread_env()
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--neg-per-pos", type=int, default=4)
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="cpu",
        help="Compute device (default: cpu).",
    )
    p.add_argument("--cpu", action="store_true", help="Shorthand for --device cpu.")
    p.add_argument(
        "--no-position-bias",
        action="store_true",
        help="Disable the position-bias side tower (ablation).",
    )
    args = p.parse_args()

    device = resolve_device("cpu" if args.cpu else args.device)
    set_random_seed(device=device)

    train_df, val_df, _ = load_splits()
    user_map, anime_map = _load_id_maps()
    anime_df = load_anime().sort_values("anime_id", kind="stable").reset_index(drop=True)
    feats = build_anime_features(anime_df)
    with open(CACHE_DIR / "user_features.pkl", "rb") as f:
        user_features = pickle.load(f)

    train_cfg = TrainConfig()
    pop_bias = build_popularity_bias_vector(anime_df, anime_map)

    import torch

    ckpt_path = MODELS_DIR / "two_tower.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    retriever = load_two_tower_from_checkpoint(
        ckpt,
        recency_dim=RECENCY_BUCKETS,
        n_anime=feats["numerical"].shape[0],
        popularity_bias=pop_bias,
    )

    print(f"Training MMoE ranker on device={device}, epochs={args.epochs}, "
          f"position_bias={'off' if args.no_position_bias else 'on'}")
    artifacts = train_ranker(
        retriever=retriever,
        train_df=train_df,
        val_df=val_df,
        feats=feats,
        user_features=user_features,
        user_map=user_map,
        anime_map=anime_map,
        train_cfg=train_cfg,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        neg_per_pos=args.neg_per_pos,
        use_position_bias=not args.no_position_bias,
        ckpt_path=MODELS_DIR / "ranker.pt",
    )
    print(
        f"Best train loss = {artifacts.best_val:.4f} at epoch {artifacts.best_epoch}; "
        f"saved -> {MODELS_DIR / 'ranker.pt'}"
    )


if __name__ == "__main__":
    main()
