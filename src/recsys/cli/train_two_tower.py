"""Train the two-tower model.

Run:
    python -m recsys.cli.train_two_tower --epochs 10
    python -m recsys.cli.train_two_tower --cpu --epochs 5 --batch-size 512
    python -m recsys.cli.train_two_tower --device cuda --epochs 10  # if CUDA PyTorch installed
    python -m recsys.cli.train_two_tower --best  # use Optuna's saved best params
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import asdict

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
from recsys.data.load import load_anime
from recsys.data.split import load_splits
from recsys.models.popularity_bias import build_popularity_bias_vector
from recsys.training.trainer import build_model_from_features, train


def _load_id_maps() -> tuple[dict[str, int], dict[int, int]]:
    user_map = json.loads((ARTIFACTS_DIR / "user_map.json").read_text())
    anime_map_raw = json.loads((ARTIFACTS_DIR / "anime_map.json").read_text())
    return user_map, {int(k): v for k, v in anime_map_raw.items()}


def main() -> None:
    set_thread_env()
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=CFG.train.epochs)
    p.add_argument("--batch-size", type=int, default=CFG.train.batch_size)
    p.add_argument("--lr", type=float, default=CFG.train.lr)
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="cpu",
        help="Compute device (default: cpu). Use cuda only with a CUDA-enabled PyTorch wheel.",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Train on CPU (shorthand for --device cpu).",
    )
    p.add_argument("--best", action="store_true", help="Load best params from artifacts/best_params.json")
    p.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional cap on training rows (for smoke tests)",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers. 0 = main-thread (GPU starves). 4-8 is usually right on a single GPU.",
    )
    args = p.parse_args()

    device_name = "cpu" if args.cpu else args.device
    device = resolve_device(device_name)
    set_random_seed(device=device)

    train_df, val_df, _ = load_splits()
    user_map, anime_map = _load_id_maps()
    anime_df = load_anime().sort_values("anime_id", kind="stable").reset_index(drop=True)
    feats = build_anime_features(anime_df)
    with open(CACHE_DIR / "user_features.pkl", "rb") as f:
        user_features = pickle.load(f)

    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    if args.best:
        best_payload = json.loads((ARTIFACTS_DIR / "best_params.json").read_text())
        best = best_payload["params"]
        for k, v in best.items():
            if hasattr(train_cfg, k):
                setattr(train_cfg, k, v)
        print(f"Loaded best params: {best}")

    # CLI overrides take precedence over --best (so you can tune at submit
    # time without re-running Optuna).
    if args.num_workers is not None:
        train_cfg.num_workers = args.num_workers
    if os.environ.get("RECSYS_HARD_NEG_K_EASY"):
        train_cfg.hard_neg_K_easy = int(os.environ["RECSYS_HARD_NEG_K_EASY"])

    print(f"Training on device={device}")
    print(f"  config={asdict(train_cfg)}")

    pop_bias = build_popularity_bias_vector(anime_df, anime_map)
    model = build_model_from_features(feats, train_cfg, popularity_bias=pop_bias)
    artifacts = train(
        model=model,
        train_df=train_df,
        val_df=val_df,
        feats=feats,
        user_features=user_features,
        user_map=user_map,
        anime_map=anime_map,
        train_cfg=train_cfg,
        device=device,
        ckpt_path=MODELS_DIR / "two_tower.pt",
        max_train_rows=args.max_train_rows,
    )
    print(f"Best NDCG@10 = {artifacts.best_val:.4f} at epoch {artifacts.best_epoch}")


if __name__ == "__main__":
    main()
