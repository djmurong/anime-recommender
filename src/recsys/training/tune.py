from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from recsys.config import ARTIFACTS_DIR, CFG, TrainConfig
from recsys.data.features_user import UserFeaturePack
from recsys.training.trainer import build_model_from_features, train


def make_objective(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feats: dict,
    user_features: UserFeaturePack,
    user_map: dict[str, int],
    anime_map: dict[int, int],
    popularity_bias: np.ndarray,
    base_epochs: int = 6,
    max_train_rows: int | None = None,
):
    def objective(trial: optuna.Trial) -> float:
        cfg = TrainConfig(
            embedding_dim=trial.suggest_categorical("embedding_dim", [64, 128, 256]),
            hidden_dim=256,
            dropout=trial.suggest_float("dropout", 0.1, 0.3),
            studio_emb_dim=CFG.train.studio_emb_dim,
            temperature=trial.suggest_categorical("temperature", [0.03, 0.05, 0.1, 0.2]),
            batch_size=trial.suggest_categorical("batch_size", [512, 1024, 2048]),
            lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            weight_decay=1e-6,
            epochs=base_epochs,
            hard_neg_ratio=trial.suggest_categorical("hard_neg_ratio", [0, 2, 4, 8]),
            hard_neg_start_epoch=2,
            hard_neg_refresh_every=2,
            val_every_epoch=1,
            grad_clip=1.0,
            num_workers=0,
            catalog_neg_count=trial.suggest_categorical("catalog_neg_count", [32, 64, 128]),
            catalog_neg_weight=1.0,
            recency_sample_tau_days=CFG.train.recency_sample_tau_days,
            max_history_len=CFG.train.max_history_len,
            use_score_weighted_pool=True,
        )
        model = build_model_from_features(feats, cfg, popularity_bias=popularity_bias)
        artifacts = train(
            model=model,
            train_df=train_df,
            val_df=val_df,
            feats=feats,
            user_features=user_features,
            user_map=user_map,
            anime_map=anime_map,
            train_cfg=cfg,
            device=CFG.device,
            ckpt_path=ARTIFACTS_DIR / "models" / f"trial_{trial.number}.pt",
            progress=False,
            max_train_rows=max_train_rows,
        )
        if artifacts.best_val < 0:
            raise optuna.TrialPruned()
        return artifacts.best_val

    return objective


def run_study(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feats: dict,
    user_features: UserFeaturePack,
    user_map: dict[str, int],
    anime_map: dict[int, int],
    popularity_bias: np.ndarray,
    n_trials: int = 30,
    out_path: Path | None = None,
    max_train_rows: int | None = 2_000_000,
) -> dict:
    sampler = TPESampler(seed=CFG.seed)
    pruner = MedianPruner(n_warmup_steps=CFG.tune.pruner_warmup_epochs)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    objective = make_objective(
        train_df,
        val_df,
        feats,
        user_features,
        user_map,
        anime_map,
        popularity_bias,
        max_train_rows=max_train_rows,
    )
    study.optimize(objective, n_trials=n_trials)
    best = {
        "value": study.best_value,
        "params": study.best_params,
        "n_trials": n_trials,
        "max_train_rows": max_train_rows,
    }
    out_path = out_path or ARTIFACTS_DIR / "best_params.json"
    out_path.write_text(json.dumps(best, indent=2))
    return best
