from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from recsys.config import ARTIFACTS_DIR, CFG, TrainConfig, set_random_seed
from recsys.data.features_user import UserFeaturePack
from recsys.training.trainer import build_model_from_features, stabilize_train_config, train


def make_objective(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feats: dict,
    user_features: UserFeaturePack,
    user_map: dict[str, int],
    anime_map: dict[int, int],
    popularity_bias: np.ndarray,
    device: torch.device,
    base_epochs: int = 6,
    max_train_rows: int | None = None,
):
    def objective(trial: optuna.Trial) -> float:
        print(f"[tune] trial {trial.number} starting...", flush=True)
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
            use_sequence_encoder=CFG.train.use_sequence_encoder,
            seq_n_layers=CFG.train.seq_n_layers,
            seq_n_heads=CFG.train.seq_n_heads,
            seq_ffn_mult=CFG.train.seq_ffn_mult,
            seq_p_mask_recent=trial.suggest_float("seq_p_mask_recent", 0.0, 0.5)
            if CFG.train.use_sequence_encoder
            else 0.0,
            use_completion_weighted_loss=CFG.train.use_completion_weighted_loss,
            completion_floor=CFG.train.completion_floor,
            hard_neg_curriculum=CFG.train.hard_neg_curriculum,
            hard_neg_K_easy=CFG.train.hard_neg_K_easy,
            hard_neg_K_hard=CFG.train.hard_neg_K_hard,
        )
        notes = stabilize_train_config(cfg)
        if notes:
            print(f"[tune] trial {trial.number} stabilized: {', '.join(notes)}", flush=True)
        print(
            f"[tune] trial {trial.number} params: batch_size={cfg.batch_size} "
            f"lr={cfg.lr:.4g} temp={cfg.temperature} hard_neg_ratio={cfg.hard_neg_ratio}",
            flush=True,
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
            device=device,
            ckpt_path=ARTIFACTS_DIR / "models" / f"trial_{trial.number}.pt",
            progress=True,
            max_train_rows=max_train_rows,
        )
        if artifacts.best_val < 0:
            raise optuna.TrialPruned()
        print(
            f"[tune] trial {trial.number} done: best_val_ndcg@10={artifacts.best_val:.4f} "
            f"(epoch {artifacts.best_epoch})",
            flush=True,
        )
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
    device: torch.device | None = None,
) -> dict:
    device = device or CFG.device
    set_random_seed(device=device)
    sampler = TPESampler(seed=CFG.tune.sampler_seed)
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
        device=device,
        max_train_rows=max_train_rows,
    )
    def _on_trial_done(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        val = trial.value
        val_s = f"{val:.4f}" if val is not None else "pruned/failed"
        print(f"[tune] Optuna recorded trial {trial.number}: value={val_s}", flush=True)

    print(
        f"[tune] starting study: {n_trials} trials, max_train_rows={max_train_rows}, "
        f"device={device}, hard_neg_K_easy={CFG.train.hard_neg_K_easy}",
        flush=True,
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[_on_trial_done])
    best = {
        "value": study.best_value,
        "params": study.best_params,
        "n_trials": n_trials,
        "max_train_rows": max_train_rows,
    }
    out_path = out_path or ARTIFACTS_DIR / "best_params.json"
    out_path.write_text(json.dumps(best, indent=2))
    return best
