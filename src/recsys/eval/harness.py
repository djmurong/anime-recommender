from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recsys.config import ARTIFACTS_DIR, CFG
from recsys.data.features_user import UserFeaturePack
from recsys.eval.metrics import (
    catalog_coverage,
    intra_list_diversity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    serendipity,
)
from recsys.models.baselines import Recommender
from recsys.models.two_tower import TwoTowerModel, encode_all_anime, feature_pack_to_tensors, score_all_items
from recsys.training.trainer import encode_history_batch


@dataclass
class EvalResult:
    name: str
    recall10: float
    precision10: float
    ndcg10: float
    ild: float
    coverage: float
    serendipity: float


def _build_test_pairs(
    test_df: pd.DataFrame,
    user_map: dict[str, int],
    anime_map: dict[int, int],
    max_users: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    df = test_df.copy()
    df["user_idx"] = df["username"].map(user_map)
    df["anime_idx"] = df["anime_id"].astype(int).map(anime_map)
    df = df.dropna(subset=["user_idx", "anime_idx"])
    df["user_idx"] = df["user_idx"].astype("int64")
    df["anime_idx"] = df["anime_idx"].astype("int64")
    if len(df) > max_users:
        df = df.sample(max_users, random_state=CFG.seed)
    return df["user_idx"].to_numpy(), df["anime_idx"].to_numpy()


def evaluate_baseline(
    rec: Recommender,
    test_df: pd.DataFrame,
    user_features: UserFeaturePack,
    user_map: dict[str, int],
    anime_map: dict[int, int],
    item_features_for_ild: np.ndarray,
    popularity_rank: np.ndarray,
    k: int = 10,
    max_users: int = 5000,
) -> EvalResult:
    u_idx, target = _build_test_pairs(test_df, user_map, anime_map, max_users)
    recalls, precs, ndcgs, ilds, serens = [], [], [], [], []
    rec_lists: list[np.ndarray] = []
    for u, t in zip(u_idx, target):
        excl = user_features.history.get(int(u), np.zeros(0, dtype=np.int64))
        preds, _ = rec.recommend(int(u), k, excl)
        recalls.append(recall_at_k(preds, [int(t)], k))
        precs.append(precision_at_k(preds, [int(t)], k))
        ndcgs.append(ndcg_at_k(preds, [int(t)], k))
        ilds.append(intra_list_diversity(preds, item_features_for_ild))
        serens.append(serendipity(preds, [int(t)], popularity_rank, k))
        rec_lists.append(preds)
    return EvalResult(
        name=rec.name,
        recall10=float(np.mean(recalls)) if recalls else 0.0,
        precision10=float(np.mean(precs)) if precs else 0.0,
        ndcg10=float(np.mean(ndcgs)) if ndcgs else 0.0,
        ild=float(np.mean(ilds)) if ilds else 0.0,
        coverage=catalog_coverage(rec_lists, item_features_for_ild.shape[0]),
        serendipity=float(np.mean(serens)) if serens else 0.0,
    )


def evaluate_two_tower(
    model: TwoTowerModel,
    feats: dict,
    test_df: pd.DataFrame,
    user_features: UserFeaturePack,
    user_map: dict[str, int],
    anime_map: dict[int, int],
    item_features_for_ild: np.ndarray,
    popularity_rank: np.ndarray,
    k: int = 10,
    max_users: int = 5000,
    max_history_len: int | None = None,
    use_score_weighted_pool: bool | None = None,
) -> EvalResult:
    max_history_len = max_history_len if max_history_len is not None else CFG.train.max_history_len
    use_score_weighted_pool = (
        use_score_weighted_pool
        if use_score_weighted_pool is not None
        else CFG.train.use_score_weighted_pool
    )

    device = CFG.device
    model = model.to(device).eval()
    anime_tensors = feature_pack_to_tensors(feats, device)
    u_idx, target = _build_test_pairs(test_df, user_map, anime_map, max_users)

    affinity = torch.from_numpy(user_features.genre_affinity).to(device)
    centered = torch.from_numpy(user_features.centered_avg_score).unsqueeze(-1).to(device)
    recency = torch.from_numpy(user_features.recency).to(device)

    recalls, precs, ndcgs, ilds, serens = [], [], [], [], []
    rec_lists: list[np.ndarray] = []
    with torch.no_grad():
        all_anime = encode_all_anime(model, anime_tensors)
        batch = 256
        for start in range(0, len(u_idx), batch):
            ub = u_idx[start : start + batch]
            tb = target[start : start + batch]
            histories = [user_features.history.get(int(u), np.zeros(0, dtype=np.int64)) for u in ub]
            hist_scores = [
                user_features.history_scores.get(int(u), np.zeros(0, dtype=np.float32)) for u in ub
            ]
            max_h = max((len(h) for h in histories), default=1)
            max_h = max(max_h, 1)
            hist = np.zeros((len(ub), max_h), dtype=np.int64)
            mask = np.zeros((len(ub), max_h), dtype=np.float32)
            weights = np.zeros((len(ub), max_h), dtype=np.float32)
            for i, h in enumerate(histories):
                if len(h) == 0:
                    continue
                sc = hist_scores[i]
                if len(h) > max_history_len:
                    h = h[-max_history_len:]
                    sc = sc[-max_history_len:]
                hist[i, : len(h)] = h
                mask[i, : len(h)] = 1.0
                if use_score_weighted_pool:
                    mu = float(user_features.centered_avg_score[int(ub[i])]) + 7.0
                    weights[i, : len(h)] = np.maximum(sc - mu, 0.1)
                else:
                    weights[i, : len(h)] = 1.0
            hist_t = torch.from_numpy(hist).to(device)
            mask_t = torch.from_numpy(mask).to(device)
            w_t = torch.from_numpy(weights).to(device) if use_score_weighted_pool else None
            pooled = encode_history_batch(model, hist_t, mask_t, anime_tensors, w_t)
            ub_t = torch.from_numpy(np.asarray(ub, dtype=np.int64)).to(device)
            user_emb = model.encode_user(pooled, affinity[ub_t], centered[ub_t], recency[ub_t])
            scores = score_all_items(model, user_emb, all_anime)
            for i in range(len(ub)):
                pos = histories[i]
                if len(pos):
                    scores[i, pos] = -float("inf")
            topk = scores.topk(k, dim=1).indices.cpu().numpy()
            for i, t_idx in enumerate(tb):
                preds = topk[i]
                recalls.append(recall_at_k(preds, [int(t_idx)], k))
                precs.append(precision_at_k(preds, [int(t_idx)], k))
                ndcgs.append(ndcg_at_k(preds, [int(t_idx)], k))
                ilds.append(intra_list_diversity(preds, item_features_for_ild))
                serens.append(serendipity(preds, [int(t_idx)], popularity_rank, k))
                rec_lists.append(preds)

    return EvalResult(
        name="TwoTower",
        recall10=float(np.mean(recalls)) if recalls else 0.0,
        precision10=float(np.mean(precs)) if precs else 0.0,
        ndcg10=float(np.mean(ndcgs)) if ndcgs else 0.0,
        ild=float(np.mean(ilds)) if ilds else 0.0,
        coverage=catalog_coverage(rec_lists, item_features_for_ild.shape[0]),
        serendipity=float(np.mean(serens)) if serens else 0.0,
    )


def format_table(results: list[EvalResult]) -> str:
    header = "| Model | Recall@10 | Precision@10 | NDCG@10 | ILD | Coverage | Serendipity |\n"
    sep = "|---|---|---|---|---|---|---|\n"
    rows = []
    for r in results:
        rows.append(
            f"| {r.name} | {r.recall10:.4f} | {r.precision10:.4f} | {r.ndcg10:.4f} | {r.ild:.4f} | {r.coverage:.4f} | {r.serendipity:.4f} |"
        )
    return header + sep + "\n".join(rows) + "\n"


def write_report(results: list[EvalResult], path: Path | None = None) -> Path:
    path = path or (ARTIFACTS_DIR / "eval.md")
    path.write_text(format_table(results))
    return path
