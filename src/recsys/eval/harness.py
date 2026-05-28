"""Offline evaluation harness.

Two paths exist for the two-tower / cascade model:
  * `evaluate_two_tower`: legacy brute-force full-catalog topk. Useful as a
    ceiling on what the retriever can in principle return.
  * `evaluate_cascade`: runs the same model through the four-stage cascade
    (Retrieve -> PreRank -> Rank -> ReRank). This is what production sees and
    is the eval row reported in `artifacts/eval.md` for cascade variants.

Baselines (Popularity, ContentCosine, ImplicitMF) keep their existing full-
catalog path on the completed-only data slice so their numbers stay comparable
across the refactor.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

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
from recsys.models.two_tower import (
    TwoTowerModel,
    encode_all_anime,
    feature_pack_to_tensors,
    score_all_items,
)
from recsys.retrieval.cascade import Cascade
from recsys.training.trainer import _build_val_history_batch, encode_history_batch


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


def _encode_user_embeddings_for_batch(
    model: TwoTowerModel,
    user_features: UserFeaturePack,
    user_indices: np.ndarray,
    anime_tensors: dict,
    device: torch.device,
    train_cfg=CFG.train,
) -> torch.Tensor:
    packed = _build_val_history_batch(user_features, user_indices, None, train_cfg)
    affinity = torch.from_numpy(user_features.genre_affinity).to(device)
    centered = torch.from_numpy(user_features.centered_avg_score).unsqueeze(-1).to(device)
    recency = torch.from_numpy(user_features.recency).to(device)
    hist_t = torch.from_numpy(packed["hist"]).to(device)
    mask_t = torch.from_numpy(packed["mask"]).to(device)
    w_t = torch.from_numpy(packed["weights"]).to(device)
    scores_t = torch.from_numpy(packed["scores"]).to(device)
    comp_t = torch.from_numpy(packed["completions"]).to(device)
    days_t = torch.from_numpy(packed["days_ago"]).to(device)
    pooled = encode_history_batch(
        model,
        hist_t,
        mask_t,
        anime_tensors,
        w_t,
        history_scores=scores_t,
        history_completion=comp_t,
        history_days_ago=days_t,
        train_cfg=train_cfg,
        training_mask_prob=0.0,
    )
    ub_t = torch.from_numpy(np.asarray(user_indices, dtype=np.int64)).to(device)
    return model.encode_user(pooled, affinity[ub_t], centered[ub_t], recency[ub_t]), packed


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
    """Brute-force full-catalog top-k. Kept for ceiling comparison."""
    device = CFG.device
    model = model.to(device).eval()
    anime_tensors = feature_pack_to_tensors(feats, device)
    u_idx, target = _build_test_pairs(test_df, user_map, anime_map, max_users)

    recalls, precs, ndcgs, ilds, serens = [], [], [], [], []
    rec_lists: list[np.ndarray] = []
    with torch.no_grad():
        all_anime = encode_all_anime(model, anime_tensors)
        batch = 256
        for start in range(0, len(u_idx), batch):
            ub = u_idx[start : start + batch]
            tb = target[start : start + batch]
            user_emb, packed = _encode_user_embeddings_for_batch(
                model, user_features, ub, anime_tensors, device
            )
            scores = score_all_items(model, user_emb, all_anime)
            for i in range(len(ub)):
                pos = packed["histories"][i]
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


def evaluate_cascade(
    name: str,
    model: TwoTowerModel,
    cascade: Cascade,
    feats: dict,
    test_df: pd.DataFrame,
    user_features: UserFeaturePack,
    user_map: dict[str, int],
    anime_map: dict[int, int],
    item_features_for_ild: np.ndarray,
    popularity_rank: np.ndarray,
    k: int = 10,
    max_users: int = 5000,
) -> EvalResult:
    """Run the same model through the four-stage cascade and score.

    Differences vs `evaluate_two_tower`: candidate pool is bounded by
    `cascade.pool_retrieve` (FAISS) instead of the full catalog, the prerank +
    rank stages can re-order, and the reranker (MMR / DPP) shapes the final
    list. This is what production sees, so the metrics here are the canonical
    cascade row in `artifacts/eval.md`.
    """
    device = cascade.device
    model = model.to(device).eval()
    anime_tensors = feature_pack_to_tensors(feats, device)
    u_idx, target = _build_test_pairs(test_df, user_map, anime_map, max_users)

    recalls, precs, ndcgs, ilds, serens = [], [], [], [], []
    rec_lists: list[np.ndarray] = []
    with torch.no_grad():
        # Encode users in batches; cascade itself runs per-user (small overhead).
        batch = 128
        for start in range(0, len(u_idx), batch):
            ub = u_idx[start : start + batch]
            tb = target[start : start + batch]
            user_emb, packed = _encode_user_embeddings_for_batch(
                model, user_features, ub, anime_tensors, device
            )
            for i in range(len(ub)):
                pos = packed["histories"][i]
                excl = set(int(x) for x in pos.tolist()) if len(pos) else set()
                final_idxs, _ = cascade.recommend(
                    user_emb=user_emb[i],
                    excluded_anime_indices=excl,
                    k=k,
                )
                if len(final_idxs) < k:
                    # Pad with -1 so per-row metrics are still defined; these never match.
                    final_idxs = np.concatenate(
                        [final_idxs, -np.ones(k - len(final_idxs), dtype=np.int64)]
                    )
                preds = final_idxs.astype(np.int64)
                recalls.append(recall_at_k(preds, [int(tb[i])], k))
                precs.append(precision_at_k(preds, [int(tb[i])], k))
                ndcgs.append(ndcg_at_k(preds, [int(tb[i])], k))
                # ILD / coverage / serendipity expect valid indices; drop padding.
                valid = preds[preds >= 0]
                ilds.append(intra_list_diversity(valid, item_features_for_ild))
                serens.append(serendipity(valid, [int(tb[i])], popularity_rank, k))
                rec_lists.append(valid)

    return EvalResult(
        name=name,
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
