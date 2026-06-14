"""Training loop for the MMoE ranker (Stage 3 of the cascade).

The two-tower retriever is frozen here: we only differentiate through the
heavy ranker and the position-bias side tower. Positives are concatenated
with `neg_per_pos` random negatives per row; the ranker scores all of them in
one batched forward pass and the multi-task loss (Kendall-Gal weighted) is
applied to positive rows only -- negatives get only the implicit "completion
should be near zero, drop should be 1 - completion" signal via binary cross
entropy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from recsys.config import CFG, MODELS_DIR, TrainConfig, set_random_seed
from recsys.data.features_user import UserFeaturePack
from recsys.models.position_bias import PositionBiasTower
from recsys.models.ranker import MMoEConfig, MMoERanker, rank_blend_score
from recsys.models.two_tower import (
    TwoTowerModel,
    encode_all_anime,
    feature_pack_to_tensors,
)
from recsys.retrieval.index import build_index
from recsys.training.ranker_dataset import RankerDataset, make_ranker_collate
from recsys.training.trainer import _build_val_history_batch, encode_history_batch


@dataclass
class RankerTrainArtifacts:
    model: MMoERanker
    position_bias: PositionBiasTower
    best_val: float
    best_epoch: int


def _encode_user_emb_for_batch(
    retriever: TwoTowerModel,
    user_features: UserFeaturePack,
    batch: dict,
    anime_tensors: dict,
    device: torch.device,
    train_cfg: TrainConfig,
) -> torch.Tensor:
    ub = batch["user_idx"].cpu().numpy()
    packed = _build_val_history_batch(user_features, ub, None, train_cfg)
    hist_t = torch.from_numpy(packed["hist"]).to(device)
    mask_t = torch.from_numpy(packed["mask"]).to(device)
    w_t = torch.from_numpy(packed["weights"]).to(device)
    scores_t = torch.from_numpy(packed["scores"]).to(device)
    comp_t = torch.from_numpy(packed["completions"]).to(device)
    days_t = torch.from_numpy(packed["days_ago"]).to(device)
    pooled = encode_history_batch(
        retriever,
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
    return retriever.encode_user(
        pooled, batch["genre_affinity"], batch["centered_score"], batch["recency"]
    )


def _gather_item_emb(
    retriever: TwoTowerModel,
    anime_tensors: dict,
    item_idx: torch.Tensor,
) -> torch.Tensor:
    flat = item_idx.reshape(-1)
    emb = retriever.encode_anime(
        anime_tensors["numerical"][flat],
        anime_tensors["genres"][flat],
        anime_tensors["studio_idx"][flat],
        anime_tensors["synopsis"][flat],
    )
    return emb.view(*item_idx.shape, -1)


@torch.no_grad()
def _build_hard_neg_pool(
    retriever: TwoTowerModel,
    anime_tensors: dict,
    feats: dict,
    m_neighbors: int = 50,
) -> np.ndarray:
    """Item-item nearest-neighbor table over the two-tower item embeddings.

    Row i holds the `m_neighbors` closest catalog items to item i (excluding
    itself). Used by `RankerDataset` to draw hard negatives.
    """
    emb = encode_all_anime(retriever, anime_tensors).cpu().numpy().astype(np.float32)
    n = emb.shape[0]
    index = build_index(
        embeddings=emb,
        anime_ids=feats["anime_ids"],
        nlist=CFG.retrieval.faiss_nlist,
        nprobe=CFG.retrieval.faiss_nprobe,
    )
    # Search m_neighbors + 1 because the top hit is the item itself.
    _, nbrs = index.faiss_index.search(emb, min(m_neighbors + 1, n))
    width = nbrs.shape[1] - 1
    pool = np.empty((n, width), dtype=np.int64)
    for i in range(n):
        row = nbrs[i]
        row = row[(row != i) & (row >= 0)]
        if len(row) < width:
            # Pad short rows (rare, only if FAISS returned -1) with item i so the
            # dataset's exclude-self check simply skips them.
            row = np.concatenate([row, np.full(width - len(row), i, dtype=row.dtype)])
        pool[i] = row[:width]
    return pool


def _rank_logits(
    ranker: MMoERanker,
    user_emb: torch.Tensor,
    pos_item_emb: torch.Tensor,
    neg_item_emb: torch.Tensor,
    w_completion: float,
    w_rating: float,
    w_drop: float,
) -> torch.Tensor:
    """Per-row [positive | negatives] blended scores, shape (B, 1 + neg_k).

    Uses the exact same bounded blend as serving (`rank_blend_score`) so the
    training objective matches what ranks candidates at eval/serve time.
    """
    pos_out = ranker(user_emb, pos_item_emb)
    neg_out = ranker(user_emb, neg_item_emb)
    pos_score = rank_blend_score(pos_out, w_completion, w_rating, w_drop)
    neg_score = rank_blend_score(neg_out, w_completion, w_rating, w_drop)
    return torch.cat([pos_score.unsqueeze(1), neg_score], dim=1)


@torch.no_grad()
def _evaluate_val_auc(
    ranker: MMoERanker,
    retriever: TwoTowerModel,
    val_loader: DataLoader,
    user_features: UserFeaturePack,
    anime_tensors: dict,
    device: torch.device,
    train_cfg: TrainConfig,
    w_completion: float,
    w_rating: float,
    w_drop: float,
) -> float:
    """Ranking AUC = P(positive scored above a sampled negative). Higher is better.

    This is the metric we checkpoint on, because it directly reflects whether the
    ranker orders the true item above plausible alternatives -- unlike train loss.
    """
    ranker.eval()
    wins = 0.0
    total = 0
    for batch in val_loader:
        batch_dev = {
            k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        user_emb = _encode_user_emb_for_batch(
            retriever, user_features, batch_dev, anime_tensors, device, train_cfg
        )
        pos_item_emb = _gather_item_emb(retriever, anime_tensors, batch_dev["pos_anime_idx"])
        neg_item_emb = _gather_item_emb(retriever, anime_tensors, batch_dev["neg_anime_idx"])
        logits = _rank_logits(
            ranker, user_emb, pos_item_emb, neg_item_emb,
            w_completion, w_rating, w_drop,
        )
        pos_score = logits[:, :1]
        neg_score = logits[:, 1:]
        wins += float((pos_score > neg_score).float().sum().item())
        total += int(neg_score.numel())
    return wins / max(total, 1)


def train_ranker(
    retriever: TwoTowerModel,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feats: dict,
    user_features: UserFeaturePack,
    user_map: dict[str, int],
    anime_map: dict[int, int],
    train_cfg: TrainConfig,
    device: torch.device,
    *,
    epochs: int = 5,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    neg_per_pos: int = 4,
    use_position_bias: bool = True,
    ckpt_path: Path | None = None,
    progress: bool = True,
    rank_loss_weight: float = 1.0,
    aux_loss_weight: float = 0.5,
    distill_weight: float = 0.5,
    rank_loss_scale: float = 12.0,
    distill_temp: float = 0.1,
    hard_neg_frac: float = 0.5,
    hard_neg_m: int = 50,
    val_neg_per_pos: int = 20,
    val_max_rows: int = 20_000,
    w_completion: float | None = None,
    w_rating: float | None = None,
    w_drop: float | None = None,
) -> RankerTrainArtifacts:
    set_random_seed(device=device)
    # Serve-time blend weights drive the training ranking objective so the two match.
    w_completion = (
        CFG.retrieval.mmoe_w_completion if w_completion is None else w_completion
    )
    w_rating = CFG.retrieval.mmoe_w_rating if w_rating is None else w_rating
    w_drop = CFG.retrieval.mmoe_w_drop if w_drop is None else w_drop
    retriever = retriever.to(device).eval()
    for p in retriever.parameters():
        p.requires_grad_(False)
    anime_tensors = feature_pack_to_tensors(feats, device)
    n_anime = feats["numerical"].shape[0]

    hard_neg_pool = None
    if hard_neg_frac > 0.0:
        print(f"Building hard-negative neighbor table (top {hard_neg_m})...", flush=True)
        hard_neg_pool = _build_hard_neg_pool(retriever, anime_tensors, feats, hard_neg_m)

    dataset = RankerDataset(
        train_df,
        user_map,
        anime_map,
        user_features,
        n_anime=n_anime,
        neg_per_pos=neg_per_pos,
        hard_neg_pool=hard_neg_pool,
        hard_neg_frac=hard_neg_frac,
    )
    collate = make_ranker_collate(user_features)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        collate_fn=collate,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )

    # Validation set for ranking-AUC checkpoint selection. Random (easy) negatives
    # give a stable, comparable AUC across epochs; a larger neg count smooths it.
    val_df_eval = val_df
    if val_max_rows and len(val_df_eval) > val_max_rows:
        val_df_eval = val_df_eval.sample(val_max_rows, random_state=CFG.seed)
    val_dataset = RankerDataset(
        val_df_eval,
        user_map,
        anime_map,
        user_features,
        n_anime=n_anime,
        neg_per_pos=val_neg_per_pos,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(batch_size // 4, 1),
        shuffle=False,
        num_workers=train_cfg.num_workers,
        collate_fn=collate,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    cfg = MMoEConfig(
        embedding_dim=train_cfg.embedding_dim,
        bottom_hidden=train_cfg.hidden_dim,
        expert_hidden=train_cfg.hidden_dim // 2,
        n_experts=4,
        n_heads=3,
        side_dim=0,
        dropout=train_cfg.dropout,
    )
    ranker = MMoERanker(cfg).to(device)
    position_bias = PositionBiasTower().to(device) if use_position_bias else None

    params = list(ranker.parameters())
    if position_bias is not None:
        params += list(position_bias.parameters())
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = -1.0  # ranker eval = val ranking AUC; higher is better
    best_epoch = -1
    ckpt_path = ckpt_path or (MODELS_DIR / "ranker.pt")

    for epoch in range(1, epochs + 1):
        ranker.train()
        if position_bias is not None:
            position_bias.train()
        running = 0.0
        n_batches = 0
        iterator = tqdm(loader, disable=not progress, desc=f"ranker epoch {epoch}", leave=False)
        for batch in iterator:
            batch_dev = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            optim.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                with torch.no_grad():
                    user_emb = _encode_user_emb_for_batch(
                        retriever, user_features, batch_dev, anime_tensors, device, train_cfg
                    )
                    pos_item_emb = _gather_item_emb(
                        retriever, anime_tensors, batch_dev["pos_anime_idx"]
                    )
                    neg_item_emb = _gather_item_emb(
                        retriever, anime_tensors, batch_dev["neg_anime_idx"]
                    )

                pos_preds = ranker(user_emb, pos_item_emb)
                pos_targets = {
                    "completion": batch_dev["pos_completion"],
                    "rating": batch_dev["pos_rating_z"],
                    "drop": batch_dev["pos_drop"],
                }

                # Add the position-bias logit to the completion head only (the
                # head most affected by display position in production logs).
                if position_bias is not None:
                    pb_shift = position_bias(
                        batch_dev["pos_position"],
                        batch_dev["pos_slate_size"],
                        torch.zeros_like(batch_dev["pos_position"]),
                        training=True,
                    )
                    pos_preds_shifted = dict(pos_preds)
                    pos_preds_shifted["completion"] = pos_preds["completion"] + pb_shift
                else:
                    pos_preds_shifted = pos_preds

                loss_pos, report = ranker.loss(pos_preds_shifted, pos_targets)

                # Implicit negatives: random items should be predicted as
                # low-completion and high-drop. We use BCE with constant
                # targets; rating head gets no negative signal (the user simply
                # never rated it).
                neg_preds = ranker(user_emb, neg_item_emb)
                neg_comp = neg_preds["completion"]
                neg_drop = neg_preds["drop"]
                neg_loss = F.binary_cross_entropy_with_logits(
                    neg_comp, torch.zeros_like(neg_comp)
                ) + F.binary_cross_entropy_with_logits(
                    neg_drop, torch.zeros_like(neg_drop)
                )
                aux_loss = loss_pos + 0.2 * neg_loss

                # --- Ranking loss: the blended serve score of the positive must
                # beat its negatives (sampled-softmax / listwise CE). This is the
                # term that aligns the ranker with recall/NDCG instead of only
                # the per-head regression/classification targets. ---
                pos_score = rank_blend_score(pos_preds, w_completion, w_rating, w_drop)
                neg_score = rank_blend_score(neg_preds, w_completion, w_rating, w_drop)
                logits = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1) * rank_loss_scale
                target = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
                rank_loss = F.cross_entropy(logits, target)

                # --- Distillation: anchor the ranking to the two-tower retrieval
                # ordering so the ranker re-orders within, rather than fighting,
                # the retriever. Teacher = dot-product softmax over [pos, negs]. ---
                pos_dot = (user_emb * pos_item_emb).sum(-1)
                neg_dot = (user_emb.unsqueeze(1) * neg_item_emb).sum(-1)
                teacher_logits = torch.cat([pos_dot.unsqueeze(1), neg_dot], dim=1) / distill_temp
                teacher_p = F.softmax(teacher_logits, dim=1)
                distill_loss = -(teacher_p * F.log_softmax(logits, dim=1)).sum(1).mean()

                loss = (
                    rank_loss_weight * rank_loss
                    + distill_weight * distill_loss
                    + aux_loss_weight * aux_loss
                )

            scaler.scale(loss).backward()
            if train_cfg.grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(params, train_cfg.grad_clip)
            scaler.step(optim)
            scaler.update()

            running += float(loss.item())
            n_batches += 1
            if progress:
                iterator.set_postfix(
                    loss=f"{running / max(n_batches, 1):.4f}",
                    rank=f"{float(rank_loss.detach()):.3f}",
                    dist=f"{float(distill_loss.detach()):.3f}",
                    aux=f"{float(aux_loss.detach()):.3f}",
                )

        avg_loss = running / max(n_batches, 1)
        val_auc = _evaluate_val_auc(
            ranker, retriever, val_loader, user_features, anime_tensors, device,
            train_cfg, w_completion, w_rating, w_drop,
        )
        if progress:
            print(
                f"[ranker epoch {epoch}] loss={avg_loss:.4f} val_auc={val_auc:.4f}",
                flush=True,
            )
        if val_auc > best_val:
            best_val = val_auc
            best_epoch = epoch
            torch.save(
                {
                    "model_state": ranker.state_dict(),
                    "position_bias_state": (
                        position_bias.state_dict() if position_bias is not None else None
                    ),
                    "cfg": vars(cfg),
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_auc": val_auc,
                },
                ckpt_path,
            )

    return RankerTrainArtifacts(
        model=ranker,
        position_bias=position_bias if position_bias is not None else PositionBiasTower(),
        best_val=best_val,
        best_epoch=best_epoch,
    )


def load_ranker(
    path: Path,
    map_location: str | torch.device = "cpu",
) -> tuple[MMoERanker, PositionBiasTower | None]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = MMoEConfig(**ckpt["cfg"])
    model = MMoERanker(cfg)
    model.load_state_dict(ckpt["model_state"])
    pb = None
    if ckpt.get("position_bias_state") is not None:
        pb = PositionBiasTower()
        pb.load_state_dict(ckpt["position_bias_state"])
    return model, pb
