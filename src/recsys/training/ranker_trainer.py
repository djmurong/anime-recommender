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
from recsys.models.ranker import MMoEConfig, MMoERanker
from recsys.models.two_tower import (
    TwoTowerModel,
    feature_pack_to_tensors,
)
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
) -> RankerTrainArtifacts:
    set_random_seed(device=device)
    retriever = retriever.to(device).eval()
    for p in retriever.parameters():
        p.requires_grad_(False)
    anime_tensors = feature_pack_to_tensors(feats, device)
    n_anime = feats["numerical"].shape[0]

    dataset = RankerDataset(
        train_df,
        user_map,
        anime_map,
        user_features,
        n_anime=n_anime,
        neg_per_pos=neg_per_pos,
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

    best_val = float("inf")  # ranker eval = val loss; lower is better
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
                loss = loss_pos + 0.2 * neg_loss

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
                    c=f"{report['loss/completion']:.3f}",
                    r=f"{report['loss/rating']:.3f}",
                    d=f"{report['loss/drop']:.3f}",
                )

        avg_loss = running / max(n_batches, 1)
        if progress:
            print(f"[ranker epoch {epoch}] loss={avg_loss:.4f}")
        if avg_loss < best_val:
            best_val = avg_loss
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
