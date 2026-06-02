from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from recsys.config import CFG, MODELS_DIR, TrainConfig, set_random_seed
from recsys.data.features_user import RECENCY_BUCKETS, UserFeaturePack
from recsys.eval.metrics import ndcg_at_k, recall_at_k
from recsys.models.losses import combined_ranking_loss
from recsys.models.two_tower import (
    TowerDims,
    TwoTowerModel,
    encode_all_anime,
    feature_pack_to_tensors,
    score_all_items,
)
from recsys.training.dataset import InteractionDataset, make_collate, make_train_sampler
from recsys.training.hard_negatives import HardNegativeMiner


@dataclass
class TrainArtifacts:
    model: TwoTowerModel
    best_val: float
    best_epoch: int


def build_model_from_features(
    feats: dict,
    train_cfg: TrainConfig,
    recency_dim: int = RECENCY_BUCKETS,
    popularity_bias: np.ndarray | None = None,
) -> TwoTowerModel:
    n_anime = feats["numerical"].shape[0]
    dims = TowerDims(
        n_studios=len(feats["studio_vocab"]),
        studio_emb_dim=train_cfg.studio_emb_dim,
        n_genres=feats["genres"].shape[1],
        n_numerical=feats["numerical"].shape[1],
        synopsis_dim=feats["synopsis"].shape[1],
        embedding_dim=train_cfg.embedding_dim,
        hidden_dim=train_cfg.hidden_dim,
        dropout=train_cfg.dropout,
        n_anime=n_anime,
        use_sequence_encoder=train_cfg.use_sequence_encoder,
        seq_n_layers=train_cfg.seq_n_layers,
        seq_n_heads=train_cfg.seq_n_heads,
        seq_ffn_mult=train_cfg.seq_ffn_mult,
        seq_max_history=max(train_cfg.max_history_len, 16),
    )
    return TwoTowerModel(dims, recency_dim=recency_dim, popularity_bias=popularity_bias)


def encode_history_batch(
    model: TwoTowerModel,
    history_idx: torch.Tensor,
    history_mask: torch.Tensor,
    anime_tensors: dict,
    history_weights: torch.Tensor | None = None,
    *,
    history_scores: torch.Tensor | None = None,
    history_completion: torch.Tensor | None = None,
    history_days_ago: torch.Tensor | None = None,
    train_cfg: TrainConfig | None = None,
    training_mask_prob: float = 0.0,
    history_emb: torch.Tensor | None = None,
) -> torch.Tensor:
    """Embed each history slot and reduce to a single per-user vector.

    Routes through the SequenceEncoder when the model has one and we were given
    the auxiliary signals (scores, completion, days_ago); otherwise falls back
    to the weighted mean pool.

    ``history_emb``: optional precomputed [B, T, D] from `_batch_encode_anime_by_index`.
    """
    if history_emb is not None:
        emb = history_emb
    else:
        b, t = history_idx.shape
        flat = history_idx.reshape(-1)
        emb = model.encode_anime(
            anime_tensors["numerical"][flat],
            anime_tensors["genres"][flat],
            anime_tensors["studio_idx"][flat],
            anime_tensors["synopsis"][flat],
        )
        emb = emb.view(b, t, -1)

    can_use_seq = (
        model.sequence_encoder is not None
        and history_scores is not None
        and history_completion is not None
        and history_days_ago is not None
    )
    if can_use_seq:
        mask_window = (
            CFG.train.seq_mask_window_min if train_cfg is None else train_cfg.seq_mask_window_min,
            CFG.train.seq_mask_window_max if train_cfg is None else train_cfg.seq_mask_window_max,
        )
        return model.encode_history_sequence(
            history_emb=emb,
            history_mask=history_mask,
            history_scores=history_scores,
            history_completion=history_completion,
            days_ago=history_days_ago,
            training_mask_prob=training_mask_prob,
            mask_window=mask_window,
        )
    return model.pool_history(emb, history_mask, history_weights)


def _gather_anime_features(anime_tensors: dict, idx: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return (
        anime_tensors["numerical"][idx],
        anime_tensors["genres"][idx],
        anime_tensors["studio_idx"][idx],
        anime_tensors["synopsis"][idx],
    )


def _batch_encode_anime_by_index(
    model: TwoTowerModel,
    anime_tensors: dict,
    parts: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """One deduplicated item-tower pass for all index tensors (same embeddings)."""
    if not parts:
        return {}
    names: list[str] = []
    flats: list[torch.Tensor] = []
    shapes: dict[str, tuple[int, ...]] = {}
    for name, idx in parts.items():
        shapes[name] = tuple(idx.shape)
        flats.append(idx.reshape(-1))
        names.append(name)
    all_idx = torch.cat(flats)
    unique_idx, inverse = torch.unique(all_idx, return_inverse=True)
    unique_emb = model.encode_anime(*_gather_anime_features(anime_tensors, unique_idx))
    out: dict[str, torch.Tensor] = {}
    offset = 0
    for name in names:
        n_elem = int(np.prod(shapes[name]))
        inv = inverse[offset : offset + n_elem]
        emb = unique_emb[inv]
        out[name] = emb.view(*shapes[name], -1)
        offset += n_elem
    return out


def _sample_catalog_negatives(
    n_anime: int,
    positives_per_user: list[set[int]],
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorized rejection sampling — same distribution as the legacy Python loop."""
    b = len(positives_per_user)
    out = np.empty((b, k), dtype=np.int64)
    chunk = max(k * 8, 256)
    for i, pos in enumerate(positives_per_user):
        if not pos:
            out[i] = rng.integers(0, n_anime, size=k, dtype=np.int64)
            continue
        pos_arr = np.fromiter(pos, count=len(pos), dtype=np.int64)
        chosen: list[int] = []
        while len(chosen) < k:
            cands = rng.integers(0, n_anime, size=chunk, dtype=np.int64)
            good = cands[~np.isin(cands, pos_arr, assume_unique=False)]
            if chosen:
                good = good[~np.isin(good, np.asarray(chosen, dtype=np.int64), assume_unique=False)]
            for c in good:
                chosen.append(int(c))
                if len(chosen) >= k:
                    break
        out[i] = np.asarray(chosen[:k], dtype=np.int64)
    return out


def _build_val_history_batch(
    user_features: UserFeaturePack,
    user_indices: np.ndarray,
    target_ts_ns: np.ndarray | None,
    train_cfg: TrainConfig,
) -> dict[str, np.ndarray]:
    """Pack each user's causal history into padded numpy arrays for eval/val."""
    max_history = train_cfg.max_history_len
    full = user_features.history_full
    full_scores = user_features.history_full_scores
    full_completion = user_features.history_full_completion
    full_ts = user_features.history_full_ts_ns
    has_full = (
        full is not None
        and full_scores is not None
        and full_completion is not None
        and full_ts is not None
    )

    histories: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    completions: list[np.ndarray] = []
    days: list[np.ndarray] = []
    for i, u in enumerate(user_indices):
        u_int = int(u)
        if has_full:
            h_all = full.get(u_int, np.zeros(0, dtype=np.int64))
            s_all = full_scores.get(u_int, np.zeros(0, dtype=np.float32))
            c_all = full_completion.get(u_int, np.zeros(0, dtype=np.float32))
            ts_all = full_ts.get(u_int, np.zeros(0, dtype=np.int64))
        else:
            h_all = user_features.history.get(u_int, np.zeros(0, dtype=np.int64))
            s_all = user_features.history_scores.get(u_int, np.zeros(0, dtype=np.float32))
            c_all = (s_all > 0).astype(np.float32)
            ts_all = np.zeros(len(h_all), dtype=np.int64)

        if target_ts_ns is not None and len(ts_all):
            target = int(target_ts_ns[i])
            end = int(np.searchsorted(ts_all, target, side="left"))
            h_all = h_all[:end]
            s_all = s_all[:end]
            c_all = c_all[:end]
            ts_all = ts_all[:end]

        if len(h_all) > max_history:
            h_all = h_all[-max_history:]
            s_all = s_all[-max_history:]
            c_all = c_all[-max_history:]
            ts_all = ts_all[-max_history:]

        ref_ts = int(ts_all.max()) if len(ts_all) else 0
        d_all = (
            (ref_ts - ts_all) / (86400.0 * 1e9)
            if len(ts_all)
            else np.zeros(0, dtype=np.float32)
        ).astype(np.float32)

        histories.append(h_all.astype(np.int64))
        scores.append(s_all.astype(np.float32))
        completions.append(c_all.astype(np.float32))
        days.append(d_all)

    b = len(user_indices)
    max_h = max((len(h) for h in histories), default=1)
    max_h = max(max_h, 1)
    hist = np.zeros((b, max_h), dtype=np.int64)
    mask = np.zeros((b, max_h), dtype=np.float32)
    weights = np.zeros((b, max_h), dtype=np.float32)
    score_arr = np.zeros((b, max_h), dtype=np.float32)
    comp_arr = np.zeros((b, max_h), dtype=np.float32)
    days_arr = np.zeros((b, max_h), dtype=np.float32)
    for i, h in enumerate(histories):
        if len(h) == 0:
            continue
        hist[i, : len(h)] = h
        mask[i, : len(h)] = 1.0
        score_arr[i, : len(h)] = scores[i]
        comp_arr[i, : len(h)] = completions[i]
        days_arr[i, : len(h)] = days[i]
        if train_cfg.use_score_weighted_pool:
            mu = float(user_features.centered_avg_score[int(user_indices[i])]) + 7.0
            weights[i, : len(h)] = np.maximum(scores[i] - mu, 0.1)
        else:
            weights[i, : len(h)] = 1.0
    return {
        "histories": histories,
        "hist": hist,
        "mask": mask,
        "weights": weights,
        "scores": score_arr,
        "completions": comp_arr,
        "days_ago": days_arr,
    }


def _val_metrics(
    model: TwoTowerModel,
    anime_tensors: dict,
    val: pd.DataFrame,
    user_features: UserFeaturePack,
    user_map: dict[str, int],
    anime_map: dict[int, int],
    train_cfg: TrainConfig,
    k: int = 10,
    max_users: int = 5000,
) -> dict[str, float]:
    if val.empty:
        return {"recall@10": 0.0, "ndcg@10": 0.0}

    val = val.copy()
    val["user_idx"] = val["username"].map(user_map)
    val["anime_idx"] = val["anime_id"].astype(int).map(anime_map)
    val = val.dropna(subset=["user_idx", "anime_idx"])
    val["user_idx"] = val["user_idx"].astype("int64")
    val["anime_idx"] = val["anime_idx"].astype("int64")
    if len(val) > max_users:
        val = val.sample(max_users, random_state=CFG.seed)

    model.eval()
    with torch.no_grad():
        all_anime = encode_all_anime(model, anime_tensors)
        u_idx = val["user_idx"].to_numpy()
        target = val["anime_idx"].to_numpy()
        recalls = []
        ndcgs = []
        device = all_anime.device
        affinity = torch.from_numpy(user_features.genre_affinity).to(device)
        centered = torch.from_numpy(user_features.centered_avg_score).unsqueeze(-1).to(device)
        recency = torch.from_numpy(user_features.recency).to(device)

        batch = 256
        for start in range(0, len(u_idx), batch):
            ub = u_idx[start : start + batch]
            tb = target[start : start + batch]
            packed = _build_val_history_batch(user_features, ub, None, train_cfg)
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
            ub_t = torch.from_numpy(np.asarray(ub, dtype=np.int64)).to(device)
            user_emb = model.encode_user(pooled, affinity[ub_t], centered[ub_t], recency[ub_t])
            scores = score_all_items(model, user_emb, all_anime)
            for i in range(len(ub)):
                pos = packed["histories"][i]
                if len(pos):
                    scores[i, pos] = -float("inf")
            topk = scores.topk(k, dim=1).indices.cpu().numpy()
            for i, t_idx in enumerate(tb):
                preds = topk[i]
                recalls.append(recall_at_k(preds, [int(t_idx)], k))
                ndcgs.append(ndcg_at_k(preds, [int(t_idx)], k))

    return {
        "recall@10": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


def _curriculum_hard_neg_k(epoch: int, train_cfg: TrainConfig, n_anime: int) -> int:
    """Linear schedule: K_easy at hard_neg_start_epoch -> K_hard at epochs."""
    if not train_cfg.hard_neg_curriculum:
        return -1  # sentinel: miner uses its default oversample
    if epoch < train_cfg.hard_neg_start_epoch:
        k = train_cfg.hard_neg_K_easy
    else:
        span = max(train_cfg.epochs - train_cfg.hard_neg_start_epoch, 1)
        frac = min(max((epoch - train_cfg.hard_neg_start_epoch) / span, 0.0), 1.0)
        k = train_cfg.hard_neg_K_easy + (
            train_cfg.hard_neg_K_hard - train_cfg.hard_neg_K_easy
        ) * frac
        k = int(round(k))
    # Never scan (almost) the full catalog each batch.
    return int(min(max(k, train_cfg.hard_neg_K_hard), max(n_anime // 4, train_cfg.hard_neg_K_hard)))


def train(
    model: TwoTowerModel,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feats: dict,
    user_features: UserFeaturePack,
    user_map: dict[str, int],
    anime_map: dict[int, int],
    train_cfg: TrainConfig,
    device: torch.device,
    ckpt_path: Path | None = None,
    progress: bool = True,
    max_train_rows: int | None = None,
) -> TrainArtifacts:
    rng = set_random_seed(device=device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    if max_train_rows is not None and len(train_df) > max_train_rows:
        train_df = train_df.sample(max_train_rows, random_state=CFG.seed)

    model = model.to(device)
    anime_tensors = feature_pack_to_tensors(feats, device)
    n_anime = feats["numerical"].shape[0]

    dataset = InteractionDataset(train_df, user_map, anime_map, user_features)
    collate = make_collate(
        user_features,
        n_anime,
        dataset.log_q,
        completion_floor=train_cfg.completion_floor,
    )
    sampler = make_train_sampler(dataset, seed=CFG.seed)
    loader_kwargs: dict[str, object] = {}
    if train_cfg.num_workers > 0:
        # Keep workers alive across epochs. Low prefetch: each batch is large
        # (padded history tensors); high prefetch x many workers blows RAM/IPC.
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        sampler=sampler,
        num_workers=train_cfg.num_workers,
        collate_fn=collate,
        drop_last=True,
        pin_memory=device.type == "cuda",
        **loader_kwargs,
    )

    positives_lookup: dict[int, set[int]] = {
        u: set(arr.tolist()) for u, arr in user_features.history.items()
    }

    miner = HardNegativeMiner(n_anime=n_anime, embedding_dim=train_cfg.embedding_dim)

    optim = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = -1.0
    best_epoch = -1
    ckpt_path = ckpt_path or (MODELS_DIR / "two_tower.pt")
    seq_mask_prob = train_cfg.seq_p_mask_recent if train_cfg.use_sequence_encoder else 0.0
    log_every = max(len(loader) // 20, 1)  # ~5% progress lines in batch logs (SLURM)

    for epoch in range(1, train_cfg.epochs + 1):
        do_hardneg = train_cfg.hard_neg_ratio > 0 and epoch >= train_cfg.hard_neg_start_epoch
        if do_hardneg and (
            (epoch - train_cfg.hard_neg_start_epoch) % train_cfg.hard_neg_refresh_every == 0
        ):
            if progress:
                print(f"[epoch {epoch}] refreshing hard-negative FAISS index...", flush=True)
            miner.refresh(model, anime_tensors, device)
        cur_curriculum_k = (
            _curriculum_hard_neg_k(epoch, train_cfg, n_anime) if do_hardneg else -1
        )

        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        if progress:
            print(
                f"[epoch {epoch}] training {len(loader)} batches "
                f"(hard_neg={do_hardneg}, pool_k={cur_curriculum_k})...",
                flush=True,
            )

        model.train()
        running = 0.0
        n_batches = 0
        iterator = tqdm(loader, disable=not progress, desc=f"epoch {epoch}", leave=False)
        for batch in iterator:
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            optim.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                k_cat = train_cfg.catalog_neg_count
                need_pos = do_hardneg or k_cat > 0
                user_pos: list[set[int]] | None = None
                if need_pos:
                    user_pos = [
                        positives_lookup.get(int(u), set()) | {int(p)}
                        for u, p in zip(batch["user_idx"].tolist(), batch["pos_anime_idx"].tolist())
                    ]

                encode_parts: dict[str, torch.Tensor] = {
                    "pos": batch["pos_anime_idx"],
                    "hist": batch["history_idx"],
                }
                if k_cat > 0 and user_pos is not None:
                    neg_idx = _sample_catalog_negatives(n_anime, user_pos, k_cat, rng)
                    encode_parts["cat"] = torch.from_numpy(neg_idx).to(device)

                encoded = _batch_encode_anime_by_index(model, anime_tensors, encode_parts)
                pos_anime_emb = encoded["pos"]
                pooled_hist = encode_history_batch(
                    model,
                    batch["history_idx"],
                    batch["history_mask"],
                    anime_tensors,
                    batch["history_weights"] if train_cfg.use_score_weighted_pool else None,
                    history_scores=batch["history_scores"],
                    history_completion=batch["history_completion"],
                    history_days_ago=batch["history_days_ago"],
                    train_cfg=train_cfg,
                    training_mask_prob=seq_mask_prob,
                    history_emb=encoded["hist"],
                )
                user_emb = model.encode_user(
                    pooled_hist, batch["genre_affinity"], batch["centered_score"], batch["recency"]
                )

                hard_emb = None
                if do_hardneg and user_pos is not None:
                    hard_emb = miner.mine(
                        user_emb,
                        user_pos,
                        train_cfg.hard_neg_ratio,
                        rng,
                        candidate_pool_k=cur_curriculum_k,
                    )

                catalog_neg_emb = encoded.get("cat")

                pos_weight = batch["pos_weight"] if train_cfg.use_completion_weighted_loss else None
                loss = combined_ranking_loss(
                    user_emb=user_emb,
                    pos_anime_emb=pos_anime_emb,
                    log_q=batch["pos_log_q"],
                    temperature=train_cfg.temperature,
                    catalog_neg_emb=catalog_neg_emb,
                    catalog_neg_weight=train_cfg.catalog_neg_weight,
                    extra_neg_emb=hard_emb,
                    pos_weight=pos_weight,
                )

            scaler.scale(loss).backward()
            if train_cfg.grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            scaler.step(optim)
            scaler.update()

            running += float(loss.item())
            n_batches += 1
            if progress:
                iterator.set_postfix(loss=f"{running / max(n_batches, 1):.4f}")
                if n_batches % log_every == 0 or n_batches == len(loader):
                    print(
                        f"[epoch {epoch}] batch {n_batches}/{len(loader)} "
                        f"loss={running / n_batches:.4f}",
                        flush=True,
                    )

        avg_loss = running / max(n_batches, 1)

        if epoch % train_cfg.val_every_epoch == 0:
            metrics = _val_metrics(
                model, anime_tensors, val_df, user_features, user_map, anime_map, train_cfg
            )
            score = metrics["ndcg@10"]
            if progress:
                print(
                    f"[epoch {epoch}] loss={avg_loss:.4f} val_ndcg@10={score:.4f} "
                    f"val_recall@10={metrics['recall@10']:.4f}",
                    flush=True,
                )
            if score > best_val:
                best_val = score
                best_epoch = epoch
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "dims": vars(model.dims),
                        "epoch": epoch,
                        "val_ndcg10": score,
                    },
                    ckpt_path,
                )

    return TrainArtifacts(model=model, best_val=best_val, best_epoch=best_epoch)
