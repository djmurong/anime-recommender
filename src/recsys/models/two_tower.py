from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recsys.models.sequence import SequenceEncoder, random_time_window_mask


@dataclass
class TowerDims:
    n_studios: int
    studio_emb_dim: int
    n_genres: int
    n_numerical: int
    synopsis_dim: int
    embedding_dim: int
    hidden_dim: int
    dropout: float
    n_anime: int = 0
    use_sequence_encoder: bool = True
    seq_n_layers: int = 4
    seq_n_heads: int = 4
    seq_ffn_mult: int = 2
    seq_max_history: int = 128


class AnimeEncoder(nn.Module):
    """Maps raw anime features -> L2-normalized embedding."""

    def __init__(self, dims: TowerDims):
        super().__init__()
        self.studio_emb = nn.Embedding(dims.n_studios, dims.studio_emb_dim)
        in_dim = dims.n_numerical + dims.n_genres + dims.studio_emb_dim + dims.synopsis_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, dims.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dims.dropout),
            nn.Linear(dims.hidden_dim, dims.embedding_dim),
        )
        self.out_dim = dims.embedding_dim

    def forward(
        self,
        numerical: torch.Tensor,
        genres: torch.Tensor,
        studio_idx: torch.Tensor,
        synopsis: torch.Tensor,
    ) -> torch.Tensor:
        s = self.studio_emb(studio_idx)
        x = torch.cat([numerical, genres, s, synopsis], dim=-1)
        e = self.net(x)
        return F.normalize(e, dim=-1)


class UserTower(nn.Module):
    """Maps user features (pooled history + side features) -> L2-normalized embedding."""

    def __init__(self, dims: TowerDims, n_genres: int, recency_dim: int):
        super().__init__()
        in_dim = dims.embedding_dim + n_genres + 1 + recency_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, dims.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dims.dropout),
            nn.Linear(dims.hidden_dim, dims.embedding_dim),
        )

    def forward(
        self,
        pooled_history: torch.Tensor,
        genre_affinity: torch.Tensor,
        centered_score: torch.Tensor,
        recency: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([pooled_history, genre_affinity, centered_score, recency], dim=-1)
        e = self.net(x)
        return F.normalize(e, dim=-1)


class TwoTowerModel(nn.Module):
    """Bundles the shared AnimeEncoder + UserTower and exposes scoring helpers."""

    def __init__(
        self,
        dims: TowerDims,
        recency_dim: int,
        popularity_bias: np.ndarray | None = None,
    ):
        super().__init__()
        self.anime_encoder = AnimeEncoder(dims)
        self.user_tower = UserTower(dims, dims.n_genres, recency_dim)
        self.dims = dims
        n_anime = dims.n_anime
        self.item_bias = nn.Embedding(n_anime, 1)
        if popularity_bias is not None and len(popularity_bias) == n_anime:
            with torch.no_grad():
                self.item_bias.weight.copy_(
                    torch.from_numpy(popularity_bias.astype(np.float32)).view(-1, 1)
                )
        else:
            nn.init.zeros_(self.item_bias.weight)

        self.sequence_encoder: SequenceEncoder | None = None
        if dims.use_sequence_encoder:
            self.sequence_encoder = SequenceEncoder(
                item_emb_dim=dims.embedding_dim,
                d_model=dims.embedding_dim,
                n_layers=dims.seq_n_layers,
                n_heads=dims.seq_n_heads,
                ffn_mult=dims.seq_ffn_mult,
                dropout=dims.dropout,
                max_history=dims.seq_max_history,
            )

    def encode_anime(
        self,
        numerical: torch.Tensor,
        genres: torch.Tensor,
        studio_idx: torch.Tensor,
        synopsis: torch.Tensor,
    ) -> torch.Tensor:
        return self.anime_encoder(numerical, genres, studio_idx, synopsis)

    def pool_history(
        self,
        history_emb: torch.Tensor,
        history_mask: torch.Tensor,
        history_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history_weights is None:
            denom = history_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            return (history_emb * history_mask.unsqueeze(-1)).sum(dim=1) / denom
        w = history_weights * history_mask
        denom = w.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (history_emb * w.unsqueeze(-1)).sum(dim=1) / denom

    def encode_history_sequence(
        self,
        history_emb: torch.Tensor,
        history_mask: torch.Tensor,
        history_scores: torch.Tensor,
        history_completion: torch.Tensor,
        days_ago: torch.Tensor,
        *,
        training_mask_prob: float = 0.0,
        mask_window: tuple[int, int] = (5, 30),
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Returns (B, embedding_dim) history vector via the Transformer encoder.

        Falls back to mean pool if the model was built without a sequence encoder
        (e.g. ablation runs or backward-compat checkpoints).
        """
        if self.sequence_encoder is None:
            return self.pool_history(history_emb, history_mask)

        if training_mask_prob > 0.0 and self.training:
            history_mask = random_time_window_mask(
                history_mask,
                days_ago,
                p_mask=training_mask_prob,
                window_min=mask_window[0],
                window_max=mask_window[1],
                generator=generator,
            )
        return self.sequence_encoder(
            item_emb=history_emb,
            history_mask=history_mask,
            history_scores=history_scores,
            history_completion=history_completion,
            days_ago=days_ago,
        )

    def encode_user(
        self,
        pooled_history: torch.Tensor,
        genre_affinity: torch.Tensor,
        centered_score: torch.Tensor,
        recency: torch.Tensor,
    ) -> torch.Tensor:
        return self.user_tower(pooled_history, genre_affinity, centered_score, recency)

    def score_candidates(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        item_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dot-product scores plus item popularity bias."""
        if item_emb.dim() == 2 and user_emb.dim() == 2 and item_emb.size(0) != user_emb.size(0):
            logits = user_emb @ item_emb.t()
            bias = self.item_bias.weight.squeeze(-1)
            return logits + bias.unsqueeze(0)
        if item_indices is None:
            raise ValueError("item_indices required when item_emb is not a full catalog matrix")
        logits = (user_emb * item_emb).sum(dim=-1)
        return logits + self.item_bias(item_indices).squeeze(-1)


def load_two_tower_from_checkpoint(
    ckpt: dict[str, Any],
    recency_dim: int,
    n_anime: int,
    popularity_bias: np.ndarray | None = None,
) -> TwoTowerModel:
    """Load model from checkpoint dict with backward compatibility for older checkpoints."""
    dims_raw = dict(ckpt["dims"])
    if "n_anime" not in dims_raw or not dims_raw["n_anime"]:
        dims_raw["n_anime"] = n_anime
    # Older checkpoints don't carry the sequence-encoder fields; default to the
    # old mean-pool behavior so they can still be loaded for comparison.
    dims_raw.setdefault("use_sequence_encoder", False)
    dims_raw.setdefault("seq_n_layers", 4)
    dims_raw.setdefault("seq_n_heads", 4)
    dims_raw.setdefault("seq_ffn_mult", 2)
    dims_raw.setdefault("seq_max_history", 128)
    dims = TowerDims(**dims_raw)
    model = TwoTowerModel(dims, recency_dim=recency_dim, popularity_bias=popularity_bias)
    model.load_state_dict(ckpt["model_state"], strict=False)
    return model


def feature_pack_to_tensors(
    feats: dict,
    device: torch.device,
) -> dict:
    """Move the cached numpy feature pack onto a device as torch tensors (read-only)."""
    return {
        "numerical": torch.from_numpy(feats["numerical"]).to(device),
        "genres": torch.from_numpy(feats["genres"]).to(device),
        "studio_idx": torch.from_numpy(feats["studio_idx"].astype(np.int64)).to(device),
        "synopsis": torch.from_numpy(feats["synopsis"]).to(device),
    }


@torch.no_grad()
def encode_all_anime(model: TwoTowerModel, anime_tensors: dict, batch_size: int = 4096) -> torch.Tensor:
    model.eval()
    n = anime_tensors["numerical"].shape[0]
    out = []
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        emb = model.encode_anime(
            anime_tensors["numerical"][sl],
            anime_tensors["genres"][sl],
            anime_tensors["studio_idx"][sl],
            anime_tensors["synopsis"][sl],
        )
        out.append(emb)
    return torch.cat(out, dim=0)


def score_all_items(
    model: TwoTowerModel,
    user_emb: torch.Tensor,
    all_anime_emb: torch.Tensor,
) -> torch.Tensor:
    """Full-catalog scores: cosine similarity + per-item bias."""
    return model.score_candidates(user_emb, all_anime_emb)


def load_two_tower_checkpoint(
    path: Path,
    recency_dim: int,
    n_anime: int,
    popularity_bias: np.ndarray | None = None,
    map_location: str | torch.device = "cpu",
) -> TwoTowerModel:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    return load_two_tower_from_checkpoint(ckpt, recency_dim, n_anime, popularity_bias)
