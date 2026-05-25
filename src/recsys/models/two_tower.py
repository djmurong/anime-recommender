from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
