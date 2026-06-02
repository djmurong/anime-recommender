from __future__ import annotations

import numpy as np
import torch

from recsys.models.two_tower import TwoTowerModel, encode_all_anime


def _filter_and_sample(
    row: np.ndarray,
    pos: set[int],
    k: int,
    rng: np.random.Generator,
    n_anime: int,
) -> np.ndarray:
    """Drop positives from a FAISS neighbor row and sample k indices."""
    row = row[row >= 0]
    if pos:
        pos_arr = np.fromiter(pos, count=len(pos), dtype=np.int64)
        row = row[~np.isin(row, pos_arr, assume_unique=False)]
    if len(row) >= k:
        return rng.choice(row, size=k, replace=False)
    chosen = row.astype(np.int64, copy=True)
    deficit = k - len(chosen)
    while len(chosen) < k:
        extra = rng.integers(0, n_anime, size=max(deficit * 4, 64), dtype=np.int64)
        if pos:
            extra = extra[~np.isin(extra, pos_arr, assume_unique=False)]
        for c in extra:
            if int(c) not in pos and (len(chosen) == 0 or c not in chosen):
                chosen = np.append(chosen, c)
                if len(chosen) >= k:
                    break
    return chosen[:k]


class HardNegativeMiner:
    """FAISS-backed hard negative miner with optional curriculum.

    `refresh` rebuilds the FAISS index from the current anime tower outputs.
    `mine` returns the top-`candidate_pool_k` anime by inner product with each
    user embedding, then randomly samples k of them per row (after removing
    known positives). Sampling within a wider pool gives "easy" negatives;
    sampling within a narrow pool gives "hard" near-miss negatives. The trainer
    schedules `candidate_pool_k` to shrink linearly across epochs, which is the
    PinSage curriculum-learning recipe.
    """

    def __init__(self, n_anime: int, embedding_dim: int):
        self.n_anime = n_anime
        self.embedding_dim = embedding_dim
        self._index = None
        self._anime_emb_cpu: np.ndarray | None = None

    def refresh(self, model: TwoTowerModel, anime_tensors: dict, device: torch.device) -> None:
        import faiss

        emb = encode_all_anime(model, anime_tensors).detach().cpu().numpy().astype(np.float32)
        self._anime_emb_cpu = emb
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(emb)
        self._index = index

    def mine(
        self,
        user_emb: torch.Tensor,           # [B, D]
        positives_per_user: list[set[int]],
        k: int,
        rng: np.random.Generator,
        candidate_pool_k: int = -1,
    ) -> torch.Tensor:
        """Returns [B, k, D] tensor of hard negative anime embeddings.

        Args:
            user_emb: query embeddings.
            positives_per_user: per-row set of indices to filter out.
            k: number of hard negatives per row.
            rng: random generator for in-pool sampling.
            candidate_pool_k: size of the candidate pool to sample from.
                If <= 0, falls back to the legacy behavior (take the top-k
                hits directly with a small oversample).
        """
        if self._index is None or self._anime_emb_cpu is None or k <= 0:
            return user_emb.new_zeros((user_emb.size(0), 0, user_emb.size(1)))

        q = user_emb.detach().cpu().numpy().astype(np.float32)

        if candidate_pool_k > 0:
            pool = min(candidate_pool_k, self.n_anime)
            _scores, idxs = self._index.search(q, pool)
            chosen = np.empty((q.shape[0], k), dtype=np.int64)
            for i, row in enumerate(idxs):
                chosen[i] = _filter_and_sample(row, positives_per_user[i], k, rng, self.n_anime)
            return torch.from_numpy(self._anime_emb_cpu[chosen]).to(user_emb.device)

        # Legacy path: pull a small oversample and take the top-k.
        oversample = max(k * 4, 32)
        _scores, idxs = self._index.search(q, oversample)
        chosen = np.empty((q.shape[0], k), dtype=np.int64)
        for i, row in enumerate(idxs):
            pos = positives_per_user[i]
            cands = [int(a) for a in row if int(a) not in pos]
            if len(cands) >= k:
                chosen[i] = np.asarray(cands[:k], dtype=np.int64)
            else:
                chosen[i] = _filter_and_sample(row, pos, k, rng, self.n_anime)

        return torch.from_numpy(self._anime_emb_cpu[chosen]).to(user_emb.device)
