from __future__ import annotations

import numpy as np
import torch

from recsys.models.two_tower import TwoTowerModel, encode_all_anime


class HardNegativeMiner:
    """FAISS-backed hard negative miner.

    Refresh the index after each scheduled epoch with the current anime tower outputs;
    `mine` returns top-k anime indices by inner-product similarity to a batch of user
    embeddings, filtered against per-user known positives.
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
    ) -> torch.Tensor:
        """Returns [B, k, D] tensor of hard negative anime embeddings (or empty if disabled)."""
        if self._index is None or self._anime_emb_cpu is None or k <= 0:
            return user_emb.new_zeros((user_emb.size(0), 0, user_emb.size(1)))

        q = user_emb.detach().cpu().numpy().astype(np.float32)
        # Pull more candidates than needed so we can drop user positives without underfilling.
        oversample = max(k * 4, 32)
        _scores, idxs = self._index.search(q, oversample)
        chosen = np.empty((q.shape[0], k), dtype=np.int64)
        for i, row in enumerate(idxs):
            pos = positives_per_user[i]
            cands = [int(a) for a in row if int(a) not in pos]
            if len(cands) >= k:
                chosen[i] = cands[:k]
            else:
                deficit = k - len(cands)
                rand = rng.integers(0, self.n_anime, size=deficit * 4)
                rand = [int(r) for r in rand if int(r) not in pos][:deficit]
                while len(rand) < deficit:
                    rand.append(int(rng.integers(0, self.n_anime)))
                chosen[i] = np.array(cands + rand[:deficit], dtype=np.int64)

        return torch.from_numpy(self._anime_emb_cpu[chosen]).to(user_emb.device)
