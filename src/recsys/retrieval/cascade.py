"""Four-stage cascade pipeline: Retrieve -> PreRank -> Rank -> ReRank.

Stage breakdown:
  * **Retrieve** uses FAISS over the two-tower item embeddings to fetch
    `pool_retrieve` candidates per user. Cheap and approximate.
  * **PreRank** runs a small cross-encoder (`PreRanker`) over the retrieved
    pool to cut to `pool_prerank`. In Phase 1 this is trained by distilling
    the two-tower dot product; in Phase 2 it distills the MMoE ranker.
  * **Rank** runs the heavy multi-task ranker (`MMoERanker` once Phase 2 is
    in) to cut to `pool_rank`. Phase 1 falls back to the PreRanker score.
  * **ReRank** applies MMR (Phase 1/2) or DPP (Phase 3) for diversity.

The cascade is the single entrypoint used by both online serving and offline
evaluation, so eval reflects what production would actually surface instead of
the brute-force full-catalog topk that masked the popularity collapse.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import numpy as np
import torch

from recsys.models.preranker import PreRanker
from recsys.retrieval.index import AnimeIndex
from recsys.retrieval.rerank import mmr_rerank


class RankerProtocol(Protocol):
    """Anything that scores (user_emb, item_emb) -> per-item scalar.

    Both `PreRanker` and the future `MMoERanker.serve_score` satisfy this.
    """

    def __call__(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor: ...


class RerankerProtocol(Protocol):
    """Per-user reranker that returns selected indices into the candidate pool."""

    def __call__(
        self,
        candidate_idxs: np.ndarray,
        candidate_scores: np.ndarray,
        candidate_emb: np.ndarray,
        k: int,
    ) -> np.ndarray: ...


def _zscore(x: np.ndarray) -> np.ndarray:
    """Per-pool standardization so two score scales can be blended additively.

    Returns zeros when the pool has near-zero variance (e.g. a single survivor),
    which makes the blend degrade gracefully to "no contribution from this term".
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.astype(np.float64)
    sd = x.std()
    if sd < 1e-8:
        return np.zeros_like(x)
    return (x - x.mean()) / sd


def _mmr_reranker_factory(lambda_: float) -> RerankerProtocol:
    def _fn(candidate_idxs, candidate_scores, candidate_emb, k):
        return mmr_rerank(
            candidate_idxs=candidate_idxs,
            candidate_scores=candidate_scores,
            candidate_emb=candidate_emb,
            k=k,
            lambda_=lambda_,
        )

    return _fn


@dataclass
class CascadeStageScores:
    """Per-stage diagnostic scores for one user. Useful for debugging / ablations."""

    retrieve_idx: np.ndarray
    retrieve_score: np.ndarray
    prerank_idx: np.ndarray
    prerank_score: np.ndarray
    rank_idx: np.ndarray
    rank_score: np.ndarray
    final_idx: np.ndarray
    final_score: np.ndarray


@dataclass
class Cascade:
    """Bundle of stages with sensible defaults.

    All score-producing components are optional so individual phases can be
    swapped without rewriting the call sites: e.g. Phase 1 has `ranker=None`
    (PreRanker score is used as the ranker score), and Phase 3 swaps
    `reranker` from MMR to DPP.
    """

    index: AnimeIndex
    item_emb_tensor: torch.Tensor                    # (n_items, D) on `device`
    device: torch.device
    preranker: PreRanker | None = None
    ranker: RankerProtocol | None = None
    reranker: RerankerProtocol | None = None
    pool_retrieve: int = 1000
    pool_prerank: int = 200
    pool_rank: int = 50
    # Stage-3 score blend: rank_scores = blend * z(retrieval) + (1 - blend) * z(ranker).
    # 1.0 = pure retrieval ordering (ranker ignored, reproduces the no-ranker
    # cascade); 0.0 = pure ranker. Values in between keep the retriever's
    # relevance signal while letting the ranker re-order, which avoids the recall
    # collapse from letting a multi-task ranker fully replace retrieval.
    rank_blend: float = 1.0

    def recommend(
        self,
        user_emb: torch.Tensor,                    # (D,) or (1, D)
        excluded_anime_indices: set[int] | np.ndarray | None,
        k: int,
        return_stage_scores: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, CascadeStageScores]:
        """Run the cascade for one user.

        Returns `(final_anime_indices[k], final_scores[k])`. When
        `return_stage_scores=True`, also returns a `CascadeStageScores` with the
        survivors of each stage for diagnostics.
        """
        if user_emb.dim() == 1:
            user_emb = user_emb.unsqueeze(0)

        # ---- Stage 1: FAISS retrieve ----
        u_np = user_emb.detach().cpu().numpy().astype(np.float32)
        retrieve_k = min(self.pool_retrieve, self.item_emb_tensor.shape[0])
        scores_r, idxs_r = self.index.search(u_np, retrieve_k)
        scores_r, idxs_r = scores_r[0], idxs_r[0]
        valid = idxs_r >= 0
        scores_r = scores_r[valid]
        idxs_r = idxs_r[valid]

        if excluded_anime_indices is not None:
            excl_set = (
                excluded_anime_indices
                if isinstance(excluded_anime_indices, set)
                else set(int(x) for x in np.asarray(excluded_anime_indices).tolist())
            )
            keep = np.array([int(i) not in excl_set for i in idxs_r], dtype=bool)
            scores_r = scores_r[keep]
            idxs_r = idxs_r[keep]
        if len(idxs_r) == 0:
            empty = np.zeros(0, dtype=np.int64)
            empty_f = np.zeros(0, dtype=np.float32)
            if return_stage_scores:
                return empty, empty_f, CascadeStageScores(
                    empty, empty_f, empty, empty_f, empty, empty_f, empty, empty_f
                )
            return empty, empty_f

        # ---- Stage 2: PreRank (or pass-through) ----
        if self.preranker is not None:
            item_emb_pool = self.item_emb_tensor[idxs_r].unsqueeze(0)  # (1, K, D)
            with torch.no_grad():
                pre_scores = (
                    self.preranker(user_emb, item_emb_pool).squeeze(0).detach().cpu().numpy()
                )
        else:
            pre_scores = scores_r.copy()

        prerank_n = min(self.pool_prerank, len(pre_scores))
        # take top-prerank_n by prerank score
        if prerank_n < len(pre_scores):
            top_local = np.argpartition(-pre_scores, prerank_n - 1)[:prerank_n]
            top_local = top_local[np.argsort(-pre_scores[top_local])]
        else:
            top_local = np.argsort(-pre_scores)
        idxs_pre = idxs_r[top_local]
        scores_pre = pre_scores[top_local]

        # ---- Stage 3: Rank ----
        # rank_blend >= 1.0 means "ignore the ranker", so we skip the forward
        # pass entirely and keep the retrieval ordering. Otherwise we blend the
        # standardized retrieval and ranker scores so the ranker re-orders within
        # the retriever's relevance signal instead of overriding it.
        blend = float(self.rank_blend)
        if self.ranker is not None and blend < 1.0:
            item_emb_pre = self.item_emb_tensor[idxs_pre].unsqueeze(0)
            with torch.no_grad():
                rank_raw = (
                    self.ranker(user_emb, item_emb_pre).squeeze(0).detach().cpu().numpy()
                )
            rank_raw = np.asarray(rank_raw, dtype=np.float64).reshape(-1)
            if blend <= 0.0:
                rank_scores = rank_raw
            else:
                rank_scores = blend * _zscore(scores_pre) + (1.0 - blend) * _zscore(rank_raw)
        else:
            rank_scores = scores_pre.copy()

        rank_n = min(self.pool_rank, len(rank_scores))
        if rank_n < len(rank_scores):
            top_local = np.argpartition(-rank_scores, rank_n - 1)[:rank_n]
            top_local = top_local[np.argsort(-rank_scores[top_local])]
        else:
            top_local = np.argsort(-rank_scores)
        idxs_rank = idxs_pre[top_local]
        scores_rank = rank_scores[top_local]

        # ---- Stage 4: ReRank ----
        if self.reranker is not None:
            cand_emb = self.item_emb_tensor[idxs_rank].detach().cpu().numpy()
            selected_local = self.reranker(idxs_rank, scores_rank, cand_emb, k)
        else:
            selected_local = np.arange(min(k, len(idxs_rank)))
        final_idxs = idxs_rank[selected_local]
        final_scores = scores_rank[selected_local]

        if return_stage_scores:
            stage = CascadeStageScores(
                retrieve_idx=idxs_r,
                retrieve_score=scores_r,
                prerank_idx=idxs_pre,
                prerank_score=scores_pre,
                rank_idx=idxs_rank,
                rank_score=scores_rank,
                final_idx=final_idxs,
                final_score=final_scores,
            )
            return final_idxs, final_scores, stage
        return final_idxs, final_scores


def make_default_cascade(
    index: AnimeIndex,
    item_emb_tensor: torch.Tensor,
    device: torch.device,
    *,
    preranker: PreRanker | None = None,
    ranker: RankerProtocol | None = None,
    reranker_kind: str = "mmr",
    mmr_lambda: float = 0.7,
    dpp_theta: float = 0.5,
    pool_retrieve: int = 1000,
    pool_prerank: int = 200,
    pool_rank: int = 50,
    rank_blend: float = 1.0,
) -> Cascade:
    """Convenience constructor matching the defaults in `RetrievalConfig`."""
    if reranker_kind == "mmr":
        reranker = _mmr_reranker_factory(mmr_lambda)
    elif reranker_kind == "dpp":
        from recsys.retrieval.dpp import make_dpp_reranker

        reranker = make_dpp_reranker(theta=dpp_theta)
    else:
        reranker = None
    return Cascade(
        index=index,
        item_emb_tensor=item_emb_tensor,
        device=device,
        preranker=preranker,
        ranker=ranker,
        reranker=reranker,
        pool_retrieve=pool_retrieve,
        pool_prerank=pool_prerank,
        pool_rank=pool_rank,
        rank_blend=rank_blend,
    )
