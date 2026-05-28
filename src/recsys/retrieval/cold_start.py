from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch

from recsys.retrieval.cascade import Cascade
from recsys.retrieval.index import AnimeIndex
from recsys.retrieval.rerank import epsilon_greedy_inject


FallbackReason = Literal["onboarding", "popularity", "content_similarity"] | None


@dataclass
class Recommendation:
    anime_id: int
    score: float
    rank: int
    is_exploratory: bool
    fallback_reason: FallbackReason


@dataclass
class OnboardingInput:
    favorite_genre_ids: list[str]
    favorite_anime_ids: list[int]


def _ensure_normalized(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return (v / n).astype(np.float32)


def recommend_for_known_user(
    user_emb: np.ndarray,                 # [D]
    cascade: Cascade,
    excluded_anime_ids: set[int],
    k: int,
    exploration_epsilon: float,
    rng: np.random.Generator,
) -> list[Recommendation]:
    """Run the four-stage cascade for a known user.

    `excluded_anime_ids` are user-facing MAL ids; we convert to dense indices
    against the FAISS index. Optional eps-greedy slot replacement is applied at
    the very end so exploration sits outside the ranker.
    """
    user_emb_t = torch.from_numpy(_ensure_normalized(user_emb)).to(cascade.device)
    excl_idx = {
        cascade.index.anime_id_to_index[a]
        for a in excluded_anime_ids
        if a in cascade.index.anime_id_to_index
    }

    final_idxs, final_scores = cascade.recommend(
        user_emb=user_emb_t,
        excluded_anime_indices=excl_idx,
        k=k * 2 if exploration_epsilon > 0 else k,
    )
    if len(final_idxs) == 0:
        return []

    chosen = final_idxs[:k]
    chosen_scores = final_scores[:k]
    explore_pool = final_idxs[k:]
    chosen, exploratory_mask = epsilon_greedy_inject(
        final_idxs=chosen,
        explore_pool=explore_pool,
        k=k,
        epsilon=exploration_epsilon,
        rng=rng,
    )

    out: list[Recommendation] = []
    for rank_i, (idx, expl) in enumerate(zip(chosen, exploratory_mask)):
        anime_id = cascade.index.index_to_anime_id[int(idx)]
        sc = float(chosen_scores[rank_i]) if rank_i < len(chosen_scores) and not expl else float("nan")
        out.append(
            Recommendation(
                anime_id=anime_id,
                score=sc,
                rank=rank_i + 1,
                is_exploratory=bool(expl),
                fallback_reason=None,
            )
        )
    return out


def recommend_from_onboarding(
    onboarding: OnboardingInput,
    cascade: Cascade,
    genre_to_anime_idxs: dict[str, np.ndarray],
    k: int,
    exploration_epsilon: float,
    rng: np.random.Generator,
) -> list[Recommendation]:
    """Cold-start: build a seed user_emb from favorites/genres and run the cascade.

    The item tower already runs purely on metadata (synopsis + genres + studio +
    numerical), so a brand-new user with only a handful of favorite anime gets
    real recommendations from the model itself -- no popularity fallback chain
    sitting outside the ranker.
    """
    index = cascade.index
    fav_idx = [
        index.anime_id_to_index[a]
        for a in onboarding.favorite_anime_ids
        if a in index.anime_id_to_index
    ]
    if fav_idx:
        seed = index.embeddings[fav_idx].mean(axis=0)
        seed = _ensure_normalized(seed)
        recs = recommend_for_known_user(
            user_emb=seed,
            cascade=cascade,
            excluded_anime_ids=set(onboarding.favorite_anime_ids),
            k=k,
            exploration_epsilon=exploration_epsilon,
            rng=rng,
        )
        for r in recs:
            r.fallback_reason = "onboarding"
        if recs:
            return recs

    # Fallback: average embedding of items in the requested genres.
    genre_pool: list[int] = []
    for g in onboarding.favorite_genre_ids:
        idxs = genre_to_anime_idxs.get(g)
        if idxs is not None:
            genre_pool.extend(int(i) for i in idxs)
    if genre_pool:
        seen: set[int] = set()
        unique = [i for i in genre_pool if not (i in seen or seen.add(i))]
        seed = index.embeddings[unique].mean(axis=0)
        seed = _ensure_normalized(seed)
        recs = recommend_for_known_user(
            user_emb=seed,
            cascade=cascade,
            excluded_anime_ids=set(onboarding.favorite_anime_ids),
            k=k,
            exploration_epsilon=exploration_epsilon,
            rng=rng,
        )
        for r in recs:
            r.fallback_reason = "content_similarity"
        return recs

    return []


def build_genre_to_indices(
    genre_matrix: np.ndarray,             # [n_anime, n_genres]
    genre_vocab: list[str],
) -> dict[str, np.ndarray]:
    """Map genre name -> array of anime row indices that have that genre."""
    out: dict[str, np.ndarray] = {}
    for i, g in enumerate(genre_vocab):
        out[g] = np.flatnonzero(genre_matrix[:, i] > 0).astype(np.int64)
    return out
