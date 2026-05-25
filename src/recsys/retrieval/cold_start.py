from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from recsys.retrieval.index import AnimeIndex
from recsys.retrieval.rerank import epsilon_greedy_inject, mmr_rerank


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
    index: AnimeIndex,
    excluded_anime_ids: set[int],
    k: int,
    candidate_pool: int,
    rerank_pool: int,
    mmr_lambda: float,
    exploration_epsilon: float,
    rng: np.random.Generator,
) -> list[Recommendation]:
    user_emb = _ensure_normalized(user_emb)
    scores, idxs = index.search(user_emb[None, :], candidate_pool)
    scores, idxs = scores[0], idxs[0]

    keep_mask = np.array(
        [index.index_to_anime_id[int(i)] not in excluded_anime_ids for i in idxs],
        dtype=bool,
    )
    scores = scores[keep_mask]
    idxs = idxs[keep_mask]
    if len(idxs) == 0:
        return []

    rerank_n = min(rerank_pool, len(idxs))
    cand_emb = index.embeddings[idxs[:rerank_n]]
    chosen_local = mmr_rerank(
        candidate_idxs=idxs[:rerank_n],
        candidate_scores=scores[:rerank_n],
        candidate_emb=cand_emb,
        k=k,
        lambda_=mmr_lambda,
    )
    final_idxs = idxs[chosen_local]
    final_scores = scores[chosen_local]

    explore_pool = idxs[rerank_n:]
    final_idxs, exploratory_mask = epsilon_greedy_inject(
        final_idxs=final_idxs,
        explore_pool=explore_pool,
        k=k,
        epsilon=exploration_epsilon,
        rng=rng,
    )

    out: list[Recommendation] = []
    for rank_i, (idx, expl) in enumerate(zip(final_idxs, exploratory_mask)):
        anime_id = index.index_to_anime_id[int(idx)]
        sc = float(final_scores[rank_i]) if not expl else float("nan")
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
    index: AnimeIndex,
    genre_to_anime_idxs: dict[str, np.ndarray],
    popularity_scores: np.ndarray,
    k: int,
    candidate_pool: int,
    rerank_pool: int,
    mmr_lambda: float,
    exploration_epsilon: float,
    rng: np.random.Generator,
) -> list[Recommendation]:
    """Step 2-4 of the fallback chain: onboarding -> content sim -> genre pool -> popularity."""
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
            index=index,
            excluded_anime_ids=set(onboarding.favorite_anime_ids),
            k=k,
            candidate_pool=candidate_pool,
            rerank_pool=rerank_pool,
            mmr_lambda=mmr_lambda,
            exploration_epsilon=exploration_epsilon,
            rng=rng,
        )
        for r in recs:
            r.fallback_reason = "onboarding"
        if recs:
            return recs

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
            index=index,
            excluded_anime_ids=set(onboarding.favorite_anime_ids),
            k=k,
            candidate_pool=candidate_pool,
            rerank_pool=rerank_pool,
            mmr_lambda=mmr_lambda,
            exploration_epsilon=exploration_epsilon,
            rng=rng,
        )
        for r in recs:
            r.fallback_reason = "content_similarity"
        if recs:
            return recs

    order = np.argsort(-popularity_scores)
    excl = set(onboarding.favorite_anime_ids)
    top = []
    for idx in order:
        anime_id = index.index_to_anime_id[int(idx)]
        if anime_id in excl:
            continue
        top.append((idx, popularity_scores[idx]))
        if len(top) >= k:
            break
    return [
        Recommendation(
            anime_id=index.index_to_anime_id[int(i)],
            score=float(s),
            rank=rank + 1,
            is_exploratory=False,
            fallback_reason="popularity",
        )
        for rank, (i, s) in enumerate(top)
    ]


def build_genre_to_indices(
    genre_matrix: np.ndarray,             # [n_anime, n_genres]
    genre_vocab: list[str],
) -> dict[str, np.ndarray]:
    """Map genre name -> array of anime row indices that have that genre."""
    out: dict[str, np.ndarray] = {}
    for i, g in enumerate(genre_vocab):
        out[g] = np.flatnonzero(genre_matrix[:, i] > 0).astype(np.int64)
    return out
