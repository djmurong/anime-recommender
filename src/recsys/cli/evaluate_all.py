"""Run all baselines + the two-tower model on the same leave-one-out test split.

Run:
    python -m recsys.cli.evaluate_all
    python -m recsys.cli.evaluate_all --no-cascade  # skip cascade row
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import torch

from recsys.config import (
    ARTIFACTS_DIR,
    CACHE_DIR,
    CFG,
    INDEX_DIR,
    MODELS_DIR,
    set_random_seed,
    set_thread_env,
)
from recsys.data.features_anime import build_anime_features
from recsys.data.features_user import RECENCY_BUCKETS
from recsys.data.load import load_anime
from recsys.data.split import load_splits
from recsys.eval.harness import (
    evaluate_baseline,
    evaluate_cascade,
    evaluate_two_tower,
    format_table,
    write_report,
)
from recsys.models.popularity_bias import build_popularity_bias_vector
from recsys.models.ranker import MMoEServeFn
from recsys.models.two_tower import (
    encode_all_anime,
    feature_pack_to_tensors,
    load_two_tower_from_checkpoint,
)
from recsys.retrieval.cascade import make_default_cascade
from recsys.retrieval.embedding_store import EmbeddingStore
from recsys.retrieval.index import build_index, load_index
from recsys.training.ranker_trainer import load_ranker


def _popularity_rank(feats: dict) -> np.ndarray:
    """Rank index 0 = most popular. Falls back to anime row order if numerical missing."""
    n = feats["numerical"].shape[0]
    pop_col = None
    cols = list(getattr(CFG.features, "numerical_cols", []))
    if "popularity" in cols:
        pop_col = feats["numerical"][:, cols.index("popularity")]
    if pop_col is None:
        pop_col = -np.arange(n, dtype=np.float32)  # arbitrary stable order
    order = np.argsort(pop_col)  # smaller "popularity" number = more popular on MAL
    rank = np.empty_like(order)
    rank[order] = np.arange(n)
    return rank


def _load_or_build_index(model, feats, anime_tensors):
    """Use the EmbeddingStore -> legacy FAISS index -> in-memory build, in order."""
    # 1) Prefer the versioned EmbeddingStore. The cascade only cares about the
    #    numeric embeddings; rebuilding FAISS over them is cheap relative to a
    #    full re-encode of the catalog.
    try:
        store = EmbeddingStore()
        cur = store.load("current")
        if cur.embeddings.shape[0] == feats["numerical"].shape[0]:
            print(f"  using EmbeddingStore current ({cur.manifest.get('version_tag', '?')})")
            return build_index(
                embeddings=cur.embeddings,
                anime_ids=cur.anime_ids,
                nlist=CFG.retrieval.faiss_nlist,
                nprobe=CFG.retrieval.faiss_nprobe,
            )
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"  EmbeddingStore unavailable ({e}), falling back to legacy index")

    # 2) Legacy persisted FAISS index.
    try:
        idx = load_index()
        if idx.embeddings.shape[0] == feats["numerical"].shape[0]:
            return idx
    except Exception:
        pass

    # 3) Final fallback: re-encode and build in-memory.
    print("  building FAISS index in-memory...")
    emb = encode_all_anime(model, anime_tensors).cpu().numpy()
    return build_index(
        embeddings=emb,
        anime_ids=feats["anime_ids"],
        nlist=CFG.retrieval.faiss_nlist,
        nprobe=CFG.retrieval.faiss_nprobe,
    )


def main() -> None:
    set_thread_env()
    set_random_seed()
    p = argparse.ArgumentParser()
    p.add_argument(
        "--no-cascade",
        action="store_true",
        help="Skip the cascade eval row (only run baselines + brute-force two-tower).",
    )
    p.add_argument(
        "--cascade-name",
        default="TwoTower+Cascade+Seq+CompWeighted",
        help="Label for the cascade row in eval.md (Phase tag).",
    )
    p.add_argument(
        "--no-ranker",
        action="store_true",
        help="Force the Phase-1 cascade (ignore artifacts/models/ranker.pt even if present).",
    )
    p.add_argument(
        "--mmr-lambda",
        type=float,
        default=None,
        help="Override RetrievalConfig.mmr_lambda for this eval (relevance vs diversity).",
    )
    p.add_argument(
        "--pool-retrieve",
        type=int,
        default=None,
        help="Override RetrievalConfig.pool_retrieve (FAISS candidate count).",
    )
    p.add_argument(
        "--pool-rank",
        type=int,
        default=None,
        help="Override RetrievalConfig.pool_rank (survivors into the reranker).",
    )
    p.add_argument(
        "--rank-blend",
        type=float,
        default=None,
        help="Override RetrievalConfig.rank_blend: 1.0=pure retrieval, 0.0=pure ranker.",
    )
    args = p.parse_args()

    train, val, test = load_splits()
    user_map = json.loads((ARTIFACTS_DIR / "user_map.json").read_text())
    anime_map = {int(k): v for k, v in json.loads((ARTIFACTS_DIR / "anime_map.json").read_text()).items()}
    anime_df = load_anime().sort_values("anime_id", kind="stable").reset_index(drop=True)
    feats = build_anime_features(anime_df)
    with open(CACHE_DIR / "user_features.pkl", "rb") as f:
        user_features = pickle.load(f)

    item_features_for_ild = np.concatenate([feats["genres"], feats["numerical"]], axis=1).astype(np.float32)
    pop_rank = _popularity_rank(feats)

    with open(MODELS_DIR / "baselines.pkl", "rb") as f:
        bls = pickle.load(f)

    results = []
    for key in ("popularity", "content", "mf"):
        rec = bls[key]
        print(f"Evaluating {rec.name}...")
        results.append(
            evaluate_baseline(
                rec=rec,
                test_df=test,
                user_features=user_features,
                user_map=user_map,
                anime_map=anime_map,
                item_features_for_ild=item_features_for_ild,
                popularity_rank=pop_rank,
            )
        )

    ckpt_path = MODELS_DIR / "two_tower.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        pop_bias = build_popularity_bias_vector(anime_df, anime_map)
        model = load_two_tower_from_checkpoint(
            ckpt,
            recency_dim=RECENCY_BUCKETS,
            n_anime=feats["numerical"].shape[0],
            popularity_bias=pop_bias,
        )
        print("Evaluating TwoTower (brute-force full-catalog)...")
        results.append(
            evaluate_two_tower(
                model=model,
                feats=feats,
                test_df=test,
                user_features=user_features,
                user_map=user_map,
                anime_map=anime_map,
                item_features_for_ild=item_features_for_ild,
                popularity_rank=pop_rank,
            )
        )

        if not args.no_cascade:
            device = CFG.device
            model_dev = model.to(device).eval()
            anime_tensors = feature_pack_to_tensors(feats, device)
            print("Evaluating TwoTower through the cascade...")
            index = _load_or_build_index(model_dev, feats, anime_tensors)
            item_emb_tensor = torch.from_numpy(index.embeddings).to(device)
            mmr_lambda = (
                args.mmr_lambda if args.mmr_lambda is not None else CFG.retrieval.mmr_lambda
            )
            pool_retrieve = (
                args.pool_retrieve
                if args.pool_retrieve is not None
                else CFG.retrieval.pool_retrieve
            )
            pool_rank = (
                args.pool_rank if args.pool_rank is not None else CFG.retrieval.pool_rank
            )
            rank_blend = (
                args.rank_blend if args.rank_blend is not None else CFG.retrieval.rank_blend
            )

            ranker_path = MODELS_DIR / "ranker.pt"
            ranker_fn = None
            ranker_loaded = False
            if ranker_path.exists() and not args.no_ranker and rank_blend < 1.0:
                mmoe, _ = load_ranker(ranker_path, map_location=device)
                mmoe = mmoe.to(device).eval()
                ranker_fn = MMoEServeFn(
                    model=mmoe,
                    w_completion=CFG.retrieval.mmoe_w_completion,
                    w_rating=CFG.retrieval.mmoe_w_rating,
                    w_drop=CFG.retrieval.mmoe_w_drop,
                )
                ranker_loaded = True
                cascade_name = args.cascade_name + "+MMoE"
            else:
                cascade_name = args.cascade_name
            cascade = make_default_cascade(
                index=index,
                item_emb_tensor=item_emb_tensor,
                device=device,
                preranker=None,
                ranker=ranker_fn,
                reranker_kind=CFG.retrieval.reranker,
                mmr_lambda=mmr_lambda,
                dpp_theta=CFG.retrieval.dpp_theta,
                pool_retrieve=pool_retrieve,
                pool_prerank=CFG.retrieval.pool_prerank,
                pool_rank=pool_rank,
                rank_blend=rank_blend,
            )
            print(
                f"  cascade config: pool_retrieve={pool_retrieve} pool_rank={pool_rank} "
                f"mmr_lambda={mmr_lambda} rank_blend={rank_blend} "
                f"ranker={'on' if ranker_loaded else 'off'}"
            )
            if ranker_loaded:
                print(f"  using MMoE ranker from {ranker_path} (blend={rank_blend})")
            if CFG.retrieval.reranker == "dpp":
                cascade_name = cascade_name + "+DPP"
            results.append(
                evaluate_cascade(
                    name=cascade_name,
                    model=model_dev,
                    cascade=cascade,
                    feats=feats,
                    test_df=test,
                    user_features=user_features,
                    user_map=user_map,
                    anime_map=anime_map,
                    item_features_for_ild=item_features_for_ild,
                    popularity_rank=pop_rank,
                )
            )
    else:
        print(f"No two-tower checkpoint at {ckpt_path}; skipping. Run cli.train_two_tower first.")

    table = format_table(results)
    print()
    print(table)
    out = write_report(results)
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
