"""Encode all anime via the trained two-tower and build a FAISS index.

Run:
    python -m recsys.cli.build_index
    python -m recsys.cli.build_index --no-store   # skip writing to EmbeddingStore
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch

from recsys.config import ARTIFACTS_DIR, CFG, INDEX_DIR, MODELS_DIR, set_thread_env
from recsys.data.features_anime import build_anime_features
from recsys.data.features_user import RECENCY_BUCKETS
from recsys.data.load import load_anime
from recsys.models.two_tower import (
    encode_all_anime,
    feature_pack_to_tensors,
    load_two_tower_from_checkpoint,
)
from recsys.retrieval.embedding_store import EmbeddingStore
from recsys.retrieval.index import build_index, save_index


def _sanity_check_embeddings(emb: np.ndarray, anime_ids: np.ndarray) -> None:
    """Catch silent failures in the item tower for new / cold-start anime.

    The two-tower's item tower runs purely on metadata (synopsis + genres +
    studio + numerical), so a new anime gets a real embedding the moment its
    metadata enters the catalog. This logger surfaces three classes of bugs
    that would otherwise be silent:
      * NaN / inf in the embedding (broken feature)
      * Embedding norm far from 1 (skipped F.normalize)
      * Embedding identical to the catalog mean (dead row)
    """
    n = emb.shape[0]
    norms = np.linalg.norm(emb, axis=1)
    if not np.isfinite(emb).all():
        n_bad = int(np.isnan(emb).any(axis=1).sum() + np.isinf(emb).any(axis=1).sum())
        print(f"  WARNING: {n_bad} anime have non-finite embedding components")
    if not np.allclose(norms, 1.0, atol=1e-3):
        off = int((np.abs(norms - 1.0) > 1e-3).sum())
        print(f"  WARNING: {off}/{n} anime have norm != 1 (min={norms.min():.4f} max={norms.max():.4f})")
    # Catalog mean drift: rows that look identical to the mean are likely
    # unconditional defaults (no signal). Cosine similarity threshold > 0.999.
    mu = emb.mean(axis=0, keepdims=True)
    mu = mu / (np.linalg.norm(mu) + 1e-9)
    sim = emb @ mu.T
    dead = int((sim.squeeze() > 0.999).sum())
    if dead > 0:
        print(f"  WARNING: {dead}/{n} anime are ~identical to the catalog mean (likely dead rows)")


def _diff_against_previous(store: EmbeddingStore, current_anime_ids: np.ndarray) -> None:
    """Tell the user which anime are new since the previous store version."""
    versions = store.list_versions()
    if not versions:
        return
    try:
        prev = store.load("current")
    except FileNotFoundError:
        return
    prev_ids = set(int(a) for a in prev.anime_ids.tolist())
    cur_ids = set(int(a) for a in current_anime_ids.tolist())
    new_ids = cur_ids - prev_ids
    dropped_ids = prev_ids - cur_ids
    if new_ids:
        print(f"  {len(new_ids)} new anime since previous store version (cold-start path)")
    if dropped_ids:
        print(f"  {len(dropped_ids)} anime present in previous version but missing now")


def main() -> None:
    set_thread_env()
    p = argparse.ArgumentParser()
    p.add_argument(
        "--no-store",
        action="store_true",
        help="Skip writing to the versioned EmbeddingStore (write only the legacy index dir).",
    )
    p.add_argument(
        "--version-tag",
        default=None,
        help="Tag for the new EmbeddingStore version. Defaults to a UTC timestamp.",
    )
    args = p.parse_args()
    anime_df = load_anime().sort_values("anime_id", kind="stable").reset_index(drop=True)
    feats = build_anime_features(anime_df)

    ckpt = torch.load(MODELS_DIR / "two_tower.pt", map_location="cpu", weights_only=False)
    model = load_two_tower_from_checkpoint(
        ckpt,
        recency_dim=RECENCY_BUCKETS,
        n_anime=feats["numerical"].shape[0],
    )
    model = model.to(CFG.device).eval()

    anime_tensors = feature_pack_to_tensors(feats, CFG.device)
    emb = encode_all_anime(model, anime_tensors).cpu().numpy()

    print(f"Encoded {emb.shape[0]} anime, dim={emb.shape[1]}")
    _sanity_check_embeddings(emb, feats["anime_ids"])

    idx = build_index(
        embeddings=emb,
        anime_ids=feats["anime_ids"],
        nlist=CFG.retrieval.faiss_nlist,
        nprobe=CFG.retrieval.faiss_nprobe,
    )
    save_index(idx, INDEX_DIR)

    (ARTIFACTS_DIR / "anime_index_meta.json").write_text(
        json.dumps({"n_anime": int(emb.shape[0]), "embedding_dim": int(emb.shape[1])})
    )
    print(f"Wrote FAISS index for {emb.shape[0]} anime, dim={emb.shape[1]} -> {INDEX_DIR}")

    if not args.no_store:
        store = EmbeddingStore()
        _diff_against_previous(store, feats["anime_ids"])
        version_tag = args.version_tag or datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")
        manifest_extras = {
            "model_ckpt_epoch": int(ckpt.get("epoch", -1)),
            "model_val_ndcg10": float(ckpt.get("val_ndcg10", float("nan"))),
            "embedding_dim": int(emb.shape[1]),
        }
        store.save(
            embeddings=emb,
            anime_ids=feats["anime_ids"],
            version_tag=version_tag,
            manifest_extras=manifest_extras,
        )
        print(f"Wrote EmbeddingStore version '{version_tag}' (current -> this version)")


if __name__ == "__main__":
    main()
