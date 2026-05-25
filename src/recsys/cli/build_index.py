"""Encode all anime via the trained two-tower and build a FAISS index.

Run:
    python -m recsys.cli.build_index
"""
from __future__ import annotations

import json

import torch

from recsys.config import ARTIFACTS_DIR, CFG, INDEX_DIR, MODELS_DIR, set_thread_env
from recsys.data.features_anime import build_anime_features
from recsys.data.features_user import RECENCY_BUCKETS
from recsys.data.load import load_anime
from recsys.models.two_tower import encode_all_anime, feature_pack_to_tensors, load_two_tower_from_checkpoint
from recsys.retrieval.index import build_index, save_index


def main() -> None:
    set_thread_env()
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


if __name__ == "__main__":
    main()
