from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recsys.config import INDEX_DIR


INDEX_FILE = "anime_faiss.index"
META_FILE = "anime_index_meta.json"
EMB_FILE = "anime_emb.npy"


@dataclass
class AnimeIndex:
    embeddings: np.ndarray            # [n_anime, D] float32, L2-normalized
    anime_ids: np.ndarray             # [n_anime] int64, MAL ids in row order
    index_to_anime_id: dict[int, int]
    anime_id_to_index: dict[int, int]
    faiss_index: object

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (scores, anime_indices). query: [B, D] float32, L2-normalized."""
        scores, idxs = self.faiss_index.search(query.astype(np.float32), k)
        return scores, idxs


def build_index(
    embeddings: np.ndarray,
    anime_ids: np.ndarray,
    nlist: int = 64,
    nprobe: int = 8,
) -> AnimeIndex:
    import faiss

    n, d = embeddings.shape
    if n < nlist * 4:
        # IVF needs enough training points; fall back to flat for tiny catalogs
        index = faiss.IndexFlatIP(d)
    else:
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings.astype(np.float32))
        index.nprobe = nprobe
    index.add(embeddings.astype(np.float32))

    return AnimeIndex(
        embeddings=embeddings.astype(np.float32),
        anime_ids=anime_ids.astype(np.int64),
        index_to_anime_id={i: int(a) for i, a in enumerate(anime_ids)},
        anime_id_to_index={int(a): i for i, a in enumerate(anime_ids)},
        faiss_index=index,
    )


def save_index(idx: AnimeIndex, out_dir: Path = INDEX_DIR) -> None:
    import faiss

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx.faiss_index, str(out_dir / INDEX_FILE))
    np.save(out_dir / EMB_FILE, idx.embeddings)
    (out_dir / META_FILE).write_text(
        json.dumps({"anime_ids": idx.anime_ids.tolist()})
    )


def load_index(in_dir: Path = INDEX_DIR) -> AnimeIndex:
    import faiss

    index = faiss.read_index(str(in_dir / INDEX_FILE))
    embeddings = np.load(in_dir / EMB_FILE).astype(np.float32)
    meta = json.loads((in_dir / META_FILE).read_text())
    anime_ids = np.array(meta["anime_ids"], dtype=np.int64)
    return AnimeIndex(
        embeddings=embeddings,
        anime_ids=anime_ids,
        index_to_anime_id={i: int(a) for i, a in enumerate(anime_ids)},
        anime_id_to_index={int(a): i for i, a in enumerate(anime_ids)},
        faiss_index=index,
    )
