from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from recsys.config import CACHE_DIR, CFG


GENRE_DELIM = ", "
RARE_STUDIO = "<RARE>"
UNK_STUDIO = "<UNK>"


def _split_genres(s: str) -> list[str]:
    if not s or pd.isna(s):
        return []
    return [g.strip() for g in s.split(",") if g.strip()]


def build_genre_vocab(anime_df: pd.DataFrame) -> list[str]:
    seen: set[str] = set()
    for s in anime_df["genre"].fillna("").astype(str):
        seen.update(_split_genres(s))
    return sorted(seen)


def encode_genres(anime_df: pd.DataFrame, vocab: list[str]) -> np.ndarray:
    g2i = {g: i for i, g in enumerate(vocab)}
    out = np.zeros((len(anime_df), len(vocab)), dtype=np.float32)
    for row, s in enumerate(anime_df["genre"].fillna("").astype(str)):
        for g in _split_genres(s):
            j = g2i.get(g)
            if j is not None:
                out[row, j] = 1.0
    return out


def build_studio_vocab(anime_df: pd.DataFrame, min_count: int) -> list[str]:
    counts = anime_df["studio"].fillna("").astype(str).value_counts()
    keep = counts[counts >= min_count].index.tolist()
    keep = [s for s in keep if s and s != ""]
    return [UNK_STUDIO, RARE_STUDIO] + sorted(keep)


def encode_studios(anime_df: pd.DataFrame, vocab: list[str]) -> np.ndarray:
    s2i = {s: i for i, s in enumerate(vocab)}
    out = np.empty(len(anime_df), dtype=np.int64)
    for row, s in enumerate(anime_df["studio"].fillna("").astype(str)):
        if not s:
            out[row] = s2i[UNK_STUDIO]
        else:
            out[row] = s2i.get(s, s2i[RARE_STUDIO])
    return out


def encode_numerical(anime_df: pd.DataFrame, cols: tuple[str, ...]) -> tuple[np.ndarray, StandardScaler]:
    sub = anime_df[list(cols)].copy()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.fillna(sub.median(numeric_only=True))
    scaler = StandardScaler()
    arr = scaler.fit_transform(sub.values).astype(np.float32)
    return arr, scaler


def synopsis_embeddings(
    anime_df: pd.DataFrame,
    model_name: str,
    cache_path: Path,
    text_col_priority: tuple[str, ...] = ("synopsis", "background", "title_english", "title"),
) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(anime_df):
            return cached.astype(np.float32)

    from sentence_transformers import SentenceTransformer

    texts: list[str] = []
    for _, row in anime_df.iterrows():
        chosen = ""
        for col in text_col_priority:
            if col in anime_df.columns:
                v = row.get(col)
                if isinstance(v, str) and v.strip():
                    chosen = v.strip()
                    break
        texts.append(chosen)

    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, emb)
    return emb


def build_anime_features(
    anime_df: pd.DataFrame,
    cache_dir: Path = CACHE_DIR,
) -> dict:
    """Build the full anime feature pack and cache to disk.

    Returns a dict of arrays + vocab metadata. Frame is assumed pre-sorted by `anime_id`.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    feats_path = cache_dir / "anime_features.npz"
    meta_path = cache_dir / "anime_features_meta.json"
    synopsis_path = cache_dir / "synopsis_emb.npy"

    if feats_path.exists() and meta_path.exists():
        npz = np.load(feats_path)
        meta = json.loads(meta_path.read_text())
        if int(meta["n_anime"]) == len(anime_df):
            return {
                "anime_ids": npz["anime_ids"].astype(np.int64),
                "numerical": npz["numerical"].astype(np.float32),
                "genres": npz["genres"].astype(np.float32),
                "studio_idx": npz["studio_idx"].astype(np.int64),
                "synopsis": npz["synopsis"].astype(np.float32),
                "genre_vocab": meta["genre_vocab"],
                "studio_vocab": meta["studio_vocab"],
            }

    genre_vocab = build_genre_vocab(anime_df)
    genres = encode_genres(anime_df, genre_vocab)
    studio_vocab = build_studio_vocab(anime_df, CFG.train.studio_min_count)
    studio_idx = encode_studios(anime_df, studio_vocab)
    numerical, _ = encode_numerical(anime_df, CFG.features.numerical_cols)
    synopsis = synopsis_embeddings(anime_df, CFG.features.synopsis_model, synopsis_path)

    anime_ids = anime_df["anime_id"].astype(np.int64).to_numpy()
    np.savez(
        feats_path,
        anime_ids=anime_ids,
        numerical=numerical,
        genres=genres,
        studio_idx=studio_idx,
        synopsis=synopsis,
    )
    meta_path.write_text(
        json.dumps(
            {
                "n_anime": len(anime_df),
                "genre_vocab": genre_vocab,
                "studio_vocab": studio_vocab,
                "numerical_cols": list(CFG.features.numerical_cols),
                "synopsis_model": CFG.features.synopsis_model,
                "synopsis_dim": int(synopsis.shape[1]),
            },
            indent=2,
        )
    )
    return {
        "anime_ids": anime_ids,
        "numerical": numerical,
        "genres": genres,
        "studio_idx": studio_idx,
        "synopsis": synopsis,
        "genre_vocab": genre_vocab,
        "studio_vocab": studio_vocab,
    }
