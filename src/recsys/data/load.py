from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from recsys.config import ANIME_CSV, RATINGS_CSV, USERS_CSV


ANIME_DTYPES = {
    "anime_id": "int32",
    "title": "string",
    "title_english": "string",
    "title_japanese": "string",
    "type": "category",
    "source": "category",
    "episodes": "Int32",
    "status": "category",
    "duration": "string",
    "rating": "category",
    "score": "float32",
    "scored_by": "Int32",
    "rank": "Int32",
    "popularity": "Int32",
    "members": "Int32",
    "favorites": "Int32",
    "studio": "string",
    "producer": "string",
    "licensor": "string",
    "genre": "string",
    "premiered": "string",
    "duration_min": "float32",
    "aired_from_year": "Int32",
}


RATINGS_USECOLS = [
    "username",
    "anime_id",
    "my_score",
    "my_status",
    "my_watched_episodes",
    "my_last_updated",
]


RATINGS_DTYPES = {
    "username": "string",
    "anime_id": "int32",
    "my_score": "int8",
    "my_status": "int8",
    "my_watched_episodes": "Int32",
}


def load_anime(path: Path | None = None) -> pd.DataFrame:
    p = path or ANIME_CSV
    df = pd.read_csv(p, engine="pyarrow")
    for col, dt in ANIME_DTYPES.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dt)
            except (ValueError, TypeError):
                pass
    df["genre"] = df["genre"].fillna("").astype("string")
    df["studio"] = df["studio"].fillna("").astype("string")
    return df


def load_users(path: Path | None = None, usecols: list[str] | None = None) -> pd.DataFrame:
    p = path or USERS_CSV
    return pd.read_csv(p, engine="pyarrow", usecols=usecols)


def iter_ratings_chunks(
    path: Path | None = None,
    chunksize: int = 2_000_000,
    usecols: list[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Stream the large ratings CSV in chunks. Falls back to C engine for chunked reads."""
    p = path or RATINGS_CSV
    cols = usecols or RATINGS_USECOLS
    return pd.read_csv(
        p,
        usecols=cols,
        chunksize=chunksize,
        low_memory=False,
        parse_dates=["my_last_updated"] if "my_last_updated" in cols else None,
    )


def load_ratings(path: Path | None = None) -> pd.DataFrame:
    """Load full ratings CSV. Use with care; ~36M rows.

    Prefer streaming via iter_ratings_chunks + filter for memory safety.
    """
    p = path or RATINGS_CSV
    df = pd.read_csv(
        p,
        engine="pyarrow",
        usecols=RATINGS_USECOLS,
    )
    if "my_last_updated" in df.columns:
        df["my_last_updated"] = pd.to_datetime(df["my_last_updated"], errors="coerce")
    for col, dt in RATINGS_DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dt)
    return df
