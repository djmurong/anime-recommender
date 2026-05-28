"""Versioned embedding store.

Embeddings are first-class artifacts now: every two-tower training run can
ship a new immutable version under `artifacts/embeddings/<version>/`, and
downstream consumers (FAISS, PreRanker, MMoE, DPP) load by version tag rather
than reading scattered `.npy` files. The `current` symlink (or copy on
Windows where symlinks are gated) points to the active version so the rest of
the pipeline keeps working without code changes when you roll back.

Layout:
    artifacts/embeddings/<version>/
        item_emb.npy
        item_ids.npy
        manifest.json
    artifacts/embeddings/current/  (-> <version>)

The manifest records `model_hash` (sha256 over the embedding bytes), version
tag, creation time, dim, count, and any caller-supplied metadata (e.g.
`model_ckpt_epoch`).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from recsys.config import ARTIFACTS_DIR


EMBEDDING_STORE_DIR = ARTIFACTS_DIR / "embeddings"
ITEM_EMB_FILE = "item_emb.npy"
ITEM_IDS_FILE = "item_ids.npy"
MANIFEST_FILE = "manifest.json"
CURRENT_NAME = "current"


@dataclass
class EmbeddingVersion:
    """In-memory handle to one stored version."""

    version_tag: str
    embeddings: np.ndarray             # [n, D] float32
    anime_ids: np.ndarray              # [n] int64
    manifest: dict


def _embedding_hash(embeddings: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(embeddings.astype(np.float32).tobytes())
    return h.hexdigest()


def _set_current(store_root: Path, version_dir: Path) -> None:
    """Point `current/` at `version_dir`. Uses a symlink when supported,
    falls back to copying contents on Windows without dev mode."""
    current = store_root / CURRENT_NAME
    if current.exists() or current.is_symlink():
        if current.is_symlink() or current.is_file():
            current.unlink()
        elif current.is_dir():
            shutil.rmtree(current)
    try:
        os.symlink(version_dir.name, current, target_is_directory=True)
        return
    except OSError:
        # Symlink unsupported (typical on Windows without dev mode).
        current.mkdir(parents=True, exist_ok=True)
        for name in (ITEM_EMB_FILE, ITEM_IDS_FILE, MANIFEST_FILE):
            src = version_dir / name
            if src.exists():
                shutil.copy2(src, current / name)


class EmbeddingStore:
    """Filesystem-backed versioned store."""

    def __init__(self, root: Path | None = None):
        self.root = root or EMBEDDING_STORE_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- writing ----

    def save(
        self,
        embeddings: np.ndarray,
        anime_ids: np.ndarray,
        version_tag: str | None = None,
        manifest_extras: dict | None = None,
        make_current: bool = True,
    ) -> Path:
        """Materialize a new immutable version. Returns the version directory."""
        if version_tag is None:
            version_tag = datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")
        if version_tag == CURRENT_NAME:
            raise ValueError(f"version_tag '{CURRENT_NAME}' is reserved")
        version_dir = self.root / version_tag
        if version_dir.exists():
            raise FileExistsError(
                f"EmbeddingStore version '{version_tag}' already exists at {version_dir}"
            )
        version_dir.mkdir(parents=True, exist_ok=False)

        emb = embeddings.astype(np.float32)
        ids = anime_ids.astype(np.int64)
        np.save(version_dir / ITEM_EMB_FILE, emb)
        np.save(version_dir / ITEM_IDS_FILE, ids)

        manifest = {
            "version_tag": version_tag,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding_hash": _embedding_hash(emb),
            "count": int(emb.shape[0]),
            "dim": int(emb.shape[1]),
        }
        if manifest_extras:
            manifest.update(manifest_extras)
        (version_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))

        if make_current:
            _set_current(self.root, version_dir)
        return version_dir

    # ---- reading ----

    def list_versions(self) -> list[str]:
        return sorted(
            p.name
            for p in self.root.iterdir()
            if p.is_dir() and p.name != CURRENT_NAME
        )

    def load(self, version_tag: str = CURRENT_NAME) -> EmbeddingVersion:
        version_dir = self.root / version_tag
        if not (version_dir / ITEM_EMB_FILE).exists():
            raise FileNotFoundError(
                f"EmbeddingStore version '{version_tag}' not found at {version_dir}"
            )
        emb = np.load(version_dir / ITEM_EMB_FILE).astype(np.float32)
        ids = np.load(version_dir / ITEM_IDS_FILE).astype(np.int64)
        manifest_path = version_dir / MANIFEST_FILE
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        return EmbeddingVersion(
            version_tag=version_tag,
            embeddings=emb,
            anime_ids=ids,
            manifest=manifest,
        )

    # ---- diff / monitoring ----

    def diff(self, v_a: str, v_b: str) -> dict:
        """Per-item drift between two versions.

        Returns a dict with `mean_cosine_drift`, `p95_cosine_drift`, and the
        set of newly added / removed anime ids.
        """
        a = self.load(v_a)
        b = self.load(v_b)
        ids_a = {int(x): i for i, x in enumerate(a.anime_ids)}
        ids_b = {int(x): i for i, x in enumerate(b.anime_ids)}
        common = sorted(set(ids_a) & set(ids_b))
        if not common:
            return {
                "common_count": 0,
                "added": sorted(set(ids_b) - set(ids_a)),
                "removed": sorted(set(ids_a) - set(ids_b)),
            }
        emb_a = a.embeddings[np.array([ids_a[i] for i in common], dtype=np.int64)]
        emb_b = b.embeddings[np.array([ids_b[i] for i in common], dtype=np.int64)]
        # Cosine drift = 1 - cosine_similarity per row.
        emb_a = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-9)
        emb_b = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-9)
        cos = (emb_a * emb_b).sum(axis=1).clip(-1.0, 1.0)
        drift = 1.0 - cos
        return {
            "common_count": len(common),
            "added": sorted(set(ids_b) - set(ids_a)),
            "removed": sorted(set(ids_a) - set(ids_b)),
            "mean_cosine_drift": float(drift.mean()),
            "p95_cosine_drift": float(np.quantile(drift, 0.95)),
        }
