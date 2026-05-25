"""Fit and persist the three baseline recommenders.

Run:
    python -m recsys.cli.train_baselines
"""
from __future__ import annotations

import json
import pickle

from recsys.config import ARTIFACTS_DIR, MODELS_DIR, set_thread_env
from recsys.data.features_anime import build_anime_features
from recsys.data.load import load_anime
from recsys.data.split import load_splits
from recsys.models.baselines import ContentCosineRec, ImplicitMFRec, PopularityRec


def _load_id_maps() -> tuple[dict[str, int], dict[int, int]]:
    user_map = json.loads((ARTIFACTS_DIR / "user_map.json").read_text())
    anime_map_raw = json.loads((ARTIFACTS_DIR / "anime_map.json").read_text())
    anime_map = {int(k): v for k, v in anime_map_raw.items()}
    return user_map, anime_map


def _build_content_matrix(feats: dict) -> "np.ndarray":
    import numpy as np

    return np.concatenate(
        [feats["numerical"], feats["genres"], feats["synopsis"]], axis=1
    ).astype(np.float32)


def main() -> None:
    set_thread_env()
    train, _, _ = load_splits()
    anime_df = load_anime().sort_values("anime_id", kind="stable").reset_index(drop=True)
    user_map, anime_map = _load_id_maps()
    feats = build_anime_features(anime_df)

    print("Fitting Popularity...")
    pop = PopularityRec().fit(anime_df, anime_map)
    print("Fitting ContentCosine...")
    cos = ContentCosineRec().fit(_build_content_matrix(feats), train, user_map, anime_map)
    print("Fitting ImplicitMF...")
    mf = ImplicitMFRec().fit(train, user_map, anime_map)

    out = MODELS_DIR / "baselines.pkl"
    with open(out, "wb") as f:
        pickle.dump({"popularity": pop, "content": cos, "mf": mf}, f)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
