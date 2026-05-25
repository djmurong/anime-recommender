from __future__ import annotations

import numpy as np
import pandas as pd


def build_popularity_bias_vector(
    anime_df: pd.DataFrame,
    anime_map: dict[int, int],
) -> np.ndarray:
    """Bayesian-shrunk popularity scores per dense anime index (same as PopularityRec)."""
    n = len(anime_map)
    scores = np.zeros(n, dtype=np.float32)
    members = pd.to_numeric(anime_df["members"], errors="coerce").fillna(0.0).to_numpy()
    score_col = pd.to_numeric(anime_df["score"], errors="coerce").fillna(0.0).to_numpy()
    global_mean = float(np.nanmean(score_col[score_col > 0])) if (score_col > 0).any() else 6.5
    c = float(np.median(members[members > 0])) if (members > 0).any() else 1000.0
    bayes = (members * score_col + c * global_mean) / (members + c + 1e-9)
    bayes = bayes * np.log1p(members)
    anime_ids = anime_df["anime_id"].astype(int).to_numpy()
    for row, raw_id in enumerate(anime_ids):
        idx = anime_map.get(int(raw_id))
        if idx is not None:
            scores[idx] = float(bayes[row])
    if scores.std() > 1e-8:
        scores = (scores - scores.mean()) / scores.std()
    return scores.astype(np.float32)
