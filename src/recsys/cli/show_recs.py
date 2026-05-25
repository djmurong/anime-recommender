"""Print sample recommendations for a user (inspect a trained checkpoint).

Run (after preprocess + train on the same subset):
    python -m recsys.cli.show_recs
    python -m recsys.cli.show_recs --username some_mal_user
    python -m recsys.cli.show_recs --model popularity
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import pandas as pd
import torch

from recsys.config import ARTIFACTS_DIR, CACHE_DIR, CFG, MODELS_DIR, resolve_device, set_thread_env
from recsys.data.features_anime import build_anime_features
from recsys.data.features_user import RECENCY_BUCKETS
from recsys.data.load import load_anime
from recsys.data.split import load_splits
from recsys.models.popularity_bias import build_popularity_bias_vector
from recsys.models.two_tower import encode_all_anime, feature_pack_to_tensors, load_two_tower_checkpoint, score_all_items
from recsys.training.trainer import encode_history_batch


def _load_maps() -> tuple[dict[str, int], dict[int, int], dict[int, int]]:
    user_map = json.loads((ARTIFACTS_DIR / "user_map.json").read_text())
    anime_map = {int(k): v for k, v in json.loads((ARTIFACTS_DIR / "anime_map.json").read_text()).items()}
    idx_to_anime = {v: k for k, v in anime_map.items()}
    return user_map, anime_map, idx_to_anime


def _titles(anime_df: pd.DataFrame) -> dict[int, str]:
    ids = anime_df["anime_id"].astype(int).to_numpy()
    titles = anime_df["title_english"].fillna(anime_df["title"]).astype(str).to_numpy()
    return dict(zip(ids, titles))


def _two_tower_topk(
    username: str,
    k: int,
    user_map: dict[str, int],
    idx_to_anime: dict[int, int],
    titles: dict[int, str],
    feats: dict,
    user_features,
    device: torch.device,
) -> list[tuple[str, float, str | None]]:
    ckpt_path = MODELS_DIR / "two_tower.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Run cli.train_two_tower first.")

    u_idx = user_map.get(username)
    if u_idx is None:
        raise KeyError(f"User '{username}' not in user_map.json (wrong subset or typo).")

    _, anime_map, _ = _load_maps()
    anime_df = load_anime().sort_values("anime_id", kind="stable").reset_index(drop=True)
    pop_bias = build_popularity_bias_vector(anime_df, anime_map)
    model = load_two_tower_checkpoint(
        ckpt_path,
        recency_dim=RECENCY_BUCKETS,
        n_anime=feats["numerical"].shape[0],
        popularity_bias=pop_bias,
    )
    model = model.to(device).eval()

    anime_tensors = feature_pack_to_tensors(feats, device)
    history = user_features.history.get(int(u_idx), np.zeros(0, dtype=np.int64))
    hist_scores = user_features.history_scores.get(int(u_idx), np.zeros(0, dtype=np.float32))
    max_h = CFG.train.max_history_len
    if len(history) > max_h:
        history = history[-max_h:]
        hist_scores = hist_scores[-max_h:]
    max_h = max(len(history), 1)
    hist = np.zeros((1, max_h), dtype=np.int64)
    mask = np.zeros((1, max_h), dtype=np.float32)
    weights = np.zeros((1, max_h), dtype=np.float32)
    if len(history):
        hist[0, : len(history)] = history
        mask[0, : len(history)] = 1.0
        mu = float(user_features.centered_avg_score[int(u_idx)]) + 7.0
        weights[0, : len(history)] = np.maximum(hist_scores - mu, 0.1)

    with torch.no_grad():
        all_anime = encode_all_anime(model, anime_tensors)
        hist_t = torch.from_numpy(hist).to(device)
        mask_t = torch.from_numpy(mask).to(device)
        w_t = torch.from_numpy(weights).to(device) if CFG.train.use_score_weighted_pool else None
        pooled = encode_history_batch(model, hist_t, mask_t, anime_tensors, w_t)
        affinity = torch.from_numpy(user_features.genre_affinity[int(u_idx) : int(u_idx) + 1]).to(device)
        centered = torch.from_numpy(user_features.centered_avg_score[int(u_idx) : int(u_idx) + 1]).unsqueeze(-1).to(device)
        recency = torch.from_numpy(user_features.recency[int(u_idx) : int(u_idx) + 1]).to(device)
        user_emb = model.encode_user(pooled, affinity, centered, recency)
        scores = score_all_items(model, user_emb, all_anime)
        if len(history):
            scores[0, history] = -float("inf")
        vals, topk = scores.topk(k, dim=1)
        topk = topk.cpu().numpy()[0]
        top_scores = vals.cpu().numpy()[0]

    out: list[tuple[str, float, str | None]] = []
    for a_idx, sc in zip(topk, top_scores):
        aid = idx_to_anime.get(int(a_idx), -1)
        out.append((titles.get(aid, f"anime_id={aid}"), float(sc), None))
    return out


def _baseline_topk(
    model_key: str,
    username: str,
    k: int,
    user_map: dict[str, int],
    idx_to_anime: dict[int, int],
    titles: dict[int, str],
    user_features,
) -> list[tuple[str, float, str | None]]:
    path = MODELS_DIR / "baselines.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No baselines at {path}. Run cli.train_baselines first.")
    with open(path, "rb") as f:
        bls = pickle.load(f)
    rec = bls[model_key]
    u_idx = user_map.get(username)
    if u_idx is None:
        raise KeyError(f"User '{username}' not in user_map.json.")
    excl = user_features.history.get(int(u_idx), np.zeros(0, dtype=np.int64))
    idxs, scores = rec.recommend(int(u_idx), k, excl)
    return [
        (titles.get(idx_to_anime.get(int(i), -1), f"idx={i}"), float(s), None)
        for i, s in zip(idxs, scores)
    ]


def main() -> None:
    set_thread_env()
    p = argparse.ArgumentParser(description="Print top-k recommendations for one user.")
    p.add_argument("--username", help="MAL username (default: random user from test split)")
    p.add_argument("--model", choices=["two_tower", "popularity", "content", "mf"], default="two_tower")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--cpu", action="store_true", help="Use CPU (shorthand for --device cpu).")
    args = p.parse_args()

    device = resolve_device("cpu" if args.cpu else args.device)
    _, val, test = load_splits()
    user_map, _, idx_to_anime = _load_maps()
    anime_df = load_anime().sort_values("anime_id", kind="stable").reset_index(drop=True)
    titles = _titles(anime_df)
    feats = build_anime_features(anime_df)
    with open(CACHE_DIR / "user_features.pkl", "rb") as f:
        user_features = pickle.load(f)

    username = args.username
    held_out_title: str | None = None
    if not username:
        row = test.sample(1, random_state=CFG.seed).iloc[0]
        username = str(row["username"])
        held_out_id = int(row["anime_id"])
        held_out_title = titles.get(held_out_id, f"anime_id={held_out_id}")

    print(f"User: {username}")
    if held_out_title:
        print(f"Held-out test item (not in history): {held_out_title}")
    print(f"Model: {args.model}")
    print()

    if args.model == "two_tower":
        rows = _two_tower_topk(username, args.k, user_map, idx_to_anime, titles, feats, user_features, device)
    else:
        key = {"popularity": "popularity", "content": "content", "mf": "mf"}[args.model]
        rows = _baseline_topk(key, username, args.k, user_map, idx_to_anime, titles, user_features)

    for rank, (title, score, _) in enumerate(rows, start=1):
        mark = " *" if held_out_title and title == held_out_title else ""
        print(f"  {rank:2d}. {title}  (score={score:.4f}){mark}")
    if held_out_title:
        print()
        print("  (* = test target; hit if it appears above)")


if __name__ == "__main__":
    main()
