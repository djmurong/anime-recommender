from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

CACHE_DIR = ARTIFACTS_DIR / "cache"
SPLITS_DIR = ARTIFACTS_DIR / "splits"
MODELS_DIR = ARTIFACTS_DIR / "models"
INDEX_DIR = ARTIFACTS_DIR / "indexes"


for d in (ARTIFACTS_DIR, CACHE_DIR, SPLITS_DIR, MODELS_DIR, INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True)


ANIME_CSV = DATA_DIR / "anime_cleaned.csv"
USERS_CSV = DATA_DIR / "users_filtered.csv"
RATINGS_CSV = DATA_DIR / "animelists_filtered.csv"


SEED = 42
MIN_USER_RATINGS = 5
COMPLETED_STATUS = 2


@dataclass
class SubsetConfig:
    name: str = "iter"
    max_users: int = 30_000
    min_user_ratings: int = 20
    max_user_ratings: int = 500


SUBSET_PRESETS = {
    "iter": SubsetConfig("iter", 30_000, 20, 500),
    "full": SubsetConfig("full", 10**9, MIN_USER_RATINGS, 10**9),
}


@dataclass
class TrainConfig:
    embedding_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.2
    studio_emb_dim: int = 32
    studio_min_count: int = 10
    temperature: float = 0.05
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-6
    epochs: int = 10
    hard_neg_ratio: int = 4
    hard_neg_start_epoch: int = 2
    hard_neg_refresh_every: int = 2
    val_every_epoch: int = 1
    grad_clip: float = 1.0
    num_workers: int = 0
    catalog_neg_count: int = 64
    catalog_neg_weight: float = 1.0
    recency_sample_tau_days: float = 180.0
    max_history_len: int = 100
    use_score_weighted_pool: bool = True


@dataclass
class TuneConfig:
    n_trials: int = 30
    pruner_warmup_epochs: int = 3
    sampler_seed: int = SEED


@dataclass
class RetrievalConfig:
    top_k: int = 10
    candidate_pool: int = 200
    rerank_pool: int = 100
    mmr_lambda: float = 0.7
    exploration_epsilon: float = 0.1
    faiss_nlist: int = 64
    faiss_nprobe: int = 8


@dataclass
class FeatureConfig:
    synopsis_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    synopsis_dim: int = 384
    numerical_cols: tuple = (
        "episodes",
        "duration_min",
        "score",
        "scored_by",
        "members",
        "favorites",
        "popularity",
        "aired_from_year",
    )


def get_device() -> torch.device:
    """Pick the best available device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(choice: str) -> torch.device:
    """Map CLI device string to torch.device with safe fallbacks."""
    key = choice.strip().lower()
    if key == "auto":
        return get_device()
    if key == "cpu":
        return torch.device("cpu")
    if key == "cuda":
        if not torch.cuda.is_available():
            print(
                "WARNING: --device cuda requested but CUDA is not available "
                "(CPU-only PyTorch or no GPU). Falling back to CPU."
            )
            return torch.device("cpu")
        return torch.device("cuda")
    if key == "mps":
        if not torch.backends.mps.is_available():
            print("WARNING: --device mps requested but MPS is not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("mps")
    raise ValueError(f"Unknown device '{choice}'. Use auto, cpu, cuda, or mps.")


@dataclass
class GlobalConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    device: torch.device = field(default_factory=get_device)
    seed: int = SEED


CFG = GlobalConfig()


def set_thread_env():
    """Limit BLAS threads to avoid implicit/numpy contention on Windows."""
    for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ.setdefault(var, "1")
