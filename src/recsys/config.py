from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
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


# Single source of truth for preprocess, training, baselines, Optuna, and eval subsampling.
SEED = 42
MIN_USER_RATINGS = 5
COMPLETED_STATUS = 2

# Cascade refactor: we no longer drop incomplete watches.
# MAL my_status values (per the dump):
#   1 = watching, 2 = completed, 3 = on-hold, 4 = dropped, 6 = plan-to-watch
# All five are kept so the MMoE ranker has completion/drop/start signal to learn from.
KEEP_STATUSES: tuple[int, ...] = (1, 2, 3, 4, 6)

# Rows with completion_fraction >= this AND status == 2 are eligible as "strong positives"
# used by the leave-one-out test target and the in-batch sampled softmax positive.
MIN_COMPLETION_FOR_POSITIVE = 0.8


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
    # When True the user history is pooled via a small Transformer (see models.sequence).
    # When False we fall back to the old score-weighted mean pool.
    use_sequence_encoder: bool = True
    use_score_weighted_pool: bool = True
    # Sequence encoder hyperparameters (only used when use_sequence_encoder=True).
    seq_n_layers: int = 4
    seq_n_heads: int = 4
    seq_ffn_mult: int = 2
    seq_p_mask_recent: float = 0.3
    seq_mask_window_min: int = 5
    seq_mask_window_max: int = 30
    # Completion-weighted loss. pos_weight = clip(completion_fraction, floor) * (1 + max(rating_z, 0)).
    use_completion_weighted_loss: bool = True
    completion_floor: float = 0.1
    # Curriculum hard negatives. K is linearly interpolated from K_easy at hard_neg_start_epoch
    # down to K_hard at the final epoch. Earlier epochs see far-out (easy) negatives;
    # later epochs see near-miss (hard) negatives.
    hard_neg_curriculum: bool = True
    # Easy pool size for curriculum (epoch 2). Keep well below catalog size — 5000
    # on ~6k items scans almost the whole index every batch and stalls training.
    hard_neg_K_easy: int = 512
    hard_neg_K_hard: int = 200


@dataclass
class TuneConfig:
    n_trials: int = 30
    pruner_warmup_epochs: int = 3
    sampler_seed: int = SEED


@dataclass
class RetrievalConfig:
    top_k: int = 10
    # Cascade pool sizes: Retrieve (FAISS) -> PreRank (light MLP) -> Rank (MMoE) -> ReRank.
    pool_retrieve: int = 1000
    pool_prerank: int = 200
    pool_rank: int = 50
    # Legacy aliases used by the old recommend_for_known_user serving helper.
    candidate_pool: int = 200
    rerank_pool: int = 100
    mmr_lambda: float = 0.7
    dpp_theta: float = 0.5
    reranker: str = "mmr"  # "mmr" (Phase 1/2 default) or "dpp" (Phase 3 default).
    exploration_epsilon: float = 0.1
    faiss_nlist: int = 64
    faiss_nprobe: int = 8
    # MMoE head weights at serve time. Final score = sum(weight * head_output).
    mmoe_w_completion: float = 0.5
    mmoe_w_rating: float = 0.3
    mmoe_w_drop: float = -0.2


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


def set_random_seed(
    seed: int | None = None,
    device: torch.device | None = None,
) -> np.random.Generator:
    """Seed NumPy and PyTorch RNGs for repeatable runs.

    Does not set ``cudnn.deterministic`` (keeps GPU fast; training may still vary slightly).
    """
    s = CFG.seed if seed is None else seed
    np.random.seed(s)
    torch.manual_seed(s)
    dev = device if device is not None else CFG.device
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    return np.random.default_rng(s)
