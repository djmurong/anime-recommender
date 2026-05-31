# Edit these for your cluster (see: sinfo, module avail, site docs)
# Source this file from other scripts in this directory — do not run alone.

# Project root (directory that contains src/, data/, artifacts/)
export RECSYS_ROOT="${RECSYS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Optional: load RECSYS_* values from repo-root .env
if [[ -f "${RECSYS_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${RECSYS_ROOT}/.env"
  set +a
fi

# SLURM partitions (queue names)
export RECSYS_PARTITION_CPU="${RECSYS_PARTITION_CPU:-common}"
export RECSYS_PARTITION_GPU="${RECSYS_PARTITION_GPU:-common-gpu}"

# Slurm account name (example: dataplus). Leave empty if site/account does not require it.
export RECSYS_SLURM_ACCOUNT="${RECSYS_SLURM_ACCOUNT:-dataplus}"

# Optional: load site modules before activating the venv
# Example: RECSYS_MODULE_PYTHON="module load python3"
export RECSYS_MODULE_PYTHON="${RECSYS_MODULE_PYTHON:-}"

# Preprocess subset: iter (fast) or full
export RECSYS_SUBSET="${RECSYS_SUBSET:-full}"

# Training
export RECSYS_EPOCHS="${RECSYS_EPOCHS:-10}"
export RECSYS_BATCH_SIZE="${RECSYS_BATCH_SIZE:-2048}"
export RECSYS_USE_BEST_PARAMS="${RECSYS_USE_BEST_PARAMS:-0}"  # set to 1 after tune
export RECSYS_NUM_WORKERS="${RECSYS_NUM_WORKERS:-8}"  # DataLoader workers (0 = GPU starves)

# Optuna (optional step)
export RECSYS_TUNE_TRIALS="${RECSYS_TUNE_TRIALS:-30}"
export RECSYS_TUNE_MAX_ROWS="${RECSYS_TUNE_MAX_ROWS:-2000000}"
