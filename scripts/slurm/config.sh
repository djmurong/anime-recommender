# Edit these for your cluster (see: sinfo, module avail, site docs)
# Source this file from other scripts in this directory — do not run alone.

# SLURM partitions (queue names)
export RECSYS_PARTITION_CPU="common"
export RECSYS_PARTITION_GPU="scavenger-gpu"

# Uncomment if your site requires an account flag:
# export RECSYS_SLURM_ACCOUNT="#SBATCH --account=your_account"
export RECSYS_SLURM_ACCOUNT="dataplus"

# Optional: load site modules before activating the venv
# Example: export RECSYS_MODULE_PYTHON="module load python3"
export RECSYS_MODULE_PYTHON="${RECSYS_MODULE_PYTHON:-}"

# Project root (directory that contains src/, data/, artifacts/)
export RECSYS_ROOT="${RECSYS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Preprocess subset: iter (fast) or full
export RECSYS_SUBSET="${RECSYS_SUBSET:-full}"

# Training
export RECSYS_EPOCHS="${RECSYS_EPOCHS:-10}"
export RECSYS_BATCH_SIZE="${RECSYS_BATCH_SIZE:-2048}"
export RECSYS_USE_BEST_PARAMS="${RECSYS_USE_BEST_PARAMS:-0}"  # set to 1 after tune

# Optuna (optional step)
export RECSYS_TUNE_TRIALS="${RECSYS_TUNE_TRIALS:-30}"
export RECSYS_TUNE_MAX_ROWS="${RECSYS_TUNE_MAX_ROWS:-2000000}"
