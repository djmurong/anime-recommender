#!/bin/bash
# Full pipeline including Optuna before training:
#   preprocess -> baselines -> tune -> train (--best) -> evaluate -> build_index
#
# Before running, set in config.sh or env:
#   export RECSYS_USE_BEST_PARAMS=1
#
# Usage:
#   RECSYS_USE_BEST_PARAMS=1 bash scripts/slurm/submit_with_tune.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

export RECSYS_USE_BEST_PARAMS=1

cd "${RECSYS_ROOT}"
mkdir -p logs

ACCT=()
if [[ -n "${RECSYS_SLURM_ACCOUNT:-}" ]]; then
  ACCT=(--account="${RECSYS_SLURM_ACCOUNT}")
fi
CHDIR=(--chdir="${RECSYS_ROOT}")

PREP_ID=$(sbatch "${ACCT[@]}" "${CHDIR[@]}" --parsable --partition="${RECSYS_PARTITION_CPU}" \
  "${SCRIPT_DIR}/01_preprocess.sh")

BASE_ID=$(sbatch "${ACCT[@]}" "${CHDIR[@]}" --parsable --partition="${RECSYS_PARTITION_CPU}" \
  --dependency="afterok:${PREP_ID}" "${SCRIPT_DIR}/02_train_baselines.sh")

TUNE_ID=$(sbatch "${ACCT[@]}" "${CHDIR[@]}" --parsable --partition="${RECSYS_PARTITION_GPU}" \
  --dependency="afterok:${BASE_ID}" "${SCRIPT_DIR}/03_tune.sh")

TRAIN_ID=$(sbatch "${ACCT[@]}" "${CHDIR[@]}" --parsable --partition="${RECSYS_PARTITION_GPU}" \
  --dependency="afterok:${TUNE_ID}" "${SCRIPT_DIR}/04_train_two_tower.sh")

EVAL_ID=$(sbatch "${ACCT[@]}" "${CHDIR[@]}" --parsable --partition="${RECSYS_PARTITION_GPU}" \
  --dependency="afterok:${TRAIN_ID}" "${SCRIPT_DIR}/05_evaluate.sh")

INDEX_ID=$(sbatch "${ACCT[@]}" "${CHDIR[@]}" --parsable --partition="${RECSYS_PARTITION_GPU}" \
  --dependency="afterok:${EVAL_ID}" "${SCRIPT_DIR}/06_build_index.sh")

echo "Jobs: prep=${PREP_ID} base=${BASE_ID} tune=${TUNE_ID} train=${TRAIN_ID} eval=${EVAL_ID} index=${INDEX_ID}"
