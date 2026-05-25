#!/bin/bash
# Use when artifacts/ already exists (skip preprocess).
# Runs: baselines -> train_two_tower -> evaluate -> build_index
#
# Usage:
#   bash scripts/slurm/submit_train_eval.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

cd "${RECSYS_ROOT}"
mkdir -p logs

ACCT=()
if [[ -n "${RECSYS_SLURM_ACCOUNT:-}" ]]; then
  ACCT=(--account="${RECSYS_SLURM_ACCOUNT}")
fi

BASE_ID=$(sbatch "${ACCT[@]}" --parsable --partition="${RECSYS_PARTITION_CPU}" \
  "${SCRIPT_DIR}/02_train_baselines.sh")
echo "baselines job: ${BASE_ID}"

TRAIN_ID=$(sbatch "${ACCT[@]}" --parsable --partition="${RECSYS_PARTITION_GPU}" \
  --dependency="afterok:${BASE_ID}" "${SCRIPT_DIR}/04_train_two_tower.sh")
echo "train job: ${TRAIN_ID}"

EVAL_ID=$(sbatch "${ACCT[@]}" --parsable --partition="${RECSYS_PARTITION_GPU}" \
  --dependency="afterok:${TRAIN_ID}" "${SCRIPT_DIR}/05_evaluate.sh")
echo "eval job: ${EVAL_ID}"

INDEX_ID=$(sbatch "${ACCT[@]}" --parsable --partition="${RECSYS_PARTITION_GPU}" \
  --dependency="afterok:${EVAL_ID}" "${SCRIPT_DIR}/06_build_index.sh")
echo "index job: ${INDEX_ID}"

echo "Monitor: squeue -u \$USER"
