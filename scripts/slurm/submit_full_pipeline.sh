#!/bin/bash
# Submit the full pipeline with SLURM dependencies:
#   preprocess -> baselines -> train_two_tower -> evaluate -> build_index
#
# Usage (from repo root):
#   bash scripts/slurm/submit_full_pipeline.sh
#
# Optional env overrides (see scripts/slurm/config.sh):
#   RECSYS_PARTITION_CPU=shared RECSYS_PARTITION_GPU=gpu-a100 bash scripts/slurm/submit_full_pipeline.sh

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

echo "Project root: ${RECSYS_ROOT}"
echo "CPU partition: ${RECSYS_PARTITION_CPU}"
echo "GPU partition: ${RECSYS_PARTITION_GPU}"

PREP_ID=$(sbatch "${ACCT[@]}" --parsable --partition="${RECSYS_PARTITION_CPU}" \
  "${SCRIPT_DIR}/01_preprocess.sh")
echo "preprocess job: ${PREP_ID}"

BASE_ID=$(sbatch "${ACCT[@]}" --parsable --partition="${RECSYS_PARTITION_CPU}" \
  --dependency="afterok:${PREP_ID}" "${SCRIPT_DIR}/02_train_baselines.sh")
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

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Cancel:   scancel ${PREP_ID} ${BASE_ID} ${TRAIN_ID} ${EVAL_ID} ${INDEX_ID}"
echo "Logs:     logs/preprocess_${PREP_ID}.out  (etc.)"
