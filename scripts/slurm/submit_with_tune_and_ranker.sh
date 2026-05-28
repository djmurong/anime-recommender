#!/bin/bash
# Full cascade pipeline with Optuna + MMoE ranker, chained via SLURM dependencies:
#
#   preprocess
#     |- baselines
#         |- tune (Optuna)
#             |- train_two_tower --best
#                 |- build_index           (persists FAISS + EmbeddingStore)
#                     |- evaluate (Phase 1) -> writes artifacts/eval.md
#                         |- archive eval.md  -> artifacts/eval_phase1.md
#                             |- train_ranker (MMoE)
#                                 |- evaluate (Phase 2) -> overwrites artifacts/eval.md
#                                                          (cascade row picks up +MMoE)
#
# After the chain finishes you have:
#   artifacts/eval_phase1.md    -- baselines + brute-force TwoTower + cascade (no MMoE)
#   artifacts/eval.md           -- same, but the cascade row is +MMoE
#
# Usage (from repo root):
#   bash scripts/slurm/submit_with_tune_and_ranker.sh
#
# Override partitions / account / subset per run, e.g.:
#   RECSYS_PARTITION_GPU=gpu-a100 RECSYS_SUBSET=iter bash scripts/slurm/submit_with_tune_and_ranker.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

# Optuna writes best_params.json; train_two_tower --best consumes it.
export RECSYS_USE_BEST_PARAMS=1

cd "${RECSYS_ROOT}"
mkdir -p logs

ACCT=()
if [[ -n "${RECSYS_SLURM_ACCOUNT:-}" ]]; then
  ACCT=(--account="${RECSYS_SLURM_ACCOUNT}")
fi
# Slurm copies scripts to spool; force cwd back to the repo root so logs/ and
# SLURM_SUBMIT_DIR/scripts/slurm/ resolve from the right place.
CHDIR=(--chdir="${RECSYS_ROOT}")

echo "Project root: ${RECSYS_ROOT}"
echo "CPU partition: ${RECSYS_PARTITION_CPU}"
echo "GPU partition: ${RECSYS_PARTITION_GPU}"
echo "Subset:        ${RECSYS_SUBSET}"
echo "Tune trials:   ${RECSYS_TUNE_TRIALS}"
echo "Train epochs:  ${RECSYS_EPOCHS}"
echo "Ranker epochs: ${RANKER_EPOCHS:-5} (default from 04b_train_ranker.sh)"
echo ""

PREP_ID=$(sbatch "${ACCT[@]}" "${CHDIR[@]}" --parsable \
  --partition="${RECSYS_PARTITION_CPU}" \
  "${SCRIPT_DIR}/01_preprocess.sh")
echo "preprocess job:           ${PREP_ID}"

BASE_ID=$(sbatch "${ACCT[@]}" "${CHDIR[@]}" --parsable \
  --partition="${RECSYS_PARTITION_CPU}" \
  --dependency="afterok:${PREP_ID}" \
  "${SCRIPT_DIR}/02_train_baselines.sh")
echo "baselines job:            ${BASE_ID}"

TUNE_ID=$(sbatch "${ACCT[@]}" "${CHDIR[@]}" --parsable \
  --partition="${RECSYS_PARTITION_GPU}" \
  --dependency="afterok:${BASE_ID}" \
  "${SCRIPT_DIR}/03_tune.sh")
echo "tune job:                 ${TUNE_ID}"

TRAIN_ID=$(sbatch "${ACCT[@]}" "${CHDIR[@]}" --parsable \
  --partition="${RECSYS_PARTITION_GPU}" \
  --dependency="afterok:${TUNE_ID}" \
  "${SCRIPT_DIR}/04_train_two_tower.sh")
echo "train_two_tower job:      ${TRAIN_ID}"

# Build the FAISS index BEFORE evaluating so the cascade eval reads a
# persisted index instead of rebuilding in-memory each time.
INDEX_ID=$(sbatch "${ACCT[@]}" "${CHDIR[@]}" --parsable \
  --partition="${RECSYS_PARTITION_GPU}" \
  --dependency="afterok:${TRAIN_ID}" \
  "${SCRIPT_DIR}/06_build_index.sh")
echo "build_index job:          ${INDEX_ID}"

EVAL1_ID=$(sbatch "${ACCT[@]}" "${CHDIR[@]}" --parsable \
  --partition="${RECSYS_PARTITION_GPU}" \
  --dependency="afterok:${INDEX_ID}" \
  "${SCRIPT_DIR}/05_evaluate.sh")
echo "evaluate (Phase 1) job:   ${EVAL1_ID}"

# Tiny CPU job to snapshot the Phase-1 eval table before Phase 2 overwrites it.
ARCHIVE_ID=$(sbatch "${ACCT[@]}" "${CHDIR[@]}" --parsable \
  --partition="${RECSYS_PARTITION_CPU}" \
  --dependency="afterok:${EVAL1_ID}" \
  --job-name=recsys-archive-eval \
  --output=logs/archive_%j.out \
  --error=logs/archive_%j.err \
  --time=00:02:00 \
  --cpus-per-task=1 \
  --mem=1G \
  --wrap='cp artifacts/eval.md artifacts/eval_phase1.md && echo "archived artifacts/eval_phase1.md"')
echo "archive Phase-1 eval job: ${ARCHIVE_ID}"

RANKER_ID=$(sbatch "${ACCT[@]}" "${CHDIR[@]}" --parsable \
  --partition="${RECSYS_PARTITION_GPU}" \
  --dependency="afterok:${ARCHIVE_ID}" \
  "${SCRIPT_DIR}/04b_train_ranker.sh")
echo "train_ranker job:         ${RANKER_ID}"

EVAL2_ID=$(sbatch "${ACCT[@]}" "${CHDIR[@]}" --parsable \
  --partition="${RECSYS_PARTITION_GPU}" \
  --dependency="afterok:${RANKER_ID}" \
  "${SCRIPT_DIR}/05_evaluate.sh")
echo "evaluate (Phase 2) job:   ${EVAL2_ID}"

echo ""
echo "All jobs queued. SLURM will run them in order and cancel downstream jobs"
echo "if any earlier step fails (afterok dependency)."
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    logs/preprocess_${PREP_ID}.out  logs/train_${TRAIN_ID}.out"
echo "         logs/eval_${EVAL1_ID}.out      logs/ranker_${RANKER_ID}.out"
echo "         logs/eval_${EVAL2_ID}.out"
echo ""
echo "Cancel all (in order):"
echo "  scancel ${PREP_ID} ${BASE_ID} ${TUNE_ID} ${TRAIN_ID} ${INDEX_ID} ${EVAL1_ID} ${ARCHIVE_ID} ${RANKER_ID} ${EVAL2_ID}"
echo ""
echo "When done:"
echo "  cat artifacts/eval_phase1.md   # cascade WITHOUT MMoE"
echo "  cat artifacts/eval.md          # cascade WITH MMoE (+ baselines + brute-force TwoTower)"
