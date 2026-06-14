#!/bin/bash
#SBATCH --job-name=recsys-sweep
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=common

# Cascade operating-point sweep (no retraining). Re-runs evaluate_all with
# different cascade knobs and snapshots each resulting eval.md into
# artifacts/sweeps/ so you can pick the balanced recall/diversity point.
#
# Reuses the existing two_tower.pt (+ ranker.pt if present). Submit with:
#   sbatch --gres=gpu:1 scripts/slurm/07_sweep_cascade.sh

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  _SLURM_DIR="${SLURM_SUBMIT_DIR}/scripts/slurm"
else
  _SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=_common.sh
source "${_SLURM_DIR}/_common.sh"

SWEEP_DIR="artifacts/sweeps"
mkdir -p "${SWEEP_DIR}"

# Each entry: "tag|extra args to evaluate_all".
# Phase-1 reference (pure retrieval), then ranker blends and pool/MMR variants.
RUNS=(
  "phase1_retrieval|--no-ranker"
  "blend10|--rank-blend 1.0"
  "blend08|--rank-blend 0.8"
  "blend06|--rank-blend 0.6"
  "blend08_retr2000|--rank-blend 0.8 --pool-retrieve 2000"
  "blend08_rank100|--rank-blend 0.8 --pool-rank 100"
  "blend08_mmr085|--rank-blend 0.8 --mmr-lambda 0.85"
  "blend08_all|--rank-blend 0.8 --pool-retrieve 2000 --pool-rank 100 --mmr-lambda 0.85"
)

for entry in "${RUNS[@]}"; do
  tag="${entry%%|*}"
  extra="${entry#*|}"
  echo "=== sweep ${tag}: evaluate_all ${extra} ==="
  # shellcheck disable=SC2086
  python -m recsys.cli.evaluate_all ${extra}
  if [[ -f artifacts/eval.md ]]; then
    cp artifacts/eval.md "${SWEEP_DIR}/eval_${tag}.md"
    echo "  saved ${SWEEP_DIR}/eval_${tag}.md"
  fi
done

echo "=== sweep done; compare ${SWEEP_DIR}/eval_*.md ==="
