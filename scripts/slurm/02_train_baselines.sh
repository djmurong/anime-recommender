#!/bin/bash
#SBATCH --job-name=recsys-baselines
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

#SBATCH --partition=common

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  _SLURM_DIR="${SLURM_SUBMIT_DIR}/scripts/slurm"
else
  _SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=_common.sh
source "${_SLURM_DIR}/_common.sh"

echo "=== train_baselines ==="
python -m recsys.cli.train_baselines
echo "=== done ==="
