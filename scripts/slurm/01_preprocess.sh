#!/bin/bash
#SBATCH --job-name=recsys-preprocess
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Partition/account overridden by submit_*.sh (sbatch --partition=...)
#SBATCH --partition=common

set -euo pipefail
# Slurm copies the script to spool; use submit cwd (repo root) to find scripts/slurm/.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  _SLURM_DIR="${SLURM_SUBMIT_DIR}/scripts/slurm"
else
  _SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=_common.sh
source "${_SLURM_DIR}/_common.sh"

echo "=== preprocess (subset=${RECSYS_SUBSET}) ==="
python -m recsys.cli.preprocess --subset "${RECSYS_SUBSET}"
echo "=== done ==="
