# Shared runtime setup for all recsys SLURM jobs.
# Sourced from scripts/slurm/*.sh — BASH_SOURCE here is the real repo path, not Slurm spool.
set -euo pipefail

RECSYS_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${RECSYS_SLURM_DIR}/config.sh"

cd "${RECSYS_ROOT}"
mkdir -p logs

if [[ -n "${RECSYS_MODULE_PYTHON}" ]]; then
  # shellcheck disable=SC1090
  eval "${RECSYS_MODULE_PYTHON}"
fi

if [[ ! -d ".venv" ]]; then
  echo "ERROR: .venv not found in ${RECSYS_ROOT}. Create it first:"
  echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -e ."
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

python -c "import recsys; print('recsys import ok')"
