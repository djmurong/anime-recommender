#!/bin/bash
#SBATCH --job-name=recsys-ranker
#SBATCH --output=logs/ranker_%j.out
#SBATCH --error=logs/ranker_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

#SBATCH --partition=common

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  _SLURM_DIR="${SLURM_SUBMIT_DIR}/scripts/slurm"
else
  _SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=_common.sh
source "${_SLURM_DIR}/_common.sh"

python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

# Default to 5 epochs unless overridden.
RANKER_EPOCHS="${RANKER_EPOCHS:-5}"
RANKER_BATCH_SIZE="${RANKER_BATCH_SIZE:-1024}"

echo "=== train_ranker epochs=${RANKER_EPOCHS} batch=${RANKER_BATCH_SIZE} ==="
python -m recsys.cli.train_ranker \
  --epochs "${RANKER_EPOCHS}" \
  --device cuda \
  --batch-size "${RANKER_BATCH_SIZE}"
echo "=== done ==="
