#!/bin/bash
#SBATCH --job-name=recsys-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

#SBATCH --partition=gpu

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

BEST_FLAG=()
if [[ "${RECSYS_USE_BEST_PARAMS}" == "1" ]]; then
  BEST_FLAG=(--best)
fi

echo "=== train_two_tower epochs=${RECSYS_EPOCHS} batch=${RECSYS_BATCH_SIZE} ==="
python -m recsys.cli.train_two_tower \
  "${BEST_FLAG[@]}" \
  --epochs "${RECSYS_EPOCHS}" \
  --device cuda \
  --batch-size "${RECSYS_BATCH_SIZE}"
echo "=== done ==="
