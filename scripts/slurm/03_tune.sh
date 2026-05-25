#!/bin/bash
#SBATCH --job-name=recsys-tune
#SBATCH --output=logs/tune_%j.out
#SBATCH --error=logs/tune_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

#SBATCH --partition=gpu

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

python -c "import torch; print('cuda available:', torch.cuda.is_available())"

echo "=== tune (trials=${RECSYS_TUNE_TRIALS}, max_rows=${RECSYS_TUNE_MAX_ROWS}) ==="
python -m recsys.cli.tune --trials "${RECSYS_TUNE_TRIALS}" --max-train-rows "${RECSYS_TUNE_MAX_ROWS}"
echo "=== done — set RECSYS_USE_BEST_PARAMS=1 for training ==="
