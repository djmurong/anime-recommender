#!/bin/bash
#SBATCH --job-name=recsys-index
#SBATCH --output=logs/index_%j.out
#SBATCH --error=logs/index_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

#SBATCH --partition=gpu

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

echo "=== build_index ==="
python -m recsys.cli.build_index
echo "=== done ==="
