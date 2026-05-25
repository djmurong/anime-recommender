#!/bin/bash
#SBATCH --job-name=recsys-baselines
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

#SBATCH --partition=cpu

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

echo "=== train_baselines ==="
python -m recsys.cli.train_baselines
echo "=== done ==="
