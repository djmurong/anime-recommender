# SLURM jobs for anime-recommender

## One-time setup (login node)

```bash
cd /path/to/anime-recommender

# 1) Edit cluster settings
nano scripts/slurm/config.sh
#   RECSYS_PARTITION_CPU, RECSYS_PARTITION_GPU, RECSYS_MODULE_PYTHON, RECSYS_SLURM_ACCOUNT

# 2) Python env + CUDA PyTorch (GPU training)
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match site CUDA

# 3) Data (if not already present)
#    data/anime_cleaned.csv, data/users_filtered.csv, data/animelists_filtered.csv

# 4) Optional: test GPU inside an allocation
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:15:00 --pty bash
source .venv/bin/activate
python -c "import torch; print(torch.cuda.is_available())"
exit
```

`torch.cuda.is_available()` is **False on the login node** — that is normal. GPU jobs use `04_train_two_tower.sh`.

## Submit pipelines

From **repo root**:

| Command | Steps |
|---------|--------|
| `bash scripts/slurm/submit_full_pipeline.sh` | preprocess → baselines → train → eval → index |
| `bash scripts/slurm/submit_train_eval.sh` | skip preprocess (need `artifacts/` already) |
| `bash scripts/slurm/submit_with_tune.sh` | full + Optuna + `train --best` |

Override partitions for one run:

```bash
RECSYS_PARTITION_CPU=shared RECSYS_PARTITION_GPU=gpu-v100 \
  bash scripts/slurm/submit_full_pipeline.sh
```

## Submit individual steps

```bash
sbatch --partition=cpu scripts/slurm/01_preprocess.sh
sbatch --partition=cpu scripts/slurm/02_train_baselines.sh
sbatch --partition=gpu --gres=gpu:1 scripts/slurm/04_train_two_tower.sh
sbatch --partition=gpu --gres=gpu:1 scripts/slurm/05_evaluate.sh
```

## Monitor / cancel

```bash
squeue -u $USER
tail -f logs/train_JOBID.out
scancel JOBID
```

## Troubleshooting

### `Batch script contains DOS line breaks (\r\n)`

Scripts edited on Windows (or copied into Open OnDemand from Windows) use CRLF line endings. Linux `sbatch` requires LF only.

From the repo on the cluster:

```bash
sed -i 's/\r$//' scripts/slurm/*.sh
# or: dos2unix scripts/slurm/*.sh
```

If you use **Open OnDemand → Jobs**, fix the script under your OnDemand project path too, or submit from a terminal:

```bash
cd /work/djm99/anime-recommender
bash scripts/slurm/submit_full_pipeline.sh
```

In Cursor/VS Code, set **Files: Eol** to `\n` for shell scripts. This repo uses `.gitattributes` to keep `*.sh` as LF.

## Copy results home

```bash
rsync -avP user@cluster:/path/to/anime-recommender/artifacts/ ./artifacts/
```

Key files: `artifacts/models/two_tower.pt`, `artifacts/eval.md`, `artifacts/best_params.json` (if tuned).

## Job scripts

| Script | Purpose |
|--------|---------|
| `01_preprocess.sh` | Build splits, maps, features |
| `02_train_baselines.sh` | Popularity / MF / content baselines |
| `03_tune.sh` | Optuna search (optional) |
| `04_train_two_tower.sh` | Main GPU training |
| `05_evaluate.sh` | Metrics table → `artifacts/eval.md` |
| `06_build_index.sh` | FAISS index for serving |
