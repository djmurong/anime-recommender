# Data directory

Large MAL dataset files are **not** committed to git (see root `.gitignore`).

Place cleaned/filtered CSVs here before running preprocessing, for example:

- `anime_cleaned.csv`
- `users_filtered.csv`
- `animelists_filtered.csv`

Then from the repo root:

```bash
python -m recsys.cli.preprocess --subset full   # or --subset iter
```

Generated training artifacts are written under `artifacts/` (also gitignored).
