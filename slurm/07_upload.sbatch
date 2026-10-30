#!/bin/bash
#SBATCH --job-name=upload
#SBATCH --partition=normal_q
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate rideshare
cd "$SLURM_SUBMIT_DIR"
: "${HF_REPO:?set HF_REPO=<org>/<dataset>}"

# --private only matters if the repo doesn't exist yet: it is never created public.
# --delete removes parquet files that are no longer part of the export.
hf upload "$HF_REPO" data/export . --repo-type dataset --private \
  --delete "*.parquet" \
  --commit-message "export $(date -u +%Y-%m-%dT%H:%MZ) from $(git rev-parse --short HEAD 2>/dev/null || echo uncommitted)"
