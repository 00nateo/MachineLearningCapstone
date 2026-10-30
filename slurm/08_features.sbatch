#!/bin/bash
#SBATCH --job-name=features
#SBATCH --partition=normal_q
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate rideshare
cd "$SLURM_SUBMIT_DIR"
# Label + fold tags, then the model feature table (MODELING_PLAN.md). Optional args: city names.
python scripts/label.py "$@"
python scripts/features.py "$@"
