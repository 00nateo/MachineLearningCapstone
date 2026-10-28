#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=normal_q
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%j.out

# Usage: sbatch --account=$ACCOUNT slurm/09_train.sbatch --city nyc --fold fold3
set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate rideshare
cd "$SLURM_SUBMIT_DIR"
python scripts/features.py
python scripts/sample.py "$@"
python scripts/baseline.py "$@"
python scripts/train.py "$@"
