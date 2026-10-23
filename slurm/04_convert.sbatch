#!/bin/bash
#SBATCH --job-name=convert
#SBATCH --partition=normal_q
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate rideshare
cd "$SLURM_SUBMIT_DIR"
python scripts/convert.py
