#!/bin/bash
#SBATCH --job-name=event_trips
#SBATCH --partition=normal_q
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate rideshare
cd "$SLURM_SUBMIT_DIR"
python scripts/aggregate_event_trips.py --threads "$SLURM_CPUS_PER_TASK" --memory-limit 8GB "$@"
