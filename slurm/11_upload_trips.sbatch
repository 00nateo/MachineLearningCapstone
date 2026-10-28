#!/bin/bash
#SBATCH --job-name=upload_trips
#SBATCH --partition=normal_q
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out

# Trip-level parquet (~65 GB) -> private HF dataset. hf upload splits large folders into several
# commits; if the job dies, resubmit it and it resumes. The dataset card is uploaded separately.
set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate rideshare
cd "$SLURM_SUBMIT_DIR"
: "${HF_TRIPS_REPO:=MLCapstoneRideshare/rideshare-trips}"

for d in chicago_trips nyc_fhvhv; do
  hf upload "$HF_TRIPS_REPO" "data/processed/$d" "$d" --repo-type dataset --private \
    --include "*.parquet" --commit-message "Add $d"
done
