#!/bin/bash
#SBATCH --job-name=dl_chicago
#SBATCH --partition=normal_q
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export OUT_DIR="/scratch/$USER/raw/Chicago-Ride-Data"
export SOCRATA_APP_TOKEN="${SOCRATA_APP_TOKEN:-$(cat ~/.socrata_token 2>/dev/null || true)}"
JOBS=24 ./download_chicago_data.sh 2018 "$(date +%Y)"
