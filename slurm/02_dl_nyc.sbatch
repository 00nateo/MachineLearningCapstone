#!/bin/bash
#SBATCH --job-name=dl_nyc
#SBATCH --partition=normal_q
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export OUT_DIR="/scratch/$USER/raw/NYC-Ride-Data"
TYPES=fhvhv ./download_tlc_data.sh 2019 "$(date +%Y)"
