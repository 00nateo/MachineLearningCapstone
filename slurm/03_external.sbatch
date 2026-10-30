#!/bin/bash
#SBATCH --job-name=external
#SBATCH --partition=normal_q
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
eval "$(conda shell.bash hook)"
conda activate rideshare

SCRIPTS="$SLURM_SUBMIT_DIR/scripts"
mkdir -p "$SLURM_SUBMIT_DIR/data/external"
cd "$SLURM_SUBMIT_DIR/data/external"

fail=0
for s in fetch_geo.sh fetch_weather.py fetch_nws.py fetch_bts.py fetch_mta.py fetch_cta.py \
         fetch_sports.py fetch_concerts.py fetch_venues.py; do
  echo "=== $s  $(date) ==="
  case "$s" in
    *.sh) bash "$SCRIPTS/$s" ;;
    *.py) python "$SCRIPTS/$s" ;;
  esac || { echo "FAILED: $s"; fail=1; }
done
exit $fail
