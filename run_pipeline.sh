#!/usr/bin/env bash
# Usage:
#   ACCOUNT=myalloc HF_REPO=org/rideshare-hotspots ./run_pipeline.sh            # everything
#   ACCOUNT=myalloc HF_REPO=... ./run_pipeline.sh process    # convert -> aggregate -> export -> upload
#   ACCOUNT=myalloc ./run_pipeline.sh external               # external fetches only
#   ACCOUNT=myalloc HF_REPO=... ./run_pipeline.sh export     # export -> upload (republish only)
#   ACCOUNT=myalloc ./run_pipeline.sh features               # label.py + features.py from data/export
set -euo pipefail
cd "$(dirname "$0")"

ACCOUNT="${ACCOUNT:?set ACCOUNT=<your slurm allocation>}"
STAGE="${1:-all}"
export SOCRATA_APP_TOKEN="$(cat ~/.socrata_token 2>/dev/null || true)"
mkdir -p logs

sub() { sbatch --parsable --account="$ACCOUNT" --kill-on-invalid-dep=yes "$@"; }
need_hf() { export HF_REPO="${HF_REPO:?set HF_REPO=<org>/<dataset> to upload}"; }

case "$STAGE" in
  all)
    need_hf
    chi=$(sub slurm/01_dl_chicago.sbatch)
    nyc=$(sub slurm/02_dl_nyc.sbatch)
    ext=$(sub slurm/03_external.sbatch)
    conv=$(sub --dependency=afterok:$chi:$nyc slurm/04_convert.sbatch)
    agg=$(sub --dependency=afterok:$conv slurm/05_aggregate.sbatch)
    exp=$(sub --dependency=afterok:$agg:$ext slurm/06_export.sbatch)
    up=$(sub --dependency=afterok:$exp slurm/07_upload.sbatch)
    echo "dl_chicago=$chi dl_nyc=$nyc external=$ext convert=$conv aggregate=$agg export=$exp upload=$up"
    ;;
  process)
    need_hf
    conv=$(sub slurm/04_convert.sbatch)
    agg=$(sub --dependency=afterok:$conv slurm/05_aggregate.sbatch)
    exp=$(sub --dependency=afterok:$agg slurm/06_export.sbatch)
    up=$(sub --dependency=afterok:$exp slurm/07_upload.sbatch)
    echo "convert=$conv aggregate=$agg export=$exp upload=$up"
    ;;
  external)
    ext=$(sub slurm/03_external.sbatch)
    echo "external=$ext"
    ;;
  export)
    need_hf
    exp=$(sub slurm/06_export.sbatch)
    up=$(sub --dependency=afterok:$exp slurm/07_upload.sbatch)
    echo "export=$exp upload=$up"
    ;;
  features)
    feat=$(sub slurm/08_features.sbatch)
    echo "features=$feat"
    ;;
  *) echo "unknown stage: $STAGE" >&2; exit 1 ;;
esac
