#!/usr/bin/env bash
# Download NYC TLC trip record parquet files (2018 -> present).
# Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
#
# Output: NYC-Ride-Data/<type>/<type>_tripdata_<YYYY-MM>.parquet
#
# Usage: ./download_tlc_data.sh [start_year] [end_year]
#   TYPES="yellow green" ./download_tlc_data.sh 2018 2024
#
# Re-running is safe: files that already exist with the correct size are skipped.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/NYC-Ride-Data}"
START_YEAR="${1:-2018}"
END_YEAR="${2:-$(date +%Y)}"
TYPES="${TYPES:-yellow green fhv fhvhv}"
BASE_URL="https://d37ci6vzurychx.cloudfront.net/trip-data"

downloaded=0; skipped=0; missing=0; failed=0

for type in $TYPES; do
  mkdir -p "$OUT_DIR/$type"
  for year in $(seq "$START_YEAR" "$END_YEAR"); do
    for month in $(seq -w 1 12); do
      file="${type}_tripdata_${year}-${month}.parquet"
      url="$BASE_URL/$file"
      dest="$OUT_DIR/$type/$file"

      # Look up remote size; non-200 means the month isn't published (or doesn't exist).
      headers=$(curl -sIL "$url")
      status=$(printf '%s' "$headers" | awk 'toupper($1) ~ /^HTTP/ {code=$2} END {print code}')
      if [[ "$status" != "200" ]]; then
        missing=$((missing + 1))
        continue
      fi
      remote_size=$(printf '%s' "$headers" | awk 'tolower($1)=="content-length:" {gsub("\r",""); size=$2} END {print size}')

      if [[ -f "$dest" ]]; then
        local_size=$(wc -c < "$dest" | tr -d ' ')
        if [[ "$local_size" == "$remote_size" ]]; then
          skipped=$((skipped + 1))
          continue
        fi
      fi

      echo "Downloading $file ($((remote_size / 1024 / 1024)) MB)"
      if curl -fL --retry 3 --retry-delay 5 -o "$dest.part" "$url"; then
        mv "$dest.part" "$dest"
        downloaded=$((downloaded + 1))
      else
        echo "  FAILED: $url" >&2
        rm -f "$dest.part"
        failed=$((failed + 1))
      fi
    done
  done
done

echo
echo "Done. downloaded=$downloaded skipped=$skipped not_published=$missing failed=$failed"
[[ $failed -eq 0 ]]
