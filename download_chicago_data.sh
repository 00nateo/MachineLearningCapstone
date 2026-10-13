#!/usr/bin/env bash
# Download Chicago Transportation Network Providers (Uber/Lyft/Via) trip records (2018 -> present).
# Source: https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips-2026-/8bbv-3wbe
#
# The portal only serves CSV (no parquet), and large/sorted queries time out, so each day is
# fetched in 6-hour chunks and each chunk is verified against the server's row count.
# Output: Chicago-Ride-Data/<year>/tnp_trips_<YYYY-MM-DD>.csv (one file per day)
#
# Usage: ./download_chicago_data.sh [start_year] [end_year]
#   JOBS=8 SOCRATA_APP_TOKEN=xxxx ./download_chicago_data.sh 2023 2024
#
# A free app token (https://data.cityofchicago.org/profile/edit/developer_settings) raises
# the API rate limits. Re-running is safe: completed days are skipped.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
START_YEAR="${1:-2018}"
END_YEAR="${2:-$(date +%Y)}"
export OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/Chicago-Ride-Data}"
export SOCRATA_APP_TOKEN="${SOCRATA_APP_TOKEN:-}"
JOBS="${JOBS:-4}"

# One dataset per year on the portal.
dataset_for_year() {
  case "$1" in
    2018) echo rpfj-eb3a ;;  2019) echo iu3g-qa69 ;;  2020) echo rmc8-eqv4 ;;
    2021) echo unf9-2zu4 ;;  2022) echo 2tdj-ffvb ;;  2023) echo ukxa-jdd7 ;;
    2024) echo aesv-xzh6 ;;  2025) echo 3q84-vs9b ;;  2026) echo 8bbv-3wbe ;;
    *) return 1 ;;
  esac
}

days_in_month() {
  local y=$1 m=$((10#$2))
  case $m in
    2) if (( (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 )); then echo 29; else echo 28; fi ;;
    4|6|9|11) echo 30 ;;
    *) echo 31 ;;
  esac
}

# soda <dataset> <query params...> : GET a SODA CSV endpoint, writing the body to stdout.
soda() {
  local dataset=$1; shift
  local args=(-sfG --max-time 600 "https://data.cityofchicago.org/resource/$dataset.csv")
  [[ -n "$SOCRATA_APP_TOKEN" ]] && args+=(-H "X-App-Token: $SOCRATA_APP_TOKEN")
  for p in "$@"; do args+=(--data-urlencode "$p"); done
  curl "${args[@]}"
}

# download_day <dataset> <YYYY-MM-DD>
download_day() {
  local dataset=$1 day=$2
  local dest="$OUT_DIR/${day:0:4}/tnp_trips_$day.csv"
  [[ -f "$dest" ]] && return 0

  local tmp_dir; tmp_dir=$(mktemp -d)
  local bounds=("T00:00:00" "T06:00:00" "T12:00:00" "T18:00:00" "T23:59:59.999")
  local total=0 i

  for i in 0 1 2 3; do
    local where="trip_start_timestamp >= '$day${bounds[$i]}' AND trip_start_timestamp < '$day${bounds[$((i + 1))]}'"
    [[ $i -eq 3 ]] && where="${where/</<=}"   # include the final 23:59:59.999 boundary

    local expected="" attempt
    for attempt in 1 2 3 4 5; do
      expected=$(soda "$dataset" "\$select=count(*)" "\$where=$where" | tail -1 | tr -d '"\r')
      [[ "$expected" =~ ^[0-9]+$ ]] && break
      sleep $((attempt * 10))
    done
    if ! [[ "$expected" =~ ^[0-9]+$ ]]; then
      echo "  $day chunk $i: could not get row count" >&2; rm -rf "$tmp_dir"; return 1
    fi
    total=$((total + expected))

    local ok=0
    for attempt in 1 2 3 4 5; do
      if soda "$dataset" "\$where=$where" "\$limit=10000000" > "$tmp_dir/$i.csv"; then
        local got=$(( $(wc -l < "$tmp_dir/$i.csv") - 1 ))
        [[ $got -eq $expected ]] && { ok=1; break; }
        echo "  $day chunk $i: got $got/$expected rows, retrying ($attempt)" >&2
      else
        echo "  $day chunk $i: request failed, retrying ($attempt)" >&2
      fi
      sleep $((attempt * 10))
    done
    if [[ $ok -eq 0 ]]; then
      echo "  FAILED: $day" >&2; rm -rf "$tmp_dir"; return 1
    fi
  done

  # Nothing published for this day yet -> leave no file so a later run picks it up.
  if [[ $total -eq 0 ]]; then rm -rf "$tmp_dir"; return 0; fi

  mkdir -p "$(dirname "$dest")"
  { head -1 "$tmp_dir/0.csv"; for i in 0 1 2 3; do tail -n +2 "$tmp_dir/$i.csv"; done; } > "$dest.part.$$"
  mv "$dest.part.$$" "$dest"
  rm -rf "$tmp_dir"
  echo "Downloaded $day ($total trips)"
}
export -f soda download_day

today=$(date +%Y-%m-%d)
for year in $(seq "$START_YEAR" "$END_YEAR"); do
  if ! dataset=$(dataset_for_year "$year"); then
    echo "No dataset id known for $year - add it to dataset_for_year()" >&2
    continue
  fi
  for month in $(seq -w 1 12); do
    for d in $(seq -w 1 "$(days_in_month "$year" "$month")"); do
      day="$year-$month-$d"
      [[ "$day" < "$today" ]] && echo "$dataset $day"
    done
  done
done | xargs -P "$JOBS" -n 2 bash -c 'download_day "$0" "$1"'

failed=$?
echo
echo "Done. $(find "$OUT_DIR" -name '*.csv' | wc -l | tr -d ' ') day files in $OUT_DIR"
[[ $failed -eq 0 ]] || { echo "Some days failed - re-run to retry them." >&2; exit 1; }
