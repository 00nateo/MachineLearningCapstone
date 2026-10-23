#!/usr/bin/env bash
# Zone geometries for both cities.
# Output (data/external/geo/):
#   taxi_zones.zip                    NYC TLC taxi zones shapefile (geopandas reads the zip directly)
#   taxi_zone_lookup.csv              LocationID -> borough, zone, service_zone
#   chicago_community_areas.geojson   77 community areas (or .csv with WKT geometry as fallback)
# Each file has candidate URLs; the first whose HEAD check returns 200 and downloads non-empty
# is used. A failed fetch doesn't stop the others; the script exits non-zero at the end.
# Re-running is safe: existing files are skipped.
set -uo pipefail

OUT="$(cd "$(dirname "$0")/.." && pwd)/data/external/geo"
TLC="https://d37ci6vzurychx.cloudfront.net/misc"
CHI="https://data.cityofchicago.org"
mkdir -p "$OUT"
fail=0

# ok200 <url>: true if a HEAD request (following redirects) ends in HTTP 200.
ok200() {
  [[ "$(curl -sIL --max-time 60 -o /dev/null -w '%{http_code}' "$1")" == 200 ]]
}

# fetch_first <label> <url> <dest> [<url> <dest> ...]
fetch_first() {
  local label=$1; shift
  while (( $# >= 2 )); do
    local url=$1 dest=$2; shift 2
    if [[ -s "$dest" ]]; then echo "exists: $dest"; return 0; fi
    if ! ok200 "$url"; then echo "  HEAD not 200: $url" >&2; continue; fi
    if curl -fL --retry 3 --retry-delay 5 --max-time 600 -o "$dest.part" "$url" && [[ -s "$dest.part" ]]; then
      mv "$dest.part" "$dest"
      echo "downloaded: $dest  <- $url"
      return 0
    fi
    rm -f "$dest.part"
    echo "  download failed or empty: $url" >&2
  done
  echo "FAILED: $label" >&2
  return 1
}

fetch_first "taxi zones" \
  "$TLC/taxi_zones.zip" "$OUT/taxi_zones.zip" || fail=1
fetch_first "taxi zone lookup" \
  "$TLC/taxi_zone_lookup.csv" "$OUT/taxi_zone_lookup.csv" \
  "$TLC/taxi+_zone_lookup.csv" "$OUT/taxi_zone_lookup.csv" || fail=1
fetch_first "chicago community areas" \
  "$CHI/api/geospatial/igwz-8jzy?method=export&format=GeoJSON" "$OUT/chicago_community_areas.geojson" \
  "$CHI/api/views/igwz-8jzy/rows.csv?accessType=DOWNLOAD" "$OUT/chicago_community_areas.csv" || fail=1

exit $fail
