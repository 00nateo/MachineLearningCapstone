#!/usr/bin/env python
"""Coordinates for every event venue, geocoded with OpenStreetMap Nominatim.

Source: https://nominatim.openstreetmap.org (usage policy: <= 1 request/s, identifying
        User-Agent, results cached). Neither ESPN nor setlist.fm gives venue coordinates.
Input:  data/external/events/sports.parquet, concert_venues.parquet
Output: data/external/events/venues.parquet
        venue_name, city, state, lat, lon, osm_name, query  (lat/lon NULL if not found)
Venues already geocoded are reused; only new ones are looked up. OVERRIDES pins venues
Nominatim can't find; unmatched venues are printed so they can be added there.
"""
import json
import sys
import time
from http.client import HTTPException
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "external" / "events"
URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "rideshare-capstone (VT ARC research)"}
OVERRIDES = {  # (venue_name, city, state) -> (lat, lon): venues Nominatim can't find
}


def geocode(q, tries=4):
    for attempt in range(1, tries + 1):
        time.sleep(1.1)  # Nominatim policy: at most 1 request per second
        try:
            with urlopen(Request(f"{URL}?{urlencode({'q': q, 'format': 'jsonv2', 'limit': 1, 'countrycodes': 'us'})}",
                                 headers=HEADERS), timeout=60) as r:
                hits = json.load(r)
            return hits[0] if hits else None
        except (OSError, HTTPException, ValueError) as e:
            print(f"  retry {attempt}/{tries}: {e}", file=sys.stderr, flush=True)
            time.sleep(10 * attempt)
    raise RuntimeError(f"giving up on {q}")


def main():
    sports = pd.read_parquet(EVENTS / "sports.parquet")[["venue_name", "city", "state"]]
    concerts = pd.read_parquet(EVENTS / "concert_venues.parquet").rename(columns={"name": "venue_name"})
    venues = pd.concat([sports, concerts[["venue_name", "city", "state"]]]).dropna(subset=["venue_name"]).drop_duplicates()
    out = EVENTS / "venues.parquet"
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["venue_name", "city", "state"])
    known = set(zip(done.venue_name, done.city, done.state))
    rows = []
    for v in venues.itertuples(index=False):
        key = (v.venue_name, v.city, v.state)
        if key in known and key not in OVERRIDES:
            continue
        if key in OVERRIDES:
            lat, lon = OVERRIDES[key]
            rows.append(dict(zip(("venue_name", "city", "state"), key), lat=lat, lon=lon, osm_name=None, query="override"))
            continue
        hit, used = None, None
        for q in (f"{v.venue_name}, {v.city}, {v.state}", f"{v.venue_name}, {v.state}"):
            hit, used = geocode(q), q
            if hit:
                break
        rows.append(dict(zip(("venue_name", "city", "state"), key), lat=float(hit["lat"]) if hit else None,
                         lon=float(hit["lon"]) if hit else None, osm_name=hit["display_name"] if hit else None, query=used))
        print(f"{'OK  ' if hit else 'MISS'} {v.venue_name} ({v.city}, {v.state})"
              + (f" -> {hit['display_name'][:80]}" if hit else ""), flush=True)
    new = pd.DataFrame(rows)
    keep = done[~done.set_index(["venue_name", "city", "state"]).index.isin(list(OVERRIDES))] if len(done) else done
    allv = pd.concat([keep, new]).drop_duplicates(["venue_name", "city", "state"], keep="last")
    part = out.with_name(out.name + ".part")
    allv.to_parquet(part, compression="zstd", index=False)
    part.replace(out)
    missing = allv[allv.lat.isna()]
    print(f"wrote {out}: {len(allv)} venues, {len(missing)} without coordinates")
    if len(missing):
        print("add these to OVERRIDES:\n" + missing[["venue_name", "city", "state"]].to_string(index=False))


if __name__ == "__main__":
    main()
