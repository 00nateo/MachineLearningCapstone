#!/usr/bin/env python
"""Coordinates for every event venue, geocoded with OpenStreetMap Nominatim.

Source: https://nominatim.openstreetmap.org (usage policy: <= 1 request/s, identifying
        User-Agent, results cached). Neither ESPN nor setlist.fm gives venue coordinates.
Input:  data/external/events/sports.parquet, concert_venues.parquet
Output: data/external/events/venues.parquet
        venue_name, city, state, lat, lon, osm_name, query  (lat/lon NULL if not found)
Venues already geocoded are reused; new ones and ones still without coordinates are looked up.
QUERY_FIXES geocodes sub-venues at their parent (e.g. "Hulu Theater at Madison Square Garden")
and small college gyms at their campus. OVERRIDES pins exact coordinates; venues still
unmatched are printed so they can be added there.
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
QUERY_FIXES = [  # (substring of the venue name, query to geocode instead); first match wins
    ("Madison Square Garden", "Madison Square Garden, New York, NY"),
    ("Barclays Center", "Barclays Center, Brooklyn, NY"),
    ("Pier 17", "Pier 17, New York, NY"),
    ("Randall's Island", "Randalls Island, New York, NY"),
    ("Prudential Center", "Prudential Center, Newark, NJ"),
    ("United Center", "United Center, Chicago, IL"),
    ("Soldier Field", "Soldier Field, Chicago, IL"),
    ("Northerly Island", "Northerly Island, Chicago, IL"),
    ("Chicago Theatre", "Chicago Theatre, Chicago, IL"),
    ("Grant Park", "Grant Park, Chicago, IL"),
    ("Amphitheatre", "Tinley Park, IL"),  # Credit Union 1 / Hollywood Casino Amphitheatre
    ("Carnesecca Arena", "St. John's University, Queens, NY"),
    ("Rose Hill Gym", "Fordham University, Bronx, NY"),
    ("Jack Coffey Field", "Fordham University, Bronx, NY"),
    ("Wagner College", "Wagner College, Staten Island, NY"),
    ("Brooklyn Athletic Center", "Long Island University Brooklyn"),
    ("Pope Physical Education Center", "St. Francis College, Brooklyn, NY"),
    ("Gentile Arena", "Loyola University Chicago"),
    ("Levien Gymnasium", "Columbia University, New York, NY"),
    ("David S. Mack", "Hofstra University, Hempstead, NY"),
    ("Fleisher Center", "New Jersey Institute of Technology, Newark, NJ"),
]


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
    found = done[done["lat"].notna()] if len(done) else done  # retry venues still without coordinates
    known = set(zip(found.venue_name, found.city, found.state))
    rows = []
    for v in venues.itertuples(index=False):
        key = (v.venue_name, v.city, v.state)
        if key in known and key not in OVERRIDES:
            continue
        if key in OVERRIDES:
            lat, lon = OVERRIDES[key]
            rows.append(dict(zip(("venue_name", "city", "state"), key), lat=lat, lon=lon, osm_name=None, query="override"))
            continue
        fix = next((q for s, q in QUERY_FIXES if s.lower() in v.venue_name.lower()), None)
        hit, used = None, None
        for q in ([fix] if fix else [f"{v.venue_name}, {v.city}, {v.state}", f"{v.venue_name}, {v.state}"]):
            hit, used = geocode(q), q
            if hit:
                break
        rows.append(dict(zip(("venue_name", "city", "state"), key), lat=float(hit["lat"]) if hit else None,
                         lon=float(hit["lon"]) if hit else None, osm_name=hit["display_name"] if hit else None, query=used))
        print(f"{'OK  ' if hit else 'MISS'} {v.venue_name} ({v.city}, {v.state})"
              + (f" -> {hit['display_name'][:80]}" if hit else ""), flush=True)
    new = pd.DataFrame(rows)
    redo = set(OVERRIDES) | set(zip(new.venue_name, new.city, new.state)) if len(new) else set(OVERRIDES)
    keep = done[~done.set_index(["venue_name", "city", "state"]).index.isin(list(redo))] if len(done) else done
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
