#!/usr/bin/env python
"""Concerts and festivals at major Chicago- and NYC-area venues from the setlist.fm API.

Source: https://api.setlist.fm/docs/1.0/index.html (free for non-commercial use)
        The key is read from ~/.setlistfm_key and sent as x-api-key; it is never printed.
Output: data/external/events/concert_venues.parquet  venue_key, setlistfm_id, name, city, state
        data/external/events/concerts.parquet        venue_key, setlistfm_id, date, artist, tour, setlist_id
setlist.fm is crowd-sourced: coverage favours well-known acts. Setlists carry a date but no
start time (treat as an evening show), and one row per act (openers included), so count
distinct (venue_key, date) for shows. One venue can have several setlist.fm ids (the main
Madison Square Garden id stops in 2022), so ids are collected by name search and printed for
review. Re-fetched in full on every run.
"""
import json
import sys
import time
from datetime import date, datetime
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "events"
BASE = "https://api.setlist.fm/rest/1.0"
KEY = (Path.home() / ".setlistfm_key").read_text().strip()
START = date(2018, 11, 1)
PAUSE = 0.6  # seconds between requests; no published rate limit
VENUES = {  # venue_key -> (state, search names incl. former names)
    # NYC, inside the taxi zones
    "msg": ("NY", ["Madison Square Garden"]),
    "barclays": ("NY", ["Barclays Center"]),
    "radio-city": ("NY", ["Radio City Music Hall"]),
    "beacon": ("NY", ["Beacon Theatre"]),
    "terminal-5": ("NY", ["Terminal 5"]),
    "hammerstein": ("NY", ["Hammerstein Ballroom"]),
    "pier-17": ("NY", ["Pier 17"]),
    "summerstage": ("NY", ["SummerStage"]),
    "brooklyn-steel": ("NY", ["Brooklyn Steel"]),
    "kings-theatre": ("NY", ["Kings Theatre"]),
    "forest-hills": ("NY", ["Forest Hills Stadium"]),
    "yankee-stadium": ("NY", ["Yankee Stadium"]),
    "citi-field": ("NY", ["Citi Field"]),
    "randalls-island": ("NY", ["Randall's Island", "Randalls Island"]),
    "flushing-meadows": ("NY", ["Flushing Meadows"]),
    # NYC area, outside the zones
    "ubs-arena": ("NY", ["UBS Arena"]),
    "metlife": ("NJ", ["MetLife Stadium"]),
    "prudential": ("NJ", ["Prudential Center"]),
    # Chicago, inside the community areas
    "united-center": ("IL", ["United Center"]),
    "wrigley": ("IL", ["Wrigley Field"]),
    "soldier-field": ("IL", ["Soldier Field"]),
    "northerly-island": ("IL", ["Northerly Island", "Huntington Bank Pavilion", "FirstMerit Bank Pavilion"]),
    "chicago-theatre": ("IL", ["Chicago Theatre"]),
    "aragon": ("IL", ["Aragon Ballroom"]),
    "riviera": ("IL", ["Riviera Theatre"]),
    "salt-shed": ("IL", ["Salt Shed"]),
    "auditorium": ("IL", ["Auditorium Theatre"]),
    "grant-park": ("IL", ["Grant Park"]),
    "union-park": ("IL", ["Union Park"]),
    "douglass-park": ("IL", ["Douglass Park"]),
    "wintrust": ("IL", ["Wintrust Arena"]),
    # Chicago area, outside the community areas
    "allstate-arena": ("IL", ["Allstate Arena"]),
    "tinley-park": ("IL", ["Credit Union 1 Amphitheatre", "Hollywood Casino Amphitheatre"]),
}


def get(path, params, tries=6):
    url = f"{BASE}/{path}?{urlencode(params)}"
    headers = {"x-api-key": KEY, "Accept": "application/json", "User-Agent": "rideshare-capstone"}
    for attempt in range(1, tries + 1):
        time.sleep(PAUSE)
        try:
            with urlopen(Request(url, headers=headers), timeout=60) as r:
                return json.load(r)
        except HTTPError as e:
            if e.code == 404:  # setlist.fm answers "nothing found" / past the last page with 404
                return {}
            wait = 60 if e.code == 429 else 10 * attempt
            print(f"  {path} HTTP {e.code}, retry {attempt}/{tries} in {wait}s", file=sys.stderr, flush=True)
        except (OSError, HTTPException, ValueError) as e:
            wait = 10 * attempt
            print(f"  {path} {e}, retry {attempt}/{tries}", file=sys.stderr, flush=True)
        time.sleep(wait)
    raise RuntimeError(f"giving up on {path}")


def venue_ids(key, state, names):
    """setlist.fm venues whose name contains one of `names`, in `state`, US only."""
    found = {}
    for name in names:
        page = 1
        while True:
            res = get("search/venues", {"name": name, "p": page})
            for v in res.get("venue", []):
                city = v.get("city") or {}
                if (city.get("stateCode") == state and (city.get("country") or {}).get("code") == "US"
                        and name.lower() in v.get("name", "").lower()):
                    found[v["id"]] = {"venue_key": key, "setlistfm_id": v["id"], "name": v["name"],
                                      "city": city.get("name"), "state": state}
            if page * res.get("itemsPerPage", 20) >= res.get("total", 0):
                break
            page += 1
    return list(found.values())


def venue_setlists(venue):
    """All setlists at one venue id since START (the API returns newest first)."""
    rows, page = [], 1
    while True:
        res = get(f"venue/{venue['setlistfm_id']}/setlists", {"p": page})
        batch = res.get("setlist", [])
        for s in batch:
            day = datetime.strptime(s["eventDate"], "%d-%m-%Y").date()
            if day >= START:
                rows.append({"venue_key": venue["venue_key"], "setlistfm_id": venue["setlistfm_id"], "date": day,
                             "artist": (s.get("artist") or {}).get("name"), "tour": (s.get("tour") or {}).get("name"),
                             "setlist_id": s.get("id")})
        if not batch or datetime.strptime(batch[-1]["eventDate"], "%d-%m-%Y").date() < START \
                or page * res.get("itemsPerPage", 20) >= res.get("total", 0):
            return rows
        page += 1


def write(df, name):
    dest = OUT / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    df.to_parquet(part, compression="zstd", index=False)
    part.replace(dest)
    print(f"wrote {dest} ({len(df):,} rows)", flush=True)


def main():
    venues = [v for key, (state, names) in VENUES.items() for v in venue_ids(key, state, names)]
    vdf = pd.DataFrame(venues)
    print("matched venue ids (review these):\n" + vdf.sort_values(["venue_key", "name"]).to_string(index=False), flush=True)
    for key in sorted(set(VENUES) - set(vdf["venue_key"])):
        print(f"WARNING {key}: no setlist.fm venue matched", flush=True)
    write(vdf, "concert_venues.parquet")
    rows = []
    for v in venues:
        got = venue_setlists(v)
        print(f"{v['venue_key']:18s} {v['setlistfm_id']} {v['name'][:40]:40s} {len(got):5d} setlists", flush=True)
        rows += got
    concerts = pd.DataFrame(rows).drop_duplicates("setlist_id")
    write(concerts, "concerts.parquet")
    shows = concerts.groupby("venue_key")["date"].nunique().sort_values(ascending=False)
    print("show nights per venue:\n" + shows.to_string())


if __name__ == "__main__":
    main()
