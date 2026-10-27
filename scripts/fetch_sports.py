#!/usr/bin/env python
"""Sports events at Chicago- and NYC-area venues from ESPN's public scoreboard API.

Source:  https://site.api.espn.com/apis/site/v2/sports/<sport>/<league>/scoreboard?dates=YYYYMMDD
         (unofficial and undocumented: no key, may change without notice)
Leagues: NFL, MLB, NBA, WNBA, NHL, MLS, NCAA football (FBS + FCS), NCAA men's and women's
         basketball (all Division I), and US Open tennis (ATP + WTA, every match).
Raw:     data/external/events/espn/<league>/<YYYY-MM>.parquet  (one file per league-month)
Output:  data/external/events/sports.parquet
         league, event_id, start_utc, venue_id, venue_name, city, state, neutral_site,
         attendance, home, away, detail, status
Only events at venues in METRO cities (in and around both cities) are kept. Mapping venues
to zones, including venues outside the zones, happens in export.py.
The US Open 2020 was played without spectators (attendance set to 0).
Past months are skipped once fetched; the last two months are refreshed.
Usage: python scripts/fetch_sports.py [--refresh LEAGUE ...]   (re-fetch every month of those leagues)
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from http.client import HTTPException
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "events"
BASE = "https://site.api.espn.com/apis/site/v2/sports"
START = date(2018, 11, 1)
WORKERS = 4
LEAGUES = {  # name -> (path, extra query params)
    "nfl": ("football/nfl", {}),
    "mlb": ("baseball/mlb", {}),
    "nba": ("basketball/nba", {}),
    "wnba": ("basketball/wnba", {}),
    "nhl": ("hockey/nhl", {}),
    "mls": ("soccer/usa.1", {}),
    "ncaaf-fbs": ("football/college-football", {"groups": 80}),
    "ncaaf-fcs": ("football/college-football", {"groups": 81}),
    "ncaam": ("basketball/mens-college-basketball", {"groups": 50}),
    "ncaaw": ("basketball/womens-college-basketball", {"groups": 50}),
}
METRO = {  # state -> venue cities kept (in or near the two cities)
    "NY": {"New York", "Brooklyn", "Bronx", "Queens", "Flushing", "Jamaica", "Staten Island",
           "Riverdale", "Elmont", "Uniondale", "Hempstead"},
    "NJ": {"East Rutherford", "Harrison", "Newark"},
    "IL": {"Chicago", "Evanston", "Rosemont", "Bridgeview"},
}
STATE_ABBR = {"New York": "NY", "New Jersey": "NJ", "Illinois": "IL"}  # MLB/MLS spell states out
CITY_STATE = {"New York City": ("New York", "NY")}  # MLS venues given with no state at all
COLS = ["league", "event_id", "start_utc", "venue_id", "venue_name", "city", "state",
        "neutral_site", "attendance", "home", "away", "detail", "status"]


def get(path, params, tries=5, timeout=120):
    url = f"{BASE}/{path}/scoreboard?{urlencode(params)}"
    for attempt in range(1, tries + 1):
        try:
            with urlopen(Request(url, headers={"User-Agent": "rideshare-capstone"}), timeout=timeout) as r:
                return json.load(r)
        except (OSError, HTTPException, ValueError) as e:  # incl. timeouts, resets, bad JSON
            print(f"  retry {attempt}/{tries}: {e}", file=sys.stderr, flush=True)
            time.sleep(10 * attempt)
    raise RuntimeError(f"giving up on {url}")


def day_rows(league, day):
    path, extra = LEAGUES[league]
    rows = []
    for e in get(path, {"dates": f"{day:%Y%m%d}", "limit": 500, **extra}).get("events", []):
        for c in e.get("competitions", []):
            v = c.get("venue") or {}
            a = v.get("address") or {}
            city, state = a.get("city"), a.get("state")
            if not state and city in CITY_STATE:
                city, state = CITY_STATE[city]
            elif not state and city and ", " in city:  # MLS: "Chicago, Illinois", no state field
                city, state = city.rsplit(", ", 1)
            state = STATE_ABBR.get(state, state)
            if city not in METRO.get(state, ()):
                continue
            teams = {t.get("homeAway"): (t.get("team") or {}).get("displayName") for t in c.get("competitors", [])}
            rows.append({"league": league, "event_id": e.get("id"), "start_utc": c.get("date") or e.get("date"),
                         "venue_id": v.get("id"), "venue_name": v.get("fullName"), "city": city,
                         "state": state, "neutral_site": c.get("neutralSite"),
                         "attendance": c.get("attendance"), "home": teams.get("home"), "away": teams.get("away"),
                         "detail": None, "status": ((c.get("status") or {}).get("type") or {}).get("name")})
    return rows


def frame(rows):
    df = pd.DataFrame(rows, columns=COLS).drop_duplicates(["league", "event_id"])
    df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True).dt.tz_localize(None)
    return df


def write(df, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    df.to_parquet(part, compression="zstd", index=False)
    part.replace(dest)


def fetch_month(league, first):
    days = [d for d in (first + timedelta(i) for i in range(31)) if d.month == first.month and d <= date.today()]
    with ThreadPoolExecutor(WORKERS) as pool:
        rows = [r for day in pool.map(lambda d: day_rows(league, d), days) for r in day]
    write(frame(rows), OUT / "espn" / league / f"{first:%Y-%m}.parquet")
    return len(rows)


def us_open(year):
    """Every US Open match (ATP + WTA draws) in a given year, venue fixed."""
    rows = []
    for tour in ("atp", "wta"):
        for e in get(f"tennis/{tour}", {"dates": f"{year}0901"}).get("events", []):
            if e.get("name") != "US Open":
                continue
            for g in e.get("groupings", []):
                for c in g.get("competitions", []):
                    rows.append({"league": "us-open", "event_id": c.get("id"), "start_utc": c.get("date"),
                                 "venue_id": None, "venue_name": "USTA Billie Jean King National Tennis Center",
                                 "city": "Queens", "state": "NY", "neutral_site": None,
                                 "attendance": 0 if year == 2020 else None, "home": None, "away": None,
                                 "detail": (g.get("grouping") or {}).get("displayName"),
                                 "status": ((c.get("status") or {}).get("type") or {}).get("name")})
    if not rows:
        print(f"us-open {year}: WARNING no 'US Open' event found", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", nargs="*", default=[], choices=list(LEAGUES))
    refresh = set(ap.parse_args().refresh)
    today = date.today()
    refresh_after = (today.replace(day=1) - timedelta(days=62)).replace(day=1)
    for league in LEAGUES:
        first = START
        while first <= today:
            dest = OUT / "espn" / league / f"{first:%Y-%m}.parquet"
            if league in refresh or not (dest.exists() and first < refresh_after):
                print(f"{league} {first:%Y-%m}: {fetch_month(league, first)} local events", flush=True)
            first = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
    tennis = [r for y in range(START.year, today.year + 1) if date(y, 8, 15) <= today for r in us_open(y)]
    sports = pd.concat([pd.read_parquet(p) for p in sorted((OUT / "espn").glob("*/*.parquet"))] + [frame(tennis)])
    write(sports, OUT / "sports.parquet")
    print(f"wrote {OUT / 'sports.parquet'} ({len(sports):,} events)")
    print(sports.groupby("league").size().to_string())
    print("top venues:\n" + sports.groupby(["venue_name", "city"]).size().nlargest(30).to_string())


if __name__ == "__main__":
    main()
