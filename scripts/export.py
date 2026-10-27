#!/usr/bin/env python
"""Build data/export/: the minimal model-ready dataset published to Hugging Face.

Every table is at the grain the model uses (zone x hour, city/airport x hour, zone x day) or is
small static reference data; trip-level data stays on the cluster in data/processed/.
Times are naive local wall-clock unless the column name ends in _utc. Tables are indexed by when
things happened; the >= 1h feature lag is applied at feature-building time, not here.

Built in data/export.tmp/ and swapped in only if every table succeeds, so a partial export is
never uploaded.

Usage: python scripts/export.py
"""
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EXT = ROOT / "data" / "external"
EXPORT = ROOT / "data" / "export"
STAGE = ROOT / "data" / "export.tmp"

AIRPORT_CITY = {"ORD": "chicago", "MDW": "chicago", "JFK": "nyc", "LGA": "nyc", "EWR": "nyc"}
ALERT_AGENCIES = ("NYCT Subway", "NYCT Bus", "LIRR", "MNR", "BT")
UTM = {"nyc": "EPSG:32618", "chicago": "EPSG:32616"}  # metres, for area and centroids
ADJ_TOLERANCE_M = 20  # polygons this close count as neighbours (bridges digitising gaps)
METRO_OF_STATE = {"NY": "nyc", "NJ": "nyc", "IL": "chicago"}
METRO_TZ = {"nyc": "America/New_York", "chicago": "America/Chicago"}
# Typical event length in hours, for an estimated end time (ESPN gives start times only).
EVENT_HOURS = {"nfl": 3.25, "ncaaf-fbs": 3.5, "ncaaf-fcs": 3.25, "mlb": 3.0, "nba": 2.5, "wnba": 2.0,
               "nhl": 2.5, "mls": 2.0, "ncaam": 2.0, "ncaaw": 2.0, "us-open": 2.5, "concert": 3.5}
CONCERT_START = "19:30"  # setlist.fm has dates only
DROPPED_STATUS = ("STATUS_CANCELED", "STATUS_POSTPONED")
# Out-of-zone venues whose crowds enter the city at known points (option 4). Each point is
# mapped to the zone containing it, or the nearest zone. Matched by venue-name substring;
# out-of-zone venues not listed here only feed the citywide event features (option 3).
ENTRY_POINTS = {
    "MetLife Stadium": [("Penn Station", 40.7506, -73.9935), ("Port Authority", 40.7570, -73.9903)],
    "Prudential Center": [("Penn Station", 40.7506, -73.9935)],
    "Red Bull Arena": [("33rd St PATH", 40.7489, -73.9881), ("World Trade Center PATH", 40.7126, -74.0111)],
    "Sports Illustrated Stadium": [("33rd St PATH", 40.7489, -73.9881), ("World Trade Center PATH", 40.7126, -74.0111)],  # Red Bull Arena, renamed
    "UBS Arena": [("UBS Arena", 40.7117, -73.7259)],
    "Allstate Arena": [("O'Hare", 41.9786, -87.9048)],
    "Ryan Field": [("Howard CTA", 42.0190, -87.6729)],
    "Welsh-Ryan Arena": [("Howard CTA", 42.0190, -87.6729)],
}

TABLES = {  # file -> (time column for the manifest, description for the dataset card)
    "nyc_zone_hour.parquet": ("hour", "Per NYC taxi zone (1-263) x local hour: pickups and dropoffs (zero-filled); median request-to-pickup wait, fare+tip, driver pay and trip miles of pickups. Source of hotspot labels and demand lags."),
    "chicago_zone_hour.parquet": ("hour", "Per Chicago community area (1-77) x local hour: pickups and dropoffs (zero-filled); median fare+tip and trip miles of pickups (no wait or driver pay published)."),
    "nyc_city_hour.parquet": ("hour", "Citywide pickups per hour, pickups with no valid zone (264/265), dst_merged_hour."),
    "chicago_city_hour.parquet": ("hour", "Citywide pickups per hour, pickups with NULL community area, dst_merged_hour."),
    "nyc_drive_times.parquet": ("year", "Median trip seconds per OD zone pair x year x hour of day x weekend. Use year y-1 for year y."),
    "chicago_drive_times.parquet": ("year", "Median trip seconds per OD community-area pair x year x hour of day x weekend."),
    "zones_nyc.parquet": (None, "GeoParquet: taxi zone polygons, borough, name, area, centroid, subway station complexes."),
    "zones_chicago.parquet": (None, "GeoParquet: community area polygons, name, area, centroid, 'L' stations."),
    "zone_adjacency.parquet": (None, "Neighbouring zone pairs per city (polygons within 20 m), both directions, with centroid distance in km."),
    "event_venues.parquet": (None, "Event venues (sports + concerts) with coordinates, their zone (NULL outside the zones) and, for some out-of-zone venues, the zones their crowds enter through."),
    "events.parquet": ("start_local", "Sports games (pro, college, US Open matches) and concert nights at NYC- and Chicago-area venues, with local start, estimated end, zone and attendance."),
    "weather_asos.parquet": ("valid_utc", "ASOS observations (ORD, MDW, LGA, JFK, Central Park), routine + special METARs."),
    "nws_warnings.parquet": ("utc_issue", "NWS VTEC products from LOT (Chicago) and OKX (NYC); gtype C rows carry UGC zone codes."),
    "bts_airport_hour.parquet": ("hour", "Flights per airport x local hour: scheduled/cancelled by scheduled time, delays by actual time."),
    "mta_zone_hour.parquet": ("hour", "Subway entries and transfers per taxi zone x hour, from 2020-07 (zones with a station only)."),
    "mta_alerts_hour.parquet": ("hour", "MTA service alert counts per agency x hour (subway, bus, LIRR, MNR, bridges & tunnels)."),
    "cta_area_day.parquet": ("date", "CTA 'L' entries per community area x day (areas with a station only). Use the previous day."),
}

CONVENTIONS = """\
## Conventions
- Times are naive local wall-clock (America/New_York, America/Chicago) unless the column ends in `_utc`.
- Hour grids skip the nonexistent spring-forward hour; the fall-back 01:00 hour holds two real hours
  (`dst_merged_hour` in `*_city_hour`).
- Tables are indexed by when things happened. Lag every feature by >= 1 hour before predicting hour t
  (CTA: previous day; drive times: previous year).
- 0 means an observed zero; a missing row or NULL means no data. MTA ridership starts 2020-07; zones
  without a station are absent from `mta_zone_hour` / `cta_area_day` (see the station counts in the
  zones tables). Alert and flight tables only have rows for hours with activity.
- Zone-hour `median_*` columns describe that hour's pickups and are NULL when it had none.
  Dropoffs are counted where and when trips ended. Like pickups, all of them are only known once
  the hour is over, so lag them >= 1 hour.
- Events: schedules are known in advance, so they may describe the target hour. ESPN gives start
  times; end times are estimates (typical length per league). setlist.fm gives dates only, so
  concerts are placed at 19:30-23:00. ESPN attendance is reported after the game. Canceled and
  postponed games are dropped. Out-of-zone venues have `zone` NULL; some list `entry_zones`
  where their crowds enter the city (e.g. MetLife -> Penn Station, Port Authority).
- Splits are by time only: train 2019-2022 excluding 2020-03..2021-06, early-stop 2023,
  calibrate 2024, test 2025+.

## Sources
NYC TLC HVFHV trip records; City of Chicago TNP trips, community areas and CTA ridership;
Iowa Environmental Mesonet (ASOS, NWS VTEC archive); BTS Reporting Carrier On-Time Performance;
MTA via data.ny.gov (subway hourly ridership, service alerts). Built by `scripts/export.py`.
"""


def connect():
    con = duckdb.connect()
    con.execute(f"SET threads = {int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))}")
    mem_mb = int(os.environ.get("SLURM_MEM_PER_NODE", 16 * 1024))
    con.execute(f"SET memory_limit = '{int(mem_mb * 0.75)}MB'")
    if os.environ.get("TMPDIR"):
        con.execute(f"SET temp_directory = '{os.environ['TMPDIR']}/duckdb'")
    return con


def copy(con, query, name):
    con.execute(f"COPY ({query}) TO '{STAGE / name}' (FORMAT parquet, COMPRESSION zstd)")


def zero_filled(con, query, time_col, values, name):
    """Write `query` (zone, time_col, *values) zero-filled over its own zones x time points."""
    fill = ", ".join(f"coalesce(v.{c}, 0) AS {c}" for c in values)
    copy(con, f"""
        WITH v AS ({query})
        SELECT z.zone::SMALLINT AS zone, t.{time_col}, {fill}
        FROM (SELECT DISTINCT {time_col} FROM v) t
        CROSS JOIN (SELECT DISTINCT zone FROM v) z
        LEFT JOIN v ON v.zone = z.zone AND v.{time_col} = t.{time_col}
        ORDER BY t.{time_col}, zone""", name)


def nyc_zones():
    zpath = EXT / "geo" / "taxi_zones.zip"
    # The shapefile sits in a subfolder of the zip, which GDAL won't find on its own.
    member = next(n for n in zipfile.ZipFile(zpath).namelist() if n.lower().endswith(".shp"))
    shp = gpd.read_file(f"/vsizip/{zpath}/{member}").to_crs("EPSG:4326")
    # LocationID has had duplicate values in some releases; OBJECTID is the fallback key.
    key = next((c for c in ("LocationID", "OBJECTID")
                if c in shp and shp[c].is_unique and set(shp[c]) == set(range(1, 264))), None)
    if key is None:
        sys.exit("taxi_zones.zip: neither LocationID nor OBJECTID is a unique 1-263 key")
    if key != "LocationID":
        print("taxi_zones: LocationID is not a clean key, using OBJECTID", flush=True)
    lookup = pd.read_csv(EXT / "geo" / "taxi_zone_lookup.csv").rename(
        columns={"LocationID": "zone", "Borough": "borough", "Zone": "zone_name"})
    zones = gpd.GeoDataFrame({"zone": shp[key].astype("int16")}, geometry=shp.geometry.values, crs="EPSG:4326")
    return zones.merge(lookup[["zone", "borough", "zone_name", "service_zone"]].astype({"zone": "int16"}),
                       on="zone", how="left")


def chicago_zones():
    geo = EXT / "geo"
    if (geo / "chicago_community_areas.geojson").exists():
        ca = gpd.read_file(geo / "chicago_community_areas.geojson")
    else:  # rows.csv fallback from fetch_geo.sh: WKT in the_geom
        df = pd.read_csv(geo / "chicago_community_areas.csv")
        ca = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df["the_geom"]), crs="EPSG:4326")
    ca = ca.rename(columns=str.lower)
    num = next(c for c in ("area_numbe", "area_number", "area_num_1") if c in ca)
    zones = gpd.GeoDataFrame({"zone": ca[num].astype(int).astype("int16"),
                              "zone_name": ca["community"].str.title()},
                             geometry=ca.geometry.values, crs="EPSG:4326")
    if sorted(zones["zone"]) != list(range(1, 78)):
        sys.exit("community areas: expected exactly zones 1-77")
    return zones


def assign_zones(points, zones, id_col, label):
    """Zone containing each point (latitude/longitude columns); points outside every zone are dropped."""
    pts = gpd.GeoDataFrame(points[[id_col]], crs="EPSG:4326",
                           geometry=gpd.points_from_xy(points["longitude"], points["latitude"]))
    hit = gpd.sjoin(pts, zones[["zone", "geometry"]], how="left", predicate="within").drop_duplicates(id_col)
    lost = hit.loc[hit["zone"].isna(), id_col].tolist()
    if lost:
        print(f"{label}: {len(lost)} stations outside every zone, dropped: {lost}", flush=True)
    return pd.DataFrame({id_col: hit[id_col], "zone": hit["zone"]}).dropna().astype({"zone": "int16"})


def write_zones(zones, station_counts, col, city, name):
    utm = zones.geometry.to_crs(UTM[city])
    cent = utm.centroid.to_crs("EPSG:4326")
    zones.assign(area_km2=utm.area / 1e6, centroid_lat=cent.y, centroid_lon=cent.x,
                 **{col: zones["zone"].map(station_counts).fillna(0).astype("int16")}
                 ).to_parquet(STAGE / name, compression="zstd", index=False)


def adjacency(zones, city):
    """Neighbouring zone pairs (polygons within ADJ_TOLERANCE_M), both directions."""
    utm = zones[["zone", "geometry"]].to_crs(UTM[city])
    buf = utm.assign(geometry=utm.buffer(ADJ_TOLERANCE_M))
    pairs = gpd.sjoin(buf, buf, predicate="intersects")
    pairs = pairs[pairs["zone_left"] != pairs["zone_right"]]
    cent = utm.set_index("zone").centroid
    a = cent.loc[pairs["zone_left"]].reset_index(drop=True)
    b = cent.loc[pairs["zone_right"]].reset_index(drop=True)
    return pd.DataFrame({"city": city, "zone": pairs["zone_left"].astype("int16").values,
                         "neighbor": pairs["zone_right"].astype("int16").values,
                         "centroid_km": (a.distance(b) / 1000).round(3).values})


def zone_of_points(points, zones, city):
    """Zone containing each (lat, lon) point, or the nearest zone when outside all of them."""
    utm = zones[["zone", "geometry"]].to_crs(UTM[city])
    pts = gpd.GeoDataFrame(points, crs="EPSG:4326",
                           geometry=gpd.points_from_xy(points["lon"], points["lat"])).to_crs(UTM[city])
    inside = gpd.sjoin(pts, utm, how="left", predicate="within").drop_duplicates("pid")
    nearest = gpd.sjoin_nearest(pts, utm, how="left").drop_duplicates("pid")
    return inside.set_index("pid")["zone"], nearest.set_index("pid")["zone"]


def event_venues(zones_by_city):
    """Every geocoded venue with its zone (inside a city's zones) or entry zones (option 4)."""
    v = pd.read_parquet(EXT / "events" / "venues.parquet")
    lost = v[v["lat"].isna()]
    if len(lost):
        print(f"events: {len(lost)} venues without coordinates, dropped: {lost['venue_name'].tolist()}", flush=True)
    v = v.dropna(subset=["lat"]).assign(metro=lambda d: d["state"].map(METRO_OF_STATE)).reset_index(drop=True)
    v["pid"] = v.index
    v["zone"], v["entry_zones"] = pd.NA, None
    for city, zones in zones_by_city.items():
        mine = v[v["metro"] == city]
        inside, _ = zone_of_points(mine[["pid", "lat", "lon"]], zones, city)
        v.loc[mine.index, "zone"] = inside.reindex(mine["pid"]).values
        for i in mine.index[v.loc[mine.index, "zone"].isna()]:
            key = next((k for k in ENTRY_POINTS if k.lower() in v.at[i, "venue_name"].lower()), None)
            if key is None:
                continue
            pts = pd.DataFrame([(n, la, lo) for n, (_, la, lo) in enumerate(ENTRY_POINTS[key])], columns=["pid", "lat", "lon"])
            ins, near = zone_of_points(pts, zones, city)
            v.at[i, "entry_zones"] = sorted({int(z) for z in ins.fillna(near).values})
    v["zone"] = v["zone"].astype("Int16")
    print(f"events: {v['zone'].notna().sum()} venues inside zones, {v['entry_zones'].notna().sum()} linked by entry "
          f"zones, {(v['zone'].isna() & v['entry_zones'].isna()).sum()} citywide only", flush=True)
    return v.drop(columns="pid")


def events_table(venues):
    """One row per game / US Open match / concert night, with local times and venue zone."""
    ev = EXT / "events"
    s = pd.read_parquet(ev / "sports.parquet")
    s = s[~s["status"].isin(DROPPED_STATUS)].assign(source="espn", title=lambda d: d["away"] + " at " + d["home"])
    s["title"] = s["title"].fillna(s["detail"])
    c = pd.read_parquet(ev / "concerts.parquet").merge(
        pd.read_parquet(ev / "concert_venues.parquet").rename(columns={"name": "venue_name"}),
        on=["venue_key", "setlistfm_id"])
    c = (c.groupby(["setlistfm_id", "venue_name", "city", "state", "date"], as_index=False)
          .agg(title=("artist", lambda a: ", ".join(sorted(set(a.dropna())))), n_acts=("artist", "nunique")))
    c = c.assign(source="setlistfm", league="concert", attendance=None,
                 event_id=lambda d: d["setlistfm_id"] + "-" + d["date"].astype(str),
                 start_local=lambda d: pd.to_datetime(d["date"].astype(str) + " " + CONCERT_START))
    s = s.assign(metro=s["state"].map(METRO_OF_STATE))
    s["start_local"] = [pd.Timestamp(t).tz_localize("UTC").tz_convert(METRO_TZ[m]).tz_localize(None)
                        for t, m in zip(s["start_utc"], s["metro"])]
    c["metro"] = c["state"].map(METRO_OF_STATE)
    cols = ["metro", "source", "league", "event_id", "venue_name", "city", "state", "start_local", "attendance", "title"]
    out = pd.concat([s[cols], c[cols]], ignore_index=True)
    out = out.merge(venues[["venue_name", "city", "state", "zone"]], on=["venue_name", "city", "state"], how="left")
    out["end_local_est"] = out["start_local"] + pd.to_timedelta(out["league"].map(EVENT_HOURS), unit="h")
    out["attendance"] = pd.to_numeric(out["attendance"], errors="coerce")
    out["event_id"] = out["event_id"].astype(str)
    print("events by metro and source:\n" + out.groupby(["metro", "source"]).size().to_string(), flush=True)
    return out.sort_values("start_local")


def mta_zone_hour(con, zones):
    src = f"read_parquet('{EXT}/mta/subway_hourly_*.parquet')"
    stations = con.execute(f"""
        SELECT station_complex_id, arg_max(latitude, transit_timestamp) AS latitude,
               arg_max(longitude, transit_timestamp) AS longitude
        FROM {src} GROUP BY station_complex_id""").df()
    station_zone = assign_zones(stations, zones, "station_complex_id", "mta")
    con.register("mta_station_zone", station_zone)
    zero_filled(con, f"""
        SELECT s.zone, transit_timestamp AS hour, sum(ridership) AS entries, sum(transfers) AS transfers
        FROM {src} JOIN mta_station_zone s USING (station_complex_id) GROUP BY ALL""",
                "hour", ("entries", "transfers"), "mta_zone_hour.parquet")
    return station_zone.groupby("zone").size()


def cta_area_day(con, zones):
    station_zone = assign_zones(pd.read_parquet(EXT / "cta" / "l_stations.parquet"), zones, "station_id", "cta")
    con.register("cta_station_zone", station_zone)
    src = f"read_parquet('{EXT}/cta/l_station_day.parquet')"
    unmatched = [n for (n,) in con.execute(
        f"SELECT DISTINCT stationname FROM {src} ANTI JOIN cta_station_zone USING (station_id)").fetchall()]
    if unmatched:
        print(f"cta: ridership for stations with no location, dropped: {unmatched}", flush=True)
    zero_filled(con, f"""
        SELECT s.zone, d.date, sum(d.rides) AS rides
        FROM {src} d JOIN cta_station_zone s USING (station_id) GROUP BY ALL""",
                "date", ("rides",), "cta_area_day.parquet")
    return station_zone.groupby("zone").size()


def bts_airport_hour(con):
    airports = ", ".join(f"'{a}'" for a in AIRPORT_CITY)
    city = " ".join(f"WHEN '{a}' THEN '{c}'" for a, c in AIRPORT_CITY.items())
    # Scheduled arrival before scheduled departure (hhmm) means it lands the next day.
    copy(con, f"""
        WITH f AS (
            SELECT Origin, Dest, Cancelled, Diverted, DepDelay, ArrDelay,
                   FlightDate + to_minutes(CRSDepTime // 100 * 60 + CRSDepTime % 100) AS sched_dep,
                   FlightDate + to_minutes(CRSArrTime // 100 * 60 + CRSArrTime % 100)
                     + CASE WHEN CRSArrTime < CRSDepTime THEN INTERVAL 1 DAY ELSE INTERVAL 0 DAY END AS sched_arr
            FROM read_parquet('{EXT}/bts/flights_*.parquet')),
        ev AS (
            SELECT Origin AS airport, sched_dep AS t, 'dep_sched' AS kind, Cancelled AS cancelled,
                   Diverted AS diverted, NULL::DOUBLE AS delay FROM f
            UNION ALL SELECT Origin, sched_dep + to_minutes(DepDelay::INTEGER), 'dep_actual', NULL, NULL, DepDelay
                      FROM f WHERE DepDelay IS NOT NULL
            UNION ALL SELECT Dest, sched_arr, 'arr_sched', Cancelled, Diverted, NULL FROM f
            UNION ALL SELECT Dest, sched_arr + to_minutes(ArrDelay::INTEGER), 'arr_actual', NULL, NULL, ArrDelay
                      FROM f WHERE ArrDelay IS NOT NULL)
        SELECT airport, CASE airport {city} END AS city, date_trunc('hour', t) AS hour,
               count(*) FILTER (kind = 'dep_sched') AS dep_scheduled,
               count(*) FILTER (kind = 'dep_sched' AND cancelled) AS dep_cancelled,
               count(*) FILTER (kind = 'dep_actual') AS dep_departed,
               avg(delay) FILTER (kind = 'dep_actual') AS dep_delay_mean,
               quantile_cont(delay, 0.9) FILTER (kind = 'dep_actual') AS dep_delay_p90,
               count(*) FILTER (kind = 'dep_actual' AND delay >= 15) AS dep_delayed15,
               count(*) FILTER (kind = 'arr_sched') AS arr_scheduled,
               count(*) FILTER (kind = 'arr_sched' AND cancelled) AS arr_cancelled,
               count(*) FILTER (kind = 'arr_sched' AND diverted) AS arr_diverted,
               count(*) FILTER (kind = 'arr_actual') AS arr_arrived,
               avg(delay) FILTER (kind = 'arr_actual') AS arr_delay_mean,
               quantile_cont(delay, 0.9) FILTER (kind = 'arr_actual') AS arr_delay_p90,
               count(*) FILTER (kind = 'arr_actual' AND delay >= 15) AS arr_delayed15
        FROM ev WHERE t IS NOT NULL AND airport IN ({airports})
        GROUP BY ALL ORDER BY airport, hour""", "bts_airport_hour.parquet")


def mta_alerts_hour(con):
    agencies = ", ".join(f"'{a}'" for a in ALERT_AGENCIES)
    copy(con, f"""
        WITH a AS (
            SELECT date, coalesce(m.agency, r.agency) AS agency, status_id AS event_id,
                   lower(status_label) AS label, affected
            FROM read_parquet('{EXT}/mta/alerts_2012_2020.parquet') r
            LEFT JOIN (VALUES ('Subway', 'NYCT Subway'), ('Bus', 'NYCT Bus')) m(old, agency) ON r.agency = m.old
            UNION ALL
            SELECT date, agency, event_id::VARCHAR, lower(status_label), affected
            FROM read_parquet('{EXT}/mta/alerts_2020_present.parquet')),
        f AS (SELECT date_trunc('hour', date) AS hour, * FROM a WHERE agency IN ({agencies})),
        routes AS (
            SELECT hour, agency, count(DISTINCT trim(r)) AS n_routes_affected
            FROM (SELECT hour, agency, unnest(string_split(affected, '|')) AS r FROM f)
            WHERE trim(r) <> '' GROUP BY ALL)
        SELECT hour, agency,
               count(*) AS n_updates,
               count(DISTINCT event_id) AS n_events,
               count(*) FILTER (label LIKE '%suspend%' OR label LIKE '%no scheduled service%') AS n_suspension_updates,
               count(*) FILTER (label LIKE '%rerout%') AS n_reroute_updates,
               count(*) FILTER (label LIKE '%delay%') AS n_delay_updates,
               coalesce(any_value(routes.n_routes_affected), 0) AS n_routes_affected
        FROM f LEFT JOIN routes USING (hour, agency)
        GROUP BY hour, agency ORDER BY hour, agency""", "mta_alerts_hour.parquet")


def build_manifest(con):
    tables = {}
    for name, (time_col, desc) in TABLES.items():
        path = STAGE / name
        t = dict(description=desc,
                 rows=con.execute(f"SELECT count(*) FROM '{path}'").fetchone()[0],
                 bytes=path.stat().st_size,
                 columns={c[0]: c[1] for c in con.execute(f"DESCRIBE SELECT * FROM '{path}'").fetchall()})
        if time_col:
            t["min"], t["max"] = con.execute(f"SELECT min({time_col}), max({time_col}) FROM '{path}'").fetchone()
        tables[name] = t
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    return dict(generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                git_commit=sha or None, tables=tables)


def dataset_card(manifest):
    lines = ["---", "license: other", "pretty_name: Rideshare Hotspot Forecasting (Chicago + NYC)", "configs:"]
    for name in manifest["tables"]:
        lines += [f"- config_name: {name.removesuffix('.parquet')}", f"  data_files: {name}"]
    lines += ["---", "", "# Rideshare hotspot forecasting: model-grain tables", "",
              f"Generated {manifest['generated_utc']} from commit `{manifest['git_commit']}`. "
              "Full column types in `MANIFEST.json`.", "",
              "| file | rows | MB | time range | description |", "|---|---|---|---|---|"]
    for name, t in manifest["tables"].items():
        span = f"{t['min']} → {t['max']}" if "min" in t else ""
        lines.append(f"| `{name}` | {t['rows']:,} | {t['bytes'] / 1e6:.1f} | {span} | {t['description']} |")
    return "\n".join(lines) + "\n\n" + CONVENTIONS


def main():
    con = connect()
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    for city in ("nyc", "chicago"):
        for kind in ("zone_hour", "city_hour", "drive_times"):
            shutil.copy2(PROC / f"{city}_{kind}.parquet", STAGE)
    shutil.copy2(EXT / "weather" / "asos.parquet", STAGE / "weather_asos.parquet")
    shutil.copy2(EXT / "nws" / "warnings.parquet", STAGE / "nws_warnings.parquet")

    nyc, chicago = nyc_zones(), chicago_zones()
    write_zones(nyc, mta_zone_hour(con, nyc), "n_subway_complexes", "nyc", "zones_nyc.parquet")
    write_zones(chicago, cta_area_day(con, chicago), "n_l_stations", "chicago", "zones_chicago.parquet")
    pd.concat([adjacency(nyc, "nyc"), adjacency(chicago, "chicago")]).to_parquet(
        STAGE / "zone_adjacency.parquet", compression="zstd", index=False)
    venues = event_venues({"nyc": nyc, "chicago": chicago})
    venues.to_parquet(STAGE / "event_venues.parquet", compression="zstd", index=False)
    events_table(venues).to_parquet(STAGE / "events.parquet", compression="zstd", index=False)
    bts_airport_hour(con)
    mta_alerts_hour(con)

    manifest = build_manifest(con)
    (STAGE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str))
    (STAGE / "README.md").write_text(dataset_card(manifest))

    if EXPORT.exists():
        shutil.rmtree(EXPORT)
    STAGE.rename(EXPORT)
    total = sum(t["bytes"] for t in manifest["tables"].values())
    print(f"export ready: {len(manifest['tables'])} tables, {total / 1e6:.1f} MB in {EXPORT}", flush=True)


if __name__ == "__main__":
    main()
