#!/usr/bin/env python
"""SUPERSEDED: the team's feature builder is scripts/label.py + scripts/features.py
(MODELING_PLAN.md), which `run_pipeline.sh features` and the training jobs use. This script is
kept as a reference for extra features (dropoffs, NYC wait/pay, neighbours, holidays, drive
times, lead time k = 2) that could be ported into features.py.

Build the model feature table: one row per zone x target hour h, per city and lead time k.

Input:  data/export/ (the published model-grain tables)
Output: data/features/{city}_k{k}.parquet

Timing. A forecast for hour h with lead time k is issued at I = start of hour h-k+1 (k = 1:
I = h). History features describe the zone up to and including hour t = h-k; conditions use
what was observed before I; schedules (calendar, flights, events, already-issued NWS products)
may describe h itself. Seasonal lags (h-24, h-168, same hour of week over 4 weeks) are relative
to h and valid for k <= 24.

Label. is_hotspot = the zone's pickups at h are > 0 and its within-hour percent_rank >= 0.9
(top 10% of the city's zones; tied zones share the lower rank). pickups and pct are kept so the
definition can change without a rebuild.

Condition flags (cond_*) mark extreme-condition subsets for evaluation. They use the same
information as the features; the thresholds below are placeholders to tune.

Usage: python scripts/build_features.py [--city nyc chicago] [--gap 1 2] [--start D --end D]
       --start/--end limit the target hours (for quick tests); history before --start is read.
"""
import argparse
import os
from datetime import date
from pathlib import Path

import duckdb
import geopandas as gpd
import holidays
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "data" / "export"
OUT = ROOT / "data" / "features"
LOOKBACK_DAYS = 29  # history needed before the first target hour (4 weeks + 1 day)
COVID = ("2020-03-01", "2021-07-01")  # excluded from training: [start, end)

CITIES = {
    "nyc": dict(
        tz="America/New_York", subdiv="NY", stations=("NYC", "LGA", "JFK"),
        ugc=("NYC005", "NYC047", "NYC061", "NYC081", "NYC085",
             "NYZ072", "NYZ073", "NYZ074", "NYZ075", "NYZ176", "NYZ178"),
        airports={"JFK": (40.6413, -73.7781), "LGA": (40.7769, -73.8740), "EWR": (40.6895, -74.1745)}),
    "chicago": dict(
        tz="America/Chicago", subdiv="IL", stations=("MDW", "ORD"),
        ugc=("ILC031", "ILZ014", "ILZ103", "ILZ104"),  # Cook County; ILZ014 split into 103/104 in 2020-03
        airports={"ORD": (41.9786, -87.9048), "MDW": (41.7868, -87.7522)}),
}
NWS_GROUPS = {  # VTEC phenomena codes per hazard group
    "winter": ("WS", "WW", "BZ", "IS", "LE", "ZR", "SN"),
    "cold": ("WC", "EC", "CW", "XC", "FZ", "HZ"),
    "heat": ("HT", "EH", "XH"),
    "flood": ("FA", "FF", "FL", "CF", "LS"),
    "severe": ("SV", "TO"),
    "wind": ("WI", "HW", "LW", "EW"),
}
THRESH = dict(heavy_precip_in=0.25, hot_f=95, cold_f=10, flight_cancels_3h=20, flight_p90_min=90)


def connect():
    con = duckdb.connect()
    con.execute(f"SET threads = {int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))}")
    mem_mb = int(os.environ.get("SLURM_MEM_PER_NODE", 16 * 1024))
    con.execute(f"SET memory_limit = '{int(mem_mb * 0.75)}MB'")
    con.execute("SET preserve_insertion_order = false")
    if os.environ.get("TMPDIR"):
        con.execute(f"SET temp_directory = '{os.environ['TMPDIR']}/duckdb'")
    return con


def t(name):
    return f"read_parquet('{EXPORT / name}')"


def airport_zones(city, c):
    """Zone containing each airport, from the zone polygons."""
    zones = gpd.read_parquet(EXPORT / f"zones_{city}.parquet")[["zone", "geometry"]]
    pts = gpd.GeoDataFrame({"airport": list(c["airports"])}, crs="EPSG:4326",
                           geometry=gpd.points_from_xy([lon for _, lon in c["airports"].values()],
                                                       [lat for lat, _ in c["airports"].values()]))
    return gpd.sjoin(pts, zones, predicate="within")[["airport", "zone"]].reset_index(drop=True)


def holiday_table(c, years):
    """Per date: is_holiday, days to the next and since the last holiday (capped at 14)."""
    days = pd.date_range(f"{years[0]}-01-01", f"{years[1]}-12-31", freq="D")
    hol = holidays.country_holidays("US", subdiv=c["subdiv"], years=range(years[0] - 1, years[1] + 2))
    hd = pd.to_datetime(sorted(hol))
    nxt = pd.Series(hd, index=hd).reindex(days, method="bfill")
    prv = pd.Series(hd, index=hd).reindex(days, method="ffill")
    return pd.DataFrame({"date": days.date, "is_holiday": days.isin(hd),
                         "days_to_holiday": ((nxt.values - days.values) / pd.Timedelta("1D")).clip(0, 14),
                         "days_since_holiday": ((days.values - prv.values) / pd.Timedelta("1D")).clip(0, 14)})


def build(con, city, k, start, end):
    c = CITIES[city]
    lo = pd.Timestamp(start) - pd.Timedelta(days=LOOKBACK_DAYS) if start else None
    rng = (f"WHERE hour >= TIMESTAMP '{lo}' AND hour < TIMESTAMP '{end}'" if start and end else
           f"WHERE hour >= TIMESTAMP '{lo}'" if start else f"WHERE hour < TIMESTAMP '{end}'" if end else "")
    target = (f"hour >= TIMESTAMP '{start}'" if start else "TRUE") + (f" AND hour < TIMESTAMP '{end}'" if end else "")

    # Hour index: consecutive real hours (the zero-filled grid), so lags count hours, not timestamps.
    con.execute(f"""CREATE OR REPLACE TEMP TABLE hours AS
        SELECT hour, row_number() OVER (ORDER BY hour) AS hi, dst_merged_hour, pickups_total, pickups_unzoned
        FROM {t(f'{city}_city_hour.parquet')} {rng}""")

    # Panel with label, per-zone history windows (relative to each row's own hour) and seasonal lags.
    con.execute(f"""CREATE OR REPLACE TEMP TABLE panel AS
        WITH p AS (
            SELECT z.*, h.hi, percent_rank() OVER (PARTITION BY z.hour ORDER BY z.pickups) AS pct
            FROM {t(f'{city}_zone_hour.parquet')} z JOIN hours h USING (hour)),
        l AS (SELECT *, pickups > 0 AND pct >= 0.9 AS is_hotspot FROM p)
        SELECT *,
            lag(pickups, 1) OVER w AS pu_lag1, lag(pickups, 2) OVER w AS pu_lag2,
            avg(pickups) OVER (w ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS pu_mean3,
            avg(pickups) OVER (w ROWS BETWEEN 23 PRECEDING AND CURRENT ROW) AS pu_mean24,
            lag(dropoffs, 1) OVER w AS do_lag1,
            sum(dropoffs) OVER (w ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS do_sum3,
            avg(median_wait_s) OVER (w ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS wait_mean3,
            lag(pickups, 24) OVER w AS pu_h24, lag(pickups, 168) OVER w AS pu_h168,
            lag(is_hotspot::INT, 24) OVER w AS hs_h24, lag(is_hotspot::INT, 168) OVER w AS hs_h168,
            lag(pct, 168) OVER w AS pct_h168,
            (lag(pickups, 168) OVER w + lag(pickups, 336) OVER w + lag(pickups, 504) OVER w
             + lag(pickups, 672) OVER w) / 4.0 AS pu_how_mean4,
            (lag(is_hotspot::INT, 168) OVER w + lag(is_hotspot::INT, 336) OVER w + lag(is_hotspot::INT, 504) OVER w
             + lag(is_hotspot::INT, 672) OVER w) / 4.0 AS hs_how_rate4
        FROM l WINDOW w AS (PARTITION BY zone ORDER BY hi)""")

    # Neighbours' state and city state at each hour.
    con.execute(f"""CREATE OR REPLACE TEMP TABLE nbr AS
        SELECT a.zone, p.hi, sum(p.pickups) AS nbr_pickups, avg(p.is_hotspot::INT) AS nbr_hotspot_share
        FROM {t('zone_adjacency.parquet')} a JOIN panel p ON p.zone = a.neighbor
        WHERE a.city = '{city}' GROUP BY a.zone, p.hi""")
    con.execute("""CREATE OR REPLACE TEMP TABLE cityh AS
        SELECT hi, pickups_total AS city_pickups, pickups_unzoned AS city_unzoned,
               pickups_total / nullif(lag(pickups_total, 168) OVER (ORDER BY hi), 0) AS city_vs_lastweek
        FROM hours""")

    # Weather per issue time I: obs in [I-1h, I) (3h and 24h windows for precip and snow depth).
    st = ", ".join(f"'{s}'" for s in c["stations"])
    con.execute(f"""CREATE OR REPLACE TEMP TABLE wx AS
        WITH b AS (
            SELECT date_trunc('hour', valid_local) + INTERVAL 1 HOUR AS issue,
                   avg(tmpf) AS temp_f, avg(feel) AS feel_f, max(sknt) AS wind_kt, max(gust) AS gust_kt,
                   min(vsby) AS vis_mi, max(p01i) AS precip_1h, max(snowdepth) AS snowdepth_in,
                   bool_or(wxcodes LIKE '%TS%') AS wx_thunder, bool_or(wxcodes LIKE '%SN%') AS wx_snow,
                   bool_or(wxcodes LIKE '%FZ%') AS wx_freezing, bool_or(wxcodes LIKE '%RA%') AS wx_rain,
                   bool_or(wxcodes LIKE '%FG%') AS wx_fog
            FROM {t('weather_asos.parquet')} WHERE city = '{city}' AND station IN ({st}) GROUP BY 1)
        SELECT h.hour AS issue, b.* EXCLUDE (issue, snowdepth_in),
               sum(b.precip_1h) OVER (ORDER BY h.hour ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS precip_3h,
               max(b.snowdepth_in) OVER (ORDER BY h.hour ROWS BETWEEN 23 PRECEDING AND CURRENT ROW) AS snowdepth_in
        FROM hours h LEFT JOIN b ON b.issue = h.hour""")

    # NWS products active at I (county/zone rows covering the city).
    ugc = ", ".join(f"'{u}'" for u in c["ugc"])
    grp = ", ".join(f"count(DISTINCT n.key) FILTER (n.phenomena IN ({', '.join(repr(p) for p in codes)})) > 0 AS nws_{g}"
                    for g, codes in NWS_GROUPS.items())
    con.execute(f"""CREATE OR REPLACE TEMP TABLE nws AS
        WITH n AS (
            SELECT phenomena, significance, phenomena || significance || eventid || vtec_year AS key,
                   utc_issue, utc_expire
            FROM {t('nws_warnings.parquet')} WHERE gtype = 'C' AND ugc IN ({ugc}) AND status <> 'CAN'),
        i AS (SELECT hour AS issue, timezone('UTC', timezone('{c['tz']}', hour)) AS issue_utc FROM hours)
        SELECT i.issue,
               count(DISTINCT n.key) FILTER (n.significance = 'W') AS nws_warnings,
               count(DISTINCT n.key) FILTER (n.significance = 'Y') AS nws_advisories,
               count(DISTINCT n.key) FILTER (n.significance = 'A') AS nws_watches, {grp}
        FROM i LEFT JOIN n ON n.utc_issue <= i.issue_utc AND i.issue_utc < n.utc_expire
        GROUP BY i.issue""")

    # Flights: schedules at h (known ahead) per airport zone and city; actuals over [I-3h, I).
    con.register("apz", airport_zones(city, c))
    con.execute(f"""CREATE OR REPLACE TEMP TABLE fl AS
        SELECT b.hour, b.airport, a.zone, b.arr_scheduled, b.dep_scheduled,
               b.dep_cancelled + b.arr_cancelled AS cancelled, b.arr_delay_p90
        FROM {t('bts_airport_hour.parquet')} b JOIN apz a USING (airport) WHERE b.city = '{city}'""")
    con.execute("""CREATE OR REPLACE TEMP TABLE flc AS
        SELECT h.hour, coalesce(sum(f.arr_scheduled), 0) AS city_arr_sched, coalesce(sum(f.dep_scheduled), 0) AS city_dep_sched,
               coalesce(sum(f.cancelled), 0) AS city_cancelled, max(f.arr_delay_p90) AS city_arr_p90
        FROM hours h LEFT JOIN fl f USING (hour) GROUP BY h.hour""")
    con.execute("""CREATE OR REPLACE TEMP TABLE flr AS
        SELECT hour AS issue,
               sum(city_cancelled) OVER (ORDER BY hour ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS fl_cancelled_3h,
               max(city_arr_p90) OVER (ORDER BY hour ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS fl_arr_p90_3h
        FROM flc""")

    # Transit at t / over [I-3h, I).
    if city == "nyc":
        con.execute(f"""CREATE OR REPLACE TEMP TABLE transit AS
            WITH cov AS (SELECT min(hour) AS lo, max(hour) AS hi FROM {t('mta_zone_hour.parquet')}),
                 zs AS (SELECT zone, n_subway_complexes FROM {t('zones_nyc.parquet')})
            SELECT p.zone, p.hi,
                   CASE WHEN p.hour BETWEEN cov.lo AND cov.hi
                        THEN CASE WHEN zs.n_subway_complexes > 0 THEN m.entries ELSE 0 END END AS subway_entries
            FROM panel p CROSS JOIN cov JOIN zs USING (zone)
            LEFT JOIN {t('mta_zone_hour.parquet')} m ON m.zone = p.zone AND m.hour = p.hour""")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE alerts AS
            WITH a AS (SELECT * FROM {t('mta_alerts_hour.parquet')}),
                 h AS (SELECT h.hour, coalesce(sum(n_events) FILTER (agency = 'NYCT Subway'), 0) AS sub_ev,
                              coalesce(sum(n_suspension_updates) FILTER (agency = 'NYCT Subway'), 0) AS sub_susp,
                              coalesce(sum(n_delay_updates) FILTER (agency = 'NYCT Subway'), 0) AS sub_delay,
                              coalesce(sum(n_events) FILTER (agency = 'NYCT Bus'), 0) AS bus_ev,
                              coalesce(sum(n_events) FILTER (agency IN ('LIRR', 'MNR')), 0) AS rail_ev,
                              coalesce(sum(n_events) FILTER (agency = 'BT'), 0) AS bt_ev
                       FROM hours h LEFT JOIN a USING (hour) GROUP BY h.hour)
            SELECT hour AS issue,
                   sum(sub_ev) OVER w AS subway_alerts_3h, sum(sub_susp) OVER w AS subway_suspensions_3h,
                   sum(sub_delay) OVER w AS subway_delays_3h, sum(bus_ev) OVER w AS bus_alerts_3h,
                   sum(rail_ev) OVER w AS rail_alerts_3h, sum(bt_ev) OVER w AS bridge_tunnel_alerts_3h
            FROM h WINDOW w AS (ORDER BY hour ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING)""")
    else:
        con.execute(f"""CREATE OR REPLACE TEMP TABLE transit AS
            WITH cov AS (SELECT min(date) AS lo, max(date) AS hi FROM {t('cta_area_day.parquet')}),
                 zs AS (SELECT zone, n_l_stations FROM {t('zones_chicago.parquet')}),
                 d AS (SELECT DISTINCT p.zone, p.hour::DATE - 1 AS prev_day FROM panel p)
            SELECT d.zone, d.prev_day,
                   CASE WHEN d.prev_day BETWEEN cov.lo AND cov.hi
                        THEN CASE WHEN zs.n_l_stations > 0 THEN coalesce(r.rides, 0) ELSE 0 END END AS l_rides_prev_day
            FROM d CROSS JOIN cov JOIN zs USING (zone)
            LEFT JOIN {t('cta_area_day.parquet')} r ON r.zone = d.zone AND r.date = d.prev_day""")

    # Drive times: last year's trip-weighted median trip time out of the zone, by hour of day and weekend.
    con.execute(f"""CREATE OR REPLACE TEMP TABLE dt AS
        SELECT pu_zone AS zone, year, hour_of_day, is_weekend,
               sum(median_trip_seconds * n_trips) / sum(n_trips) AS out_trip_s_lastyear
        FROM {t(f'{city}_drive_times.parquet')} GROUP BY ALL""")

    # Events around h (schedules, known ahead): in the zone, at neighbours, linked by entry zones,
    # and citywide at out-of-zone venues.
    # Each event is expanded to the hours it can touch (start - 3h .. end + 2h), then counted
    # per zone-hour; joining every zone-hour to every event would be far too large.
    con.execute(f"""CREATE OR REPLACE TEMP TABLE ev AS
        SELECT e.*, v.entry_zones,
               unnest(generate_series(date_trunc('hour', e.start_local) - INTERVAL 3 HOUR,
                                      date_trunc('hour', e.end_local_est) + INTERVAL 2 HOUR, INTERVAL 1 HOUR)) AS hour
        FROM {t('events.parquet')} e
        LEFT JOIN {t('event_venues.parquet')} v USING (venue_name, city, state) WHERE e.metro = '{city}'""")
    end_now = "end_local_est >= hour - INTERVAL 1 HOUR AND end_local_est < hour + INTERVAL 1 HOUR"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE evz AS
        SELECT zone, hour,
               count(*) FILTER (start_local >= hour AND start_local < hour + INTERVAL 3 HOUR) AS ev_start_3h,
               count(*) FILTER ({end_now}) AS ev_end_now,
               sum(attendance) FILTER ({end_now}) AS ev_end_attendance
        FROM ev WHERE zone IS NOT NULL GROUP BY zone, hour""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE evn AS
        SELECT a.zone, z.hour, sum(z.ev_end_now) AS ev_nbr_end_now
        FROM {t('zone_adjacency.parquet')} a JOIN evz z ON z.zone = a.neighbor
        WHERE a.city = '{city}' GROUP BY a.zone, z.hour""")
    con.execute("""CREATE OR REPLACE TEMP TABLE eve AS
        SELECT zone, hour, count(*) AS ev_entry_end
        FROM (SELECT unnest(entry_zones) AS zone, hour FROM ev
              WHERE zone IS NULL AND entry_zones IS NOT NULL
                AND end_local_est >= hour - INTERVAL 1 HOUR AND end_local_est < hour + INTERVAL 2 HOUR)
        GROUP BY zone, hour""")
    out_end = "zone IS NULL AND end_local_est >= hour - INTERVAL 2 HOUR AND end_local_est < hour + INTERVAL 1 HOUR"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE evc AS
        SELECT hour, count(*) FILTER ({out_end}) AS ev_city_outzone_end,
               sum(attendance) FILTER ({out_end}) AS ev_city_outzone_attendance,
               count(*) FILTER (start_local <= hour AND hour < end_local_est) AS ev_city_live
        FROM ev GROUP BY hour""")

    years = con.execute("SELECT year(min(hour)), year(max(hour)) FROM hours").fetchone()
    con.register("hol", holiday_table(c, years))
    transit_cols, transit_join = (
        ("tr.subway_entries AS subway_entries_t, al.subway_alerts_3h, al.subway_suspensions_3h, al.subway_delays_3h, "
         "al.bus_alerts_3h, al.rail_alerts_3h, al.bridge_tunnel_alerts_3h, al.subway_suspensions_3h > 0 AS cond_transit",
         "LEFT JOIN transit tr ON tr.zone = s.zone AND tr.hi = s.hi LEFT JOIN alerts al ON al.issue = y.issue")
        if city == "nyc" else
        ("tr.l_rides_prev_day, NULL::BOOLEAN AS cond_transit",
         "LEFT JOIN transit tr ON tr.zone = y.zone AND tr.prev_day = y.issue::DATE - 1"))

    # Assemble: target rows y (hour h) joined to state s at t = h - k and conditions at I.
    q = f"""
        WITH y AS (
            SELECT p.zone, p.hour, p.hi, p.pickups, p.pct, p.is_hotspot,
                   p.pu_h24, p.pu_h168, p.hs_h24, p.hs_h168, p.pct_h168, p.pu_how_mean4, p.hs_how_rate4,
                   h.dst_merged_hour, ti.hour + INTERVAL 1 HOUR AS issue
            FROM panel p JOIN hours h USING (hi) JOIN hours ti ON ti.hi = p.hi - {k}
            WHERE {target.replace('hour', 'p.hour')})
        SELECT y.zone, y.hour, y.issue, {k} AS gap_k, y.pickups, y.pct, y.is_hotspot,
               y.hour >= TIMESTAMP '{COVID[0]}' AND y.hour < TIMESTAMP '{COVID[1]}' AS covid_excluded,
               -- history at t = h - k
               s.pickups AS pu_t, s.pu_lag1 AS pu_t1, s.pu_lag2 AS pu_t2, s.pu_mean3, s.pu_mean24,
               s.dropoffs AS do_t, s.do_lag1 AS do_t1, s.do_sum3, s.pickups - s.dropoffs AS net_flow_t,
               s.median_wait_s AS wait_t, s.wait_mean3, s.median_fare AS fare_t, s.median_driver_pay AS pay_t,
               s.median_trip_miles AS miles_t, s.median_trip_seconds AS trip_s_t,
               s.is_hotspot::INT AS hs_t, s.pct AS pct_t,
               n.nbr_pickups AS nbr_pickups_t, n.nbr_hotspot_share AS nbr_hotspot_share_t,
               ch.city_pickups AS city_pickups_t, ch.city_unzoned AS city_unzoned_t, ch.city_vs_lastweek AS city_vs_lastweek_t,
               -- seasonal, relative to h
               y.pu_h24, y.pu_h168, y.hs_h24, y.hs_h168, y.pct_h168, y.pu_how_mean4, y.hs_how_rate4,
               -- calendar at h
               hour(y.hour) AS hour_of_day, dayofweek(y.hour) AS dow, month(y.hour) AS month,
               dayofweek(y.hour) IN (0, 6) AS is_weekend, hol.is_holiday, hol.days_to_holiday, hol.days_since_holiday,
               y.dst_merged_hour,
               -- weather, NWS before I
               w.temp_f, w.feel_f, w.wind_kt, w.gust_kt, w.vis_mi, w.precip_1h, w.precip_3h, w.snowdepth_in,
               w.wx_thunder, w.wx_snow, w.wx_freezing, w.wx_rain, w.wx_fog,
               nw.* EXCLUDE (issue),
               -- flights: schedules at h, actuals before I
               coalesce(fz.arr_scheduled, 0) AS zone_arr_sched_h, flc.city_arr_sched AS city_arr_sched_h,
               flc.city_dep_sched AS city_dep_sched_h, flr.fl_cancelled_3h, flr.fl_arr_p90_3h,
               -- transit
               {transit_cols},
               -- drive times (last year)
               dt.out_trip_s_lastyear,
               -- events around h
               coalesce(ez.ev_start_3h, 0) AS ev_start_3h, coalesce(ez.ev_end_now, 0) AS ev_end_now,
               ez.ev_end_attendance, coalesce(en.ev_nbr_end_now, 0) AS ev_nbr_end_now,
               coalesce(ee.ev_entry_end, 0) AS ev_entry_end, coalesce(ec.ev_city_outzone_end, 0) AS ev_city_outzone_end,
               ec.ev_city_outzone_attendance, coalesce(ec.ev_city_live, 0) AS ev_city_live,
               -- condition flags (evaluation subsets)
               w.precip_1h >= {THRESH['heavy_precip_in']} AS cond_heavy_precip,
               coalesce(w.wx_snow, FALSE) OR coalesce(nw.nws_winter, FALSE) AS cond_snow,
               w.feel_f >= {THRESH['hot_f']} AS cond_heat, w.feel_f <= {THRESH['cold_f']} AS cond_cold,
               nw.nws_warnings > 0 AS cond_nws_warning,
               flr.fl_cancelled_3h >= {THRESH['flight_cancels_3h']} OR flr.fl_arr_p90_3h >= {THRESH['flight_p90_min']} AS cond_flights,
               coalesce(ez.ev_end_now, 0) + coalesce(en.ev_nbr_end_now, 0) + coalesce(ee.ev_entry_end, 0) > 0 AS cond_event
        FROM y
        JOIN panel s ON s.zone = y.zone AND s.hi = y.hi - {k}
        LEFT JOIN nbr n ON n.zone = y.zone AND n.hi = s.hi
        LEFT JOIN cityh ch ON ch.hi = s.hi
        LEFT JOIN hol ON hol.date = y.hour::DATE
        LEFT JOIN wx w ON w.issue = y.issue
        LEFT JOIN nws nw ON nw.issue = y.issue
        LEFT JOIN (SELECT zone, hour, sum(arr_scheduled) AS arr_scheduled FROM fl GROUP BY ALL) fz
               ON fz.zone = y.zone AND fz.hour = y.hour
        LEFT JOIN flc ON flc.hour = y.hour
        LEFT JOIN flr ON flr.issue = y.issue
        {transit_join}
        LEFT JOIN dt ON dt.zone = y.zone AND dt.year = year(y.hour) - 1 AND dt.hour_of_day = hour(y.hour)
                    AND dt.is_weekend = (dayofweek(y.hour) IN (0, 6))
        LEFT JOIN evz ez ON ez.zone = y.zone AND ez.hour = y.hour
        LEFT JOIN evn en ON en.zone = y.zone AND en.hour = y.hour
        LEFT JOIN eve ee ON ee.zone = y.zone AND ee.hour = y.hour
        LEFT JOIN evc ec ON ec.hour = y.hour
        ORDER BY y.hour, y.zone"""
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"_{start}_{end}" if start or end else ""
    out = OUT / f"{city}_k{k}{suffix}.parquet"
    part = out.with_name(out.name + ".part")
    con.execute(f"COPY ({q}) TO '{part}' (FORMAT parquet, COMPRESSION zstd)")
    part.replace(out)
    n, pos = con.execute(f"SELECT count(*), avg(is_hotspot::INT) FROM '{out}'").fetchone()
    cols = len(con.execute(f"DESCRIBE SELECT * FROM '{out}'").fetchall())
    print(f"{city} k={k}: {n:,} rows, {cols} columns, hotspot rate {pos:.3f} -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", nargs="+", default=list(CITIES), choices=list(CITIES))
    ap.add_argument("--gap", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--start")
    ap.add_argument("--end")
    a = ap.parse_args()
    con = connect()
    for city in a.city:
        for k in a.gap:
            build(con, city, k, a.start, a.end)


if __name__ == "__main__":
    main()
