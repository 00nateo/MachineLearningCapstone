#!/usr/bin/env python
"""Build data/processed_ml/{city}_features.parquet: zone-hour label + predictive features
+ the composite extreme-condition flag.

Two different things live in this table, and they must not be confused:
  - Predictive features (the *_lag1 / *_lag24 / *_lag168 / *_sched columns) are what the
    model sees. Anything observational is lagged >= 1h; anything known in advance (events,
    flight schedules) may describe hour h itself. See MODELING_PLAN.md.
  - The composite `extreme_flag` (and its *_flag components, written alongside it so evaluation
    can slice by condition) is NEVER fed to the model. It
    is computed on the actual, unlagged hour h and used only for (a) stratified training-day
    sampling weight and (b) slicing evaluation into normal/extreme subgroups. Using same-hour
    truth here is not leakage because the model never sees this column.

Stages: own-zone lags (week 1) -> weather -> NWS warnings -> events -> transit -> flights ->
final assembly with the composite extreme flag.

Usage: python scripts/features.py [nyc|chicago ...]     (default: both)
"""
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "data" / "export"
PROC_ML = ROOT / "data" / "processed_ml"

CITIES = ("nyc", "chicago")
TZ = {"nyc": "America/New_York", "chicago": "America/Chicago"}
WEATHER_STATION = {"nyc": "NYC", "chicago": "ORD"}  # NYC: Central Park
# Found via zone-name match (NYC) / DuckDB ST_Contains against the airport's real coordinates
# (Chicago) -- Midway sits in Garfield Ridge, not Clearing, so this was checked, not assumed.
AIRPORT_ZONE = {"JFK": 132, "LGA": 138, "EWR": 1, "ORD": 76, "MDW": 56}
NYC_SUBWAY_AGENCY = "NYCT Subway"
MAJOR_EVENT_ATTENDANCE = 15000
FLIGHT_DELAY_RATE = 0.5  # ~90th percentile across JFK/LGA/EWR, checked against real data
FLIGHT_MIN_SCHEDULED = 5  # below this, delayed/scheduled is small-denominator noise
# VTEC phenomena codes for warnings that are plausibly driving-hazardous. Excludes marine-only
# (MA, GL, SR, ...) and slow-onset river/areal flood warnings (FL, FA) -- FL alone averages
# ~100h per warning in Chicago and was inflating its NWS-warning rate to 2.75x NYC's before this
# filter. Checked against total warning-hours per city (26.8k Chicago vs 29.0k NYC after this
# filter -- comparable, vs. 220.5k vs 88.5k before). Worth re-checking this list against the
# official NWS VTEC phenomena table before the final report; it's a reasonable first pass, not
# verified code-by-code.
DRIVING_PHENOMENA = "'SV','TO','FF','WS','BZ','IS','WC','HW','EH','XH','EC','DS','SQ'"
# NWS county (xxC) and forecast-zone (xxZ) codes that cover each city. The `city` column only
# names the forecast office, whose area runs well past the city (LOT covers ~20 counties), so
# warnings are kept only if they name one of these. Checked against api.weather.gov zone names.
NWS_UGC = {
    "nyc": "'NYC005','NYC047','NYC061','NYC081','NYC085',"  # the five borough counties
           "'NYZ072','NYZ073','NYZ074','NYZ075','NYZ176','NYZ178'",  # Manhattan, Bronx, Staten Is., Brooklyn, N/S Queens
    "chicago": "'ILC031','ILZ014','ILZ103','ILZ104'",  # Cook County; zone ILZ014 split into Northern/Central Cook (103/104) in 2020-03
}


def connect():
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    con.execute("INSTALL spatial; LOAD spatial;")
    return con


def lag_table(con, city):
    src = PROC_ML / f"{city}_labeled.parquet"
    if not src.exists():
        sys.exit(f"{city}: {src} not found, run label.py first")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE lagged AS
        SELECT *,
               lag(pickups, 1) OVER w AS pickup_lag1,
               lag(pickups, 24) OVER w AS pickup_lag24,
               lag(pickups, 168) OVER w AS pickup_lag168,
               avg(pickups) OVER w24 AS pickup_roll24_mean,
               stddev_samp(pickups) OVER w24 AS pickup_roll24_std,
               avg(pickups) OVER w168 AS pickup_roll168_mean,
               stddev_samp(pickups) OVER w168 AS pickup_roll168_std
        FROM read_parquet('{src}')
        WINDOW
            w AS (PARTITION BY zone ORDER BY hour),
            w24 AS (PARTITION BY zone ORDER BY hour ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING),
            w168 AS (PARTITION BY zone ORDER BY hour ROWS BETWEEN 168 PRECEDING AND 1 PRECEDING)""")


def weather_table(con, city):
    station = WEATHER_STATION[city]
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE weather_hourly AS
        WITH obs AS (
            SELECT date_trunc('hour', valid_local) AS hour,
                   avg(tmpf) AS temp_f, max(sknt) AS wind_kt, min(vsby) AS vis_mi,
                   bool_or(wxcodes ILIKE '%SN%' OR wxcodes ILIKE '%FZ%' OR wxcodes ILIKE '%TS%')
                       AS has_severe_code
            FROM read_parquet('{EXPORT}/weather_asos.parquet')
            WHERE city = '{city}' AND station = '{station}'
            GROUP BY hour),
        flagged AS (
            SELECT hour, temp_f, wind_kt, vis_mi,
                   coalesce(has_severe_code, false) OR vis_mi < 1 OR wind_kt >= 25
                       AS severe_weather_flag
            FROM obs)
        SELECT hour,
               lag(temp_f, 1) OVER (ORDER BY hour) AS wx_temp_f_lag1,
               lag(wind_kt, 1) OVER (ORDER BY hour) AS wx_wind_kt_lag1,
               lag(vis_mi, 1) OVER (ORDER BY hour) AS wx_vis_mi_lag1,
               severe_weather_flag
        FROM flagged""")


def nws_table(con, city):
    tz = TZ[city]
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE nws_hourly AS
        WITH warnings AS (
            -- utc_* are naive UTC: tag them as UTC, then convert to naive local wall-clock time
            -- (a single timezone(tz, ts) would read them as local time, 4-5 h late).
            SELECT DISTINCT timezone('{tz}', timezone('UTC', utc_issue)) AS start_local,
                   timezone('{tz}', timezone('UTC', utc_expire)) AS end_local
            FROM read_parquet('{EXPORT}/nws_warnings.parquet')
            WHERE city = '{city}' AND significance = 'W' AND phenomena IN ({DRIVING_PHENOMENA})
              AND gtype = 'C' AND ugc IN ({NWS_UGC[city]})),
        hours AS (SELECT DISTINCT hour FROM lagged),
        flagged AS (
            SELECT h.hour, bool_or(w.start_local IS NOT NULL) AS nws_warning_flag
            FROM hours h
            LEFT JOIN warnings w ON h.hour >= w.start_local AND h.hour < w.end_local
            GROUP BY h.hour)
        SELECT hour, nws_warning_flag,
               lag(nws_warning_flag, 1) OVER (ORDER BY hour) AS nws_warning_active_lag1
        FROM flagged""")


def events_table(con, city):
    # events.city / event_venues.city hold the venue's actual municipality (Queens, Newark, ...),
    # not the metro -- `metro` is the column that actually takes 'nyc' / 'chicago' values.
    #
    # expected_attendance: what a forecaster could know beforehand, since actual attendance is
    # reported after the event. Average attendance of earlier events with the same crowd key
    # (league + home team for games, e.g. "nba | New York Knicks"; league + venue otherwise),
    # over calendar days strictly before this event's date. Zero-attendance games (no fans,
    # 2020-21) are skipped. NULL when there's no history (first game, concerts, US Open).
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE events_expected AS
        WITH e AS (
            SELECT *, CASE WHEN source = 'espn' AND title LIKE '% at %'
                           THEN league || ' | ' || split_part(title, ' at ', 2)
                           ELSE league || ' | ' || venue_name END AS crowd_key
            FROM read_parquet('{EXPORT}/events.parquet') WHERE metro = '{city}')
        SELECT *, avg(CASE WHEN attendance > 0 THEN attendance END) OVER (
                      PARTITION BY crowd_key ORDER BY start_local::DATE
                      RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL 1 DAY PRECEDING) AS expected_attendance
        FROM e""")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE events_attributed AS
        SELECT zone, start_local, end_local_est::TIMESTAMP AS end_local, attendance, expected_attendance
        FROM events_expected
        WHERE zone IS NOT NULL
        UNION ALL
        SELECT unnest(v.entry_zones)::SMALLINT AS zone, e.start_local,
               e.end_local_est::TIMESTAMP AS end_local, e.attendance, e.expected_attendance
        FROM events_expected e
        JOIN read_parquet('{EXPORT}/event_venues.parquet') v
            ON v.venue_name = e.venue_name AND v.metro = e.metro
        WHERE e.zone IS NULL AND v.entry_zones IS NOT NULL""")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE events_zone_hour AS
        WITH hz AS (SELECT DISTINCT zone, hour FROM lagged)
        SELECT hz.zone, hz.hour,
               count(ev.zone) > 0 AS event_flag,
               coalesce(sum(ev.attendance), 0) AS event_attendance,
               coalesce(sum(ev.expected_attendance), 0) AS expected_attendance,
               bool_or(coalesce(ev.attendance, 0) >= {MAJOR_EVENT_ATTENDANCE}) AS major_event_flag
        FROM hz
        LEFT JOIN events_attributed ev
            ON ev.zone = hz.zone
           AND hz.hour >= ev.start_local - INTERVAL 1 HOUR
           AND hz.hour < ev.end_local + INTERVAL 1 HOUR
        GROUP BY hz.zone, hz.hour""")


def transit_table_nyc(con):
    # Routine subway schedule-change alerts touch several routes every hour (median is 4
    # routes affected) -- the plan's original ">= 3 routes" default was checked against the
    # actual distribution and was far too lenient (68% of hours). The next rule, ">= 3
    # suspension updates OR >= 10 routes affected", still fired in 8.3% of hours (9.9% of the
    # 2024 calibrate year), mostly through the routes condition, and that condition is not
    # comparable over time: MTA's alert system changed in 2020-04 and the newer alerts list
    # more routes, so ">= 10 routes" swings from 0.8% of hours (2019) to 14.1% (2020).
    # Suspension updates alone at >= 3 (~98.5th percentile) flag 2.3% of hours, in line with
    # severe weather (~2%). 2019 is still lower (0.6%): the 2012-2020 archive's labels were
    # coarser. See README Week 3.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE transit_hourly AS
        SELECT hour,
               coalesce(sum(n_suspension_updates), 0) >= 3 AS transit_disruption_flag
        FROM read_parquet('{EXPORT}/mta_alerts_hour.parquet')
        WHERE agency = '{NYC_SUBWAY_AGENCY}'
        GROUP BY hour""")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE transit_zone_hour AS
        SELECT zone, hour,
               lag(entries, 1) OVER (PARTITION BY zone ORDER BY hour) AS subway_entries_lag1,
               lag(transfers, 1) OVER (PARTITION BY zone ORDER BY hour) AS subway_transfers_lag1
        FROM read_parquet('{EXPORT}/mta_zone_hour.parquet')""")


def transit_table_chicago(con):
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE cta_anomaly AS
        WITH d AS (
            SELECT zone, date, rides, dayofweek(date) AS dow,
                   avg(rides) OVER (
                       PARTITION BY zone, dayofweek(date) ORDER BY date
                       ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) AS baseline
            FROM read_parquet('{EXPORT}/cta_area_day.parquet'))
        SELECT zone, date,
               lag(rides, 1) OVER (PARTITION BY zone ORDER BY date) AS cta_rides_prevday,
               baseline IS NOT NULL AND rides < 0.5 * baseline AS transit_disruption_flag
        FROM d""")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE transit_zone_hour AS
        SELECT l.zone, l.hour, a.cta_rides_prevday, a.transit_disruption_flag
        FROM (SELECT DISTINCT zone, hour FROM lagged) l
        LEFT JOIN cta_anomaly a ON a.zone = l.zone AND a.date = l.hour::DATE""")


def flights_table(con, city):
    airports = [a for a in AIRPORT_ZONE
                if con.execute(f"""SELECT 1 FROM read_parquet('{EXPORT}/bts_airport_hour.parquet')
                                    WHERE airport = '{a}' AND city = '{city}' LIMIT 1""").fetchone()]
    cases = " ".join(f"WHEN '{a}' THEN {AIRPORT_ZONE[a]}" for a in airports)
    # At low scheduled-flight counts (overnight hours), delayed/scheduled is a noisy ratio over
    # a tiny denominator (p99 exceeds 1.0 -- more "delayed" arrivals counted than flights
    # scheduled that hour). MIN_SCHEDULED filters those out before the flag is allowed to fire;
    # the 0.5 cutoff is the real ~90th percentile across NYC's three airports, not a guess.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE flights_zone_hour AS
        WITH f AS (
            SELECT CASE airport {cases} END::SMALLINT AS zone, hour,
                   dep_scheduled + arr_scheduled AS flight_sched_count,
                   (dep_delayed15 + arr_delayed15)::DOUBLE
                       / nullif(dep_scheduled + arr_scheduled, 0) AS delay_rate
            FROM read_parquet('{EXPORT}/bts_airport_hour.parquet')
            WHERE city = '{city}')
        SELECT zone, hour, flight_sched_count,
               lag(delay_rate, 1) OVER (PARTITION BY zone ORDER BY hour) AS flight_delay_rate_lag1,
               flight_sched_count >= {FLIGHT_MIN_SCHEDULED}
                   AND coalesce(delay_rate, 0) >= {FLIGHT_DELAY_RATE} AS flight_delay_flag
        FROM f""")


def assemble_nyc(con):
    out = PROC_ML / "nyc_features.parquet"
    con.execute(f"""
        COPY (
            SELECT l.*,
                   w.wx_temp_f_lag1, w.wx_wind_kt_lag1, w.wx_vis_mi_lag1,
                   n.nws_warning_active_lag1,
                   e.event_flag, e.event_attendance, e.expected_attendance,
                   t.subway_entries_lag1, t.subway_transfers_lag1,
                   fl.flight_sched_count, fl.flight_delay_rate_lag1,
                   coalesce(w.severe_weather_flag, false) AS severe_weather_flag,
                   coalesce(n.nws_warning_flag, false) AS nws_warning_flag,
                   coalesce(e.major_event_flag, false) AS major_event_flag,
                   coalesce(th.transit_disruption_flag, false) AS transit_disruption_flag,
                   coalesce(fl.flight_delay_flag, false) AS flight_delay_flag,
                   coalesce(w.severe_weather_flag, false)
                       OR coalesce(n.nws_warning_flag, false)
                       OR coalesce(e.major_event_flag, false)
                       OR coalesce(th.transit_disruption_flag, false)
                       OR coalesce(fl.flight_delay_flag, false) AS extreme_flag
            FROM lagged l
            LEFT JOIN weather_hourly w ON w.hour = l.hour
            LEFT JOIN nws_hourly n ON n.hour = l.hour
            LEFT JOIN transit_hourly th ON th.hour = l.hour
            LEFT JOIN events_zone_hour e ON e.zone = l.zone AND e.hour = l.hour
            LEFT JOIN transit_zone_hour t ON t.zone = l.zone AND t.hour = l.hour
            LEFT JOIN flights_zone_hour fl ON fl.zone = l.zone AND fl.hour = l.hour
            ORDER BY l.hour, l.zone
        ) TO '{out}' (FORMAT parquet, COMPRESSION zstd)""")
    print(f"nyc: wrote {out}", flush=True)


def assemble_chicago(con):
    out = PROC_ML / "chicago_features.parquet"
    con.execute(f"""
        COPY (
            SELECT l.*,
                   w.wx_temp_f_lag1, w.wx_wind_kt_lag1, w.wx_vis_mi_lag1,
                   n.nws_warning_active_lag1,
                   e.event_flag, e.event_attendance, e.expected_attendance,
                   t.cta_rides_prevday,
                   fl.flight_sched_count, fl.flight_delay_rate_lag1,
                   coalesce(w.severe_weather_flag, false) AS severe_weather_flag,
                   coalesce(n.nws_warning_flag, false) AS nws_warning_flag,
                   coalesce(e.major_event_flag, false) AS major_event_flag,
                   coalesce(t.transit_disruption_flag, false) AS transit_disruption_flag,
                   coalesce(fl.flight_delay_flag, false) AS flight_delay_flag,
                   coalesce(w.severe_weather_flag, false)
                       OR coalesce(n.nws_warning_flag, false)
                       OR coalesce(e.major_event_flag, false)
                       OR coalesce(t.transit_disruption_flag, false)
                       OR coalesce(fl.flight_delay_flag, false) AS extreme_flag
            FROM lagged l
            LEFT JOIN weather_hourly w ON w.hour = l.hour
            LEFT JOIN nws_hourly n ON n.hour = l.hour
            LEFT JOIN events_zone_hour e ON e.zone = l.zone AND e.hour = l.hour
            LEFT JOIN transit_zone_hour t ON t.zone = l.zone AND t.hour = l.hour
            LEFT JOIN flights_zone_hour fl ON fl.zone = l.zone AND fl.hour = l.hour
            ORDER BY l.hour, l.zone
        ) TO '{out}' (FORMAT parquet, COMPRESSION zstd)""")
    print(f"chicago: wrote {out}", flush=True)


def features_city(con, city):
    lag_table(con, city)
    weather_table(con, city)
    nws_table(con, city)
    events_table(con, city)
    flights_table(con, city)
    if city == "nyc":
        transit_table_nyc(con)
        assemble_nyc(con)
    else:
        transit_table_chicago(con)
        assemble_chicago(con)
    # Persist calendar annotations for subgroup analysis without changing model allowlists.
    from calendar_flags import build_for_city
    build_for_city(city)


def main():
    con = connect()
    for city in sys.argv[1:] or list(CITIES):
        features_city(con, city)


if __name__ == "__main__":
    main()
