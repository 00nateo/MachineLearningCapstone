"""Auditable local-date calendar flags for NYC/Chicago, independent of model labels.

General elections: annual November Election Day in NYC; even-year November
general elections in Chicago, plus verified 2019/2023 municipal rounds.
Primaries, special elections, and early-voting periods are not covered yet.
The in-person NYC Marathon is excluded in 2020. Flags overlap deliberately.
Federal observed days mean weekday observance for a standard Mon-Fri schedule.
All rules here cover 2018-2026; verify additions before extending the range.
"""
from datetime import date, timedelta
from pathlib import Path
import uuid

import duckdb
import holidays
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC_ML = ROOT / "data" / "processed_ml"
SOURCES = {
    "federal": "https://www.opm.gov/policy-data-oversight/pay-leave/federal-holidays/",
    "nyc_elections": "https://www.nysenate.gov/legislation/laws/ELN/8-100",
    "federal_elections": "https://www.eac.gov/voters/election-day-resources",
    "chicago_elections": "https://chicagoelections.gov/elections/results",
    "nyc_marathon": "https://www.nyrr.org/tcsnycmarathon",
    "marathon_cancellation": "https://www.nyrr.org/tcsnycmarathon/pages/2020-cancellation",
    "dst": "https://www.nist.gov/pml/time-and-frequency-division/popular-links/daylight-saving-time-dst",
}
FEDERAL_NAMES = {
    "New Year's Day": "new_years_day",
    "Martin Luther King Jr. Day": "mlk_day",
    "Washington's Birthday": "washingtons_birthday",
    "Memorial Day": "memorial_day",
    "Juneteenth National Independence Day": "juneteenth",
    "Independence Day": "independence_day",
    "Labor Day": "labor_day",
    "Columbus Day": "columbus_day",
    "Veterans Day": "veterans_day",
    "Thanksgiving Day": "thanksgiving",
    "Christmas Day": "christmas",
}
CHICAGO_MUNICIPAL = {
    date(2019, 2, 26): "Chicago municipal general election",
    date(2019, 4, 2): "Chicago municipal runoff election",
    date(2023, 2, 28): "Chicago municipal general election",
    date(2023, 4, 4): "Chicago municipal runoff election",
}
EXTRA_FLAGS = ["federal_holiday_actual", "federal_holiday_observed", "federal_holiday",
               "halloween", "halloween_eve", "halloween_next_day", "thanksgiving_eve",
               "thanksgiving_next_day", "christmas_eve", "new_years_eve", "nyc_marathon",
               "nyc_marathon_eve", "nyc_marathon_next_day", "general_election",
               "federal_election", "presidential_election", "midterm_election", "local_election",
               "any_election", "dst_spring", "dst_fall", "any_named_day", "any_special_day"]
FLAG_COLUMNS = [f"calendar_{name}" for name in dict.fromkeys([*FEDERAL_NAMES.values(), *EXTRA_FLAGS])]
AGGREGATES = {"calendar_any_named_day", "calendar_any_special_day"}


def nth_weekday(year, month, weekday, n):
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def build_calendar(city, start, end):
    start, end = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    if city not in ("nyc", "chicago") or start > end:
        raise ValueError("Need city nyc/chicago and start <= end")
    if start.year < 2018 or end.year > 2026:
        raise ValueError("Calendar rules audited for 2018-2026 only; verify events before extending them")
    dates = list(pd.date_range(start, end, freq="D").date)
    df = pd.DataFrame(False, index=pd.Index(dates, name="date"), columns=FLAG_COLUMNS)
    names = {d: set() for d in dates}
    federal_names = {d: set() for d in dates}
    records = []

    def add(day, flag, name, source="", status="active", add_name=True):
        if day not in df.index:
            return
        records.append(dict(city=city, date=day, flag=f"calendar_{flag}", event=name,
                            status=status, source_url=source))
        if status == "active":
            df.at[day, f"calendar_{flag}"] = True
            if add_name:
                names[day].add(name)

    years = range(start.year - 1, end.year + 2)  # cross-year observances, e.g. Dec 31, 2021
    actual = holidays.US(years=years, observed=False, language="en_US")
    observed = holidays.US(years=years, observed=True, language="en_US")
    for day, name in actual.items():
        if day not in df.index:
            continue
        if name not in FEDERAL_NAMES:
            raise ValueError(f"Unmapped US federal holiday: {name}; update the audited mapping")
        add(day, "federal_holiday_actual", name, SOURCES["federal"])
        add(day, FEDERAL_NAMES[name], name, SOURCES["federal"], add_name=False)
        federal_names[day].add(name)
    for day, name in observed.items():
        if day.weekday() >= 5 or day not in df.index:
            continue
        canonical = name.removesuffix(" (observed)")
        if canonical not in FEDERAL_NAMES:
            raise ValueError(f"Unmapped observed US holiday: {name}")
        add(day, "federal_holiday_observed", name, SOURCES["federal"])
        add(day, FEDERAL_NAMES[canonical], canonical, SOURCES["federal"], add_name=False)
        federal_names[day].add(name)
    for year in range(start.year, end.year + 1):
        add(date(year, 10, 31), "halloween", "Halloween")
        add(date(year, 10, 30), "halloween_eve", "Halloween eve")
        add(date(year, 11, 1), "halloween_next_day", "Day after Halloween")
        thanksgiving = nth_weekday(year, 11, 3, 4)
        add(thanksgiving - timedelta(days=1), "thanksgiving_eve", "Thanksgiving eve", SOURCES["federal"])
        add(thanksgiving + timedelta(days=1), "thanksgiving_next_day", "Day after Thanksgiving", SOURCES["federal"])
        add(date(year, 12, 24), "christmas_eve", "Christmas eve")
        add(date(year, 12, 31), "new_years_eve", "New Year's eve")
        election = nth_weekday(year, 11, 0, 1) + timedelta(days=1)
        if city == "nyc" or year % 2 == 0:
            source = SOURCES["nyc_elections"] if city == "nyc" else SOURCES["federal_elections"]
            add(election, "general_election", "General election", source)
            if year % 2 == 0:
                add(election, "federal_election", "Federal general election", source, add_name=False)
                add(election, "presidential_election" if year % 4 == 0 else "midterm_election",
                    "Presidential election" if year % 4 == 0 else "Midterm election", source, add_name=False)
            else:
                add(election, "local_election", "Local general election", source, add_name=False)
        if city == "nyc":
            marathon = nth_weekday(year, 11, 6, 1)
            if year == 2020:
                add(marathon, "nyc_marathon", "NYC Marathon (in-person canceled)",
                    SOURCES["marathon_cancellation"], status="canceled")
            else:
                add(marathon, "nyc_marathon", "NYC Marathon", SOURCES["nyc_marathon"])
                add(marathon - timedelta(days=1), "nyc_marathon_eve", "NYC Marathon eve", SOURCES["nyc_marathon"])
                add(marathon + timedelta(days=1), "nyc_marathon_next_day", "Day after NYC Marathon", SOURCES["nyc_marathon"])
        add(nth_weekday(year, 3, 6, 2), "dst_spring", "DST spring-forward", SOURCES["dst"])
        add(nth_weekday(year, 11, 6, 1), "dst_fall", "DST fall-back", SOURCES["dst"])
    if city == "chicago":
        for day, event in CHICAGO_MUNICIPAL.items():
            add(day, "local_election", event, SOURCES["chicago_elections"])
    df["calendar_federal_holiday"] = df.calendar_federal_holiday_actual | df.calendar_federal_holiday_observed
    df["calendar_any_election"] = df.calendar_general_election | df.calendar_local_election
    df["calendar_any_named_day"] = (df.calendar_federal_holiday | df.calendar_halloween |
                                      df.calendar_nyc_marathon | df.calendar_any_election)
    df["calendar_any_special_day"] = df[[c for c in FLAG_COLUMNS if c not in AGGREGATES]].any(axis=1)
    df["calendar_event_names"] = ["; ".join(sorted(names[d])) for d in dates]
    df["calendar_federal_holiday_names"] = ["; ".join(sorted(federal_names[d])) for d in dates]
    return df.reset_index(), pd.DataFrame(records)


def annotate_features(path, days):
    """Atomically append/refresh only our calendar columns in a local feature file."""
    path = Path(path)
    temporary = path.with_name(f".{path.stem}.calendar-{uuid.uuid4().hex}.parquet")
    try:
        with duckdb.connect() as con:
            columns = [r[0] for r in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()]
            existing = [c for c in days.columns if c != "date" and c in columns]
            exclude = f" EXCLUDE ({', '.join([*existing, '_calendar_row_order'])})"
            con.register("calendar_days", days)
            # DuckDB 1.4 binds COPY's destination parameter before source parameters;
            # quote this generated path as a SQL literal for consistent 1.x/2.x behavior.
            destination = "'" + str(temporary).replace("'", "''") + "'"
            con.execute(f"""COPY (WITH original AS (SELECT *, row_number() OVER () AS _calendar_row_order FROM read_parquet(?))
                            SELECT f.* {exclude}, c.* EXCLUDE (date)
                            FROM original f LEFT JOIN calendar_days c ON f.hour::DATE = c.date
                            ORDER BY f._calendar_row_order)
                            TO {destination} (FORMAT parquet, COMPRESSION zstd)""", [str(path)])
            before = con.execute("SELECT count(*) FROM read_parquet(?)", [str(path)]).fetchone()[0]
            after, missing = con.execute("""SELECT count(*), count(*) FILTER (WHERE calendar_any_special_day IS NULL)
                                            FROM read_parquet(?)""", [str(temporary)]).fetchone()
            if before != after or missing:
                raise ValueError("Calendar annotation changed row counts or left uncovered dates")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_for_city(city, annotate=True, proc_ml=PROC_ML, export=None):
    """Cover the complete demand panel, even if this city's model features aren't built yet."""
    proc_ml = Path(proc_ml)
    export = Path(export) if export is not None else ROOT / "data" / "export"
    features = proc_ml / f"{city}_features.parquet"
    source = features if features.exists() else export / f"{city}_zone_hour.parquet"
    if not source.exists():
        raise FileNotFoundError(f"No feature/demand panel found for {city}")
    with duckdb.connect() as con:
        start, end = con.execute("SELECT min(hour), max(hour) FROM read_parquet(?)", [str(source)]).fetchone()
    days, events = build_calendar(city, start, end)
    proc_ml.mkdir(parents=True, exist_ok=True)
    days.to_parquet(proc_ml / f"{city}_calendar_day.parquet", index=False)
    events.to_csv(proc_ml / f"{city}_calendar_events.csv", index=False)
    if annotate and features.exists():
        annotate_features(features, days)
    return days, events
