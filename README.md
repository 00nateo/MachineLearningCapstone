Rideshare hotspot forecasting (Chicago + NYC) — data pipeline and modeling.

Raw trip/weather/transit/event data is fetched, aggregated to zone-hour
grain, and published to a Hugging Face dataset by the pipeline in
`scripts/`. The modeling plan built on top of that dataset is in
`MODELING_PLAN.md`.

## Progress so far

**Data pipeline** (pre-existing): Chicago + NYC trip downloads, external
data (weather, NWS warnings, flights, transit, events), conversion,
aggregation to zone-hour panels, and export to the HF dataset
(`MLCapstoneRideshare/rideshare-hotspots`).

**Modeling — Week 1 (label + core features, done):**
- Hotspot label: `pickups` above the 90th percentile across zones, per
  (city, hour) — not a rank/ntile, since many night hours have zones tied
  at 0 pickups. Overall rate ~10.2%, both cities.
- Three rolling-origin folds (fold1/fold2/fold3 tags on every row); fold 3
  matches the dataset's own documented train/calibrate/test split. Verified
  no train-fold row leaks into the 2020-03–2021-06 COVID exclusion window.
- Own-zone autoregressive features: lag 1h/24h/168h, trailing 24h/168h
  rolling mean+std, computed on the full continuous hour series before any
  row subsampling (so a kept row's lag never references a hole).

**Modeling — Week 2 (external joins + extreme-condition flag, done):**
- Joined weather (ASOS), NWS warnings, events, transit (MTA/CTA), and
  flight delays (BTS) onto the zone-hour table, each lagged ≥1h unless the
  signal is genuinely known in advance (event schedules, flight schedules).
- Built a composite `extreme_flag` (severe weather OR NWS warning OR major
  event OR transit disruption OR flight delays) used only for stress-test
  evaluation and training-sample weighting — never fed to the model as a
  feature, so it's allowed to use same-hour truth without leaking.
- **Important side note for the report**: several of the plan's first-draft
  thresholds were wrong when checked against real data, and this is worth
  citing as evidence of an actually-validated pipeline, not an assumed one:
  - NYC subway "disruption" at ≥3 routes affected fired 68.5% of the time
    (routine schedule-change alerts touch multiple lines constantly,
    median is 4 routes/hour). Recalibrated to real 95th/99th percentiles.
  - A single NWS warning type (river flood warnings, code `FL`) averages
    ~100 hours per warning in Chicago vs ~14h in NYC, and was inflating
    Chicago's warning rate to 2.75x NYC's for reasons unrelated to driving
    conditions. Restricted the flag to a driving-hazard phenomena list.
  - Flight-delay threshold (≥25% delayed) fired 24-32% of hours and was
    noisy at low-traffic overnight hours (small-denominator ratios exceeding
    1.0). Raised to the real ~90th percentile with a minimum-volume floor.
  - `events.city` / `event_venues.city` hold the venue's actual municipality
    (e.g. "Queens", "Newark"), not `nyc`/`chicago` — the real join key is
    `metro`. Verified fixed against a real Madison Square Garden
    doubleheader weekend (exact attendance figures matched).
  - Airport-to-zone mapping was checked spatially (DuckDB `ST_Contains`
    against each airport's real coordinates) rather than assumed by name —
    Midway Airport is in Garfield Ridge (zone 56), not Clearing.
  - Final `extreme_flag` rate after all fixes: 11.9% of NYC hours, 9.8% of
    Chicago hours — comparable across cities, same ballpark as the ~10%
    hotspot rate.

**Modeling — Week 2 addendum (2026-10-28, NWS fixes + component flags):**
- NWS warning times were being read as local time instead of UTC
  (`timezone(tz, ts)` on a naive UTC timestamp), so every warning window ran
  4-5 h late. Now converted with `timezone(tz, timezone('UTC', ts))`.
  Check: the Jan 2022 blizzard is flagged 01-28 19:00 -> 01-29 16:00 local,
  matching the storm, instead of midnight-to-midnight on the 29th.
- NWS warnings were filtered by forecast office (`city`), whose area runs far
  beyond the city (Chicago's LOT office covers ~20 counties). Now kept only if
  they name a county or forecast zone covering the city (`NWS_UGC` in
  `features.py`; codes checked against api.weather.gov).
- Effect: NWS-warning share of hours 2.5% -> 1.3% (NYC), 3.2% -> 1.6%
  (Chicago); composite `extreme_flag` 11.9% -> 10.9% (NYC), 9.8% -> 8.4%
  (Chicago). Hotspot rates unchanged (10.2% / 10.4%).
- The five component flags (`severe_weather_flag`, `nws_warning_flag`,
  `major_event_flag`, `transit_disruption_flag`, `flight_delay_flag`) are now
  written next to `extreme_flag`, so evaluation can report each condition
  separately. Still never model inputs.
- Edge case for the write-up: the components have different scopes. Weather,
  NWS and NYC subway disruption are citywide (every zone in that hour); major
  events, flight delays and Chicago's CTA anomaly are zone-level. A snowstorm
  flags all 263 NYC zones; a Knicks game flags only MSG's zone.

**Next**: baselines (seasonal-last-week, 4-week average) and a first
LightGBM model on NYC fold 3 — the milestone's required preliminary result.
See `MODELING_PLAN.md` for the full plan, fold definitions, and threshold
justifications.
