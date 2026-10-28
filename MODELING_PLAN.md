# Hotspot model: training plan

Goal: predict which zones will be hotspots (top 10% of pickups in the city that
hour) and add a conformal "abstain" layer, so the model says "not sure" instead
of guessing. The headline question is how reliable it stays during unusual
conditions (bad weather, events, transit outages, flight delays).

## Data

One row per zone x hour.
- NYC: 263 taxi zones, 17.5M rows (`nyc_zone_hour.parquet`).
- Chicago: 77 community areas, 5.3M rows (`chicago_zone_hour.parquet`).
- ~90 features: recent demand, seasonal lags, neighbours, calendar, weather,
  NWS warnings, flights, transit, drive times, events.
- Everything is on the project's Hugging Face dataset, built by this repo's
  pipeline (`scripts/export.py` and friends).

## Label

Hotspot = pickups alone, top decile of `pickups` per (city, hour), computed
directly off `{city}_zone_hour.parquet` with a window function over zones
within each hour. Pay-adjusted demand (fare/driver-pay weighted) is a
deferred stretch goal, not part of the core label — carrying two label
definitions through the whole pipeline isn't worth it before the milestone.

## Timing (no leakage)

- The forecast for hour h is made at the start of h. It can use complete
  counts up to h-1 (lead time k = 1).
- A k = 2 version (stops at h-2) is a later variant, closer to real driving
  lead time.
- Known-in-advance things (calendar, flight schedules, game/concert
  schedules, issued weather warnings) may describe h itself.

## Splits: rolling-origin folds

Expanding-window, three folds. Fold 3 is the headline fold and matches the
dataset card's documented split exactly, so the model-building work never
fights the data provider's own convention.

| Fold | Train (excl. 2020-03 -> 2021-06) | Early-stop | Calibrate | Test |
|---|---|---|---|---|
| 1 | 2019-01 - 2020-02 | 2021-07 - 2021-12 | 2022 | 2023 |
| 2 | 2019-01 - 2021-12 (excl. gap) | 2022 | 2023 | 2024 |
| 3 (headline) | 2019-01 - 2022-12 (excl. gap) | 2023 | 2024 | 2025 - latest (~2026-08) |

Notes:
- Every fold excludes 2020-03 -> 2021-06 (COVID) from training. Fold 1's
  training window is thin (~14 months) because the exclusion eats most of
  2020 — that's an intentional "early period, less data" stress case, not a
  bug.
- Build fold 3 first. Folds 1-2 are rolling-origin robustness checks, done
  after the milestone.
- We don't randomize days across train and test. Neighbouring days leak
  through lag features and would erase the distribution shift we're trying
  to measure.

### Stratified day sampling (train windows only)

Compute every lag/rolling feature over the full continuous hourly series
first, always. Only after that, subsample which (zone, hour) rows are used
as training examples:
- Keep ~20 random calendar days per month per city.
- Always keep every day flagged "extreme" (see below), regardless of the
  random draw.
- Upweight extreme-flagged rows (e.g. 3x) in the GBM objective.

Early-stop / calibrate / test windows use every day, unsampled — held-out
coverage estimates need to be honest there.

## Extreme-condition flags

Implemented in `scripts/features.py`. Initial thresholds were reasonable
guesses; several turned out to be badly miscalibrated once checked against
the real data and were corrected (worth stating this explicitly in the
report as evidence the pipeline was actually validated, not just assumed
correct):

- **Severe weather**: METAR at the city's primary station (NYC: Central
  Park; Chicago: ORD) shows snow / freezing precip / thunderstorm code, OR
  visibility < 1 statute mile, OR sustained wind >= 25 kt. Checked against
  known storms — top severe-weather days include 2021-02-01 (the Feb 2021
  nor'easter), 2018-03-07/03-21 (the "four nor'easters of March 2018"), and
  2022-01-29 (the Jan 2022 blizzard). Rate: ~2.2% of NYC hours. No fix
  needed, this one was fine from the start.
- **NWS warning**: hour overlaps a Warning-level (`significance = 'W'`) VTEC
  product in `nws_warnings`, restricted to a driving-hazard phenomena list
  (`SV, TO, FF, WS, BZ, IS, WC, HW, EH, XH, EC, DS, SQ`) — **corrected**
  after finding that "any Warning-level product" included marine-only
  warnings and slow-onset river/areal flood warnings (`FL`/`FA`); `FL` alone
  averages ~100h per warning in Chicago (vs ~14h in NYC) and was inflating
  Chicago's flag to 2.75x NYC's (220.5k vs 88.5k total warning-hours before
  the filter; 26.8k vs 29.0k after — comparable). This phenomena list is a
  reasonable first pass, not verified code-by-code against the official NWS
  VTEC table — worth a second look before the final report. Rate: ~2.5%
  (NYC) / ~3.2% (Chicago) of hours.
- **Major event**: within [event start - 1h, end + 1h] where attendance >=
  15,000. Verified directly against a known doubleheader weekend at Madison
  Square Garden (zone 186, not the Midtown zones 161-164 it's easy to guess
  — checked by venue-name lookup): the 2023-04-22 Rangers game (18,006
  attendance) and 2023-04-23 Knicks game (19,812 attendance) both fire in
  exactly their start-1h/end+1h windows.
- **Transit disruption**: NYC via `mta_alerts_hour` (NYCT Subway agency) —
  **corrected** from ">=1 suspension update or >=3 routes affected" after
  finding the median hour already has 4 routes affected from routine
  schedule-change alerts (that threshold fired 68.5% of the time). Now
  >=3 suspension updates (~99th percentile) OR >=10 routes affected (~95th
  percentile). Chicago has no alerts table; approximate with a CTA
  ridership-anomaly proxy (daily entries < 0.5x the trailing 4-week
  same-weekday average, zone-day grain broadcast to every hour of that day)
  — this one was fine at first pass (~2.0% of days), flagged as a weaker
  proxy regardless since it can't be sharper than daily.
- **Flight delays**: **corrected** from ">=25% of flights delayed >=15 min"
  (fired 24-32% of hours — not "extreme") to >=50% (the real ~90th
  percentile across JFK/LGA/EWR) with a minimum of 5 scheduled flights in
  that hour (below that, the delayed/scheduled ratio is small-denominator
  noise — p99 of the raw ratio exceeds 1.0 in quiet overnight hours).
  Applied to the zone(s) containing that airport (JFK=132, LGA=138, EWR=1,
  ORD=76, MDW=56 — Chicago's zone IDs found via `ST_Contains` against each
  airport's real coordinates, not by name/memory; Midway sits in Garfield
  Ridge, not Clearing).
- **Composite "extreme" flag** = union of the above. Used both as the
  stress-test subgroup and as the upweight/always-keep flag in training
  sampling. Final rate: ~11.9% of NYC hours, ~9.8% of Chicago hours —
  comparable across cities, in the same ballpark as the ~10% hotspot rate.

## Steps

1. **Baselines**: seasonal-last-week (same hour, same day last week) and a
   4-week same-hour average, both ranked to the top 10%.
2. **LightGBM per city**: categorical features, early stopping, a small
   hyperparameter search, fixed seeds. Scored on AUC-PR and top-decile
   precision/recall.
3. **Mondrian conformal layer**: fit on the calibration year, with separate
   thresholds per condition (normal vs. extreme).
   - The prediction set `{hotspot}` or `{not}` is a confident call; both
     classes passing or neither passing means abstain.
   - Track empty-set and full-set abstentions separately — they mean
     different things (empty = actively conflicted / likely
     out-of-distribution; full = genuinely ambiguous).
   - Sweep the confidence level alpha from 0.01 to 0.20 to trace the
     confidence-vs-abstention tradeoff curve.
4. **Evaluation on test**:
   - Metrics: accuracy when confident, abstention rate, per-class coverage.
   - Subgroup breakdown: overall and for each extreme-condition subset
     (weather, NWS warnings, transit, flights, events), plus inside vs.
     outside the NYC congestion-pricing zone. Note explicitly that
     congestion pricing (effective Jan 2025) lands right at the fold-3
     train/test boundary — a confound to disentangle from weather/event
     shift, not ignore.
   - Two relaxed-hit metrics, kept distinct:
     - *Near-hit* (cheap, graph-based): predicted-hotspot set plus its
       `zone_adjacency` neighbours covers the true hotspot.
     - *Domain coverage metric* (the real one from the proposal): % of
       pickups occurring in a zone whose drive time (from
       `{city}_drive_times`) to the nearest predicted hotspot exceeds 5-10
       minutes.
   - Failure criterion: the model abstains on most extreme-condition cases.
5. **Shift diagnostics**: compare calibration-promised coverage vs. realized
   test coverage in each fold. A random-week split (no temporal shift) is
   the no-shift reference the real gap is measured against. The gap between
   promised and achieved coverage is the project's headline result, not a
   caveat to explain away — conformal prediction's guarantee assumes
   exchangeability, and testing exactly where that assumption breaks is the
   point.

## Open decisions (deferred past the milestone)

- Pay-adjusted demand label variant.
- Final numeric tuning of the extreme-condition thresholds above.
- Whether to add a longer horizon for whole-shift recommendations (beyond
  k = 1 / k = 2).

## Repo-level build plan

New scripts under `scripts/`, following the existing `export.py` /
`aggregate.py` DuckDB style:

- `label.py` **(done)** — decile-rank hotspot label + fold-tag column, per
  city, from `*_zone_hour.parquet`.
- `features.py` **(done)** — lag/rolling features computed on the full
  series, plus weather/NWS/event/transit/flight joins and the composite
  extreme flag. Outputs one feature table per city.
- `sample.py` **(pending)** — applies the stratified day-sampling/weighting
  rule to the train split only.
- `baseline.py` **(pending)** — seasonal-last-week and 4-week-average
  baselines, ranked to top decile.
- `train.py` **(pending)** — LightGBM per city per fold, early-stopped,
  fixed seeds, scored on AUC-PR and top-decile precision/recall.
- `conformal.py` **(pending)** — Mondrian conformal layer: fit on calibrate,
  sweep alpha 0.01-0.20, track singleton/full/empty prediction sets
  separately.
- `evaluate.py` **(pending)** — accuracy-when-confident, abstention rate,
  per-subgroup breakdown, near-hit and drive-time domain metrics,
  calibration-vs-test coverage gap per fold (+ random-week no-shift
  reference).

## Week-by-week action plan (relabel to the actual calendar)

Maps to the milestone rubric: Progress from Proposal; Approach/Technical
Rigor; Baseline/Evidence/Evaluation Design; Preliminary Findings and
Interpretation; Completion Plan/Record/Contributions/Writing.

- **Week 1 — DONE.** Label, split, core lags. `label.py` (hotspot label +
  fold1/fold2/fold3 tags) and the lag part of `features.py`, run for both
  cities and all three folds (ahead of the fold-3-only minimum). Verified: no
  `train`-tagged row falls inside the COVID gap in any fold; no duplicate
  (zone, hour) keys; lag NULLs land only at each zone's series start;
  hotspot rate ~10.2%.
- **Week 2 — DONE.** Joins + extreme flags. `features.py` extended with
  weather/NWS/events/transit/flights joins and the composite `extreme_flag`,
  run for both cities. Three threshold bugs found and fixed by checking
  against real data instead of trusting first-draft defaults (transit
  disruption, NWS phenomena, flight delay rate — see Extreme-condition flags
  above for the numbers). Final `extreme_flag` rate: 11.9% NYC, 9.8% Chicago.
- **Week 3 — NEXT. Baselines + first model.** `baseline.py` (both baselines) +
  `train.py` on NYC fold 3, scored on calibrate. This is the required
  preliminary result.
- **Week 4 — Simple abstention + one subgroup + write the milestone
  report.** A single global-threshold abstain rule (full Mondrian comes
  later) and one subgroup table (normal vs. "any extreme"). In parallel,
  draft the report: revised thesis sentence and scope changes since the
  proposal; methods section with the label/fold/feature spec above; baseline
  + model table (calibrate-set numbers, stated as preliminary/single
  -fold/NYC-only); explicit "what's unresolved" (Chicago not yet run,
  Mondrian/alpha-sweep/folds 1-2 pending); the next stress condition to
  study (name one specific thing, e.g. NWS-warning hours); draft
  class-survey case summary + comparison-matrix entry; contributions section
  with actual weeks-1-4 work per person; link to this repo.
  **MILESTONE DUE end of week 4.**
- **Week 5 — Mondrian conformal + alpha sweep**, empty-vs-full tracking, on
  NYC fold 3.
- **Week 6 — Chicago fold 3**, mirroring weeks 1-3 (reuse scripts, swap city
  arg, build the CTA-anomaly transit proxy).
- **Week 7 — Folds 1 & 2** (NYC at minimum) + shift diagnostics:
  calibration-vs-test coverage gap per fold, random-week no-shift reference
  as control.
- **Week 8 — Domain metrics**: drive-time coverage, near-hit,
  congestion-pricing in/out split (confound-aware framing). k = 2 lead-time
  variant if time remains — optional, not required.
- **Week 9 — Final write-up**: pull every week's numbers into the final
  report, polish figures, finalize contributions, clean up code/README for
  the artifact link.
