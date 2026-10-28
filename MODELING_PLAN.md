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

Reasonable, auditable defaults for a student project — not research-grade
tuned; state this in the report.

- **Severe weather**: METAR at the city's primary station (NYC: LGA or
  Central Park; Chicago: ORD) shows snow / freezing precip / thunderstorm
  code, OR visibility < 1 statute mile, OR sustained wind >= 25 kt.
- **NWS warning**: hour overlaps a Warning-level (not Watch/Advisory) VTEC
  product in `nws_warnings`.
- **Major event**: within [event start - 1h, end + 1h] where attendance >=
  15,000 or top-decile for that venue.
- **Transit disruption**: NYC only, via `mta_alerts_hour` — >=1 suspension
  update or >=3 routes affected. Chicago has no alerts table; approximate
  with a CTA ridership-anomaly proxy (daily entries far below the trailing
  4-week same-weekday average) and flag this explicitly as a weaker proxy.
- **Flight delays**: >=25% of departures or arrivals delayed >=15 min at the
  city's airport(s), applied to the zone(s) containing that airport.
- **Composite "extreme" flag** = union of the above. Used both as the
  stress-test subgroup and as the upweight/always-keep flag in training
  sampling.

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

- `label.py` — decile-rank hotspot label + fold-tag column, per city, from
  `*_zone_hour.parquet`.
- `features.py` — lag/rolling features computed on the full series, plus
  weather/NWS/event/transit/flight joins and the composite extreme flag.
  Outputs one feature table per city.
- `sample.py` — applies the stratified day-sampling/weighting rule to the
  train split only.
- `baseline.py` — seasonal-last-week and 4-week-average baselines, ranked to
  top decile.
- `train.py` — LightGBM per city per fold, early-stopped, fixed seeds,
  scored on AUC-PR and top-decile precision/recall.
- `conformal.py` — Mondrian conformal layer: fit on calibrate, sweep alpha
  0.01-0.20, track singleton/full/empty prediction sets separately.
- `evaluate.py` — accuracy-when-confident, abstention rate, per-subgroup
  breakdown, near-hit and drive-time domain metrics, calibration-vs-test
  coverage gap per fold (+ random-week no-shift reference).

## Week-by-week action plan (relabel to the actual calendar)

Maps to the milestone rubric: Progress from Proposal; Approach/Technical
Rigor; Baseline/Evidence/Evaluation Design; Preliminary Findings and
Interpretation; Completion Plan/Record/Contributions/Writing.

- **Week 1 — Label, split, core lags.** Build `label.py` and the lag part of
  `features.py`, fold-3 only. Check: no leakage across the COVID gap, DST
  -merged hour doesn't duplicate lag rows.
- **Week 2 — Joins + extreme flags.** Finish `features.py` for NYC fold 3:
  weather/NWS/flights and events/transit joins, each with its lag convention
  documented inline (reuse the `CONVENTIONS` block style from `export.py`).
- **Week 3 — Baselines + first model.** `baseline.py` (both baselines) +
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
