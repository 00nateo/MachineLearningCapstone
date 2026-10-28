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
  VTEC table — worth a second look before the final report. **Corrected
  again (2026-10-28):** (1) warning times were read as local instead of UTC,
  shifting every window 4-5 h late — now converted with
  `timezone(tz, timezone('UTC', ts))`; (2) `city` only names the NWS office,
  whose area runs far beyond the city (LOT covers ~20 counties), so warnings
  are now kept only if they name a county/zone covering the city (`NWS_UGC`
  in `features.py`: NYC's five borough counties + their six zones; Cook
  County + its zones). Check: the Jan 2022 blizzard is now flagged
  01-28 19:00 -> 01-29 16:00 local (was midnight-to-midnight on the 29th).
  Rate: ~1.3% (NYC) / ~1.6% (Chicago) of hours, 2019-01 -> 2026-08 (was
  ~2.5% / ~3.2% before these two fixes).
- **Major event**: within [event start - 1h, end + 1h] where attendance >=
  15,000. Verified directly against a known doubleheader weekend at Madison
  Square Garden (zone 186, not the Midtown zones 161-164 it's easy to guess
  — checked by venue-name lookup): the 2023-04-22 Rangers game (18,006
  attendance) and 2023-04-23 Knicks game (19,812 attendance) both fire in
  exactly their start-1h/end+1h windows.
- **Transit disruption**: NYC via `mta_alerts_hour` (NYCT Subway agency) —
  **corrected** from ">=1 suspension update or >=3 routes affected" after
  finding the median hour already has 4 routes affected from routine
  schedule-change alerts (that threshold fired 68.5% of the time). Then
  >=3 suspension updates OR >=10 routes affected — **corrected again
  (2026-10-28)** because that still fired in 8.3% of hours, mostly through
  the routes condition, which isn't comparable over time (MTA's alert
  system changed in 2020-04 and newer alerts list more routes: ">=10
  routes" went from 0.8% of hours in 2019 to 14.1% in 2020). **Now >=3
  suspension updates only** (~98.5th percentile): 2.3% of hours, in line
  with severe weather. 2019 still flags less (0.6%) because the old
  archive's labels were coarser. Chicago has no alerts table; approximate with a CTA
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
  sampling. Final rate: **~5.1% of NYC hours, ~8.4% of Chicago hours** (was
  11.9% / 9.8% originally, then 10.9% / 8.4% after the NWS fixes, then 5.1%
  for NYC after the transit fix). NYC's composite is now driven by subway
  suspensions (2.3%) and severe weather (~1.8%) rather than routine alerts.

### How the composite `extreme_flag` works

`extreme_flag` is TRUE for a zone-hour when **any** of the five component
flags is TRUE for that zone in that hour (a logical OR). Some components
apply to the whole city at once, others only to particular zones:

| Component | Scope | Fires for |
|---|---|---|
| Severe weather | citywide | every zone in that hour |
| NWS warning | citywide | every zone in that hour |
| Transit disruption | NYC: citywide (subway alerts); Chicago: per community area per day (CTA anomaly) | NYC: every zone; Chicago: that area, all day |
| Major event (attendance >= 15,000) | per zone | the venue's zone, or for out-of-zone venues (MetLife, UBS Arena, ...) their entry zones, from start - 1 h to end + 1 h |
| Flight delays | per zone | the airport's zone only (JFK 132, LGA 138, EWR 1, ORD 76, MDW 56) |

So a snowstorm hour flags all 263 NYC zones, while a Knicks game flags only
MSG's zone. The flag is computed on the **actual hour h**, not lagged, so it
describes what really happened. It is never a model input. It does three
jobs:

1. **Training sampling:** every day with an extreme hour is always kept,
   and extreme rows are up-weighted (~3x).
2. **Evaluation subsets:** metrics are reported separately for extreme vs
   normal hours. The failure criterion is the model abstaining on most
   extreme cases.
3. **Mondrian conformal:** separate calibration thresholds for extreme vs
   normal hours.

Gap to fix: `features.py` writes only `extreme_flag` (and `event_flag`), not
the five component flags. The per-condition evaluation (weather vs NWS vs
transit vs flights vs events) needs them, so `assemble_nyc` /
`assemble_chicago` should also output `severe_weather_flag`,
`nws_warning_flag`, `transit_disruption_flag`, `major_event_flag` and
`flight_delay_flag`.

## Case-study days

Specific hectic days to examine one by one on the test set, beyond the
aggregate extreme-condition subsets: how the model and the abstention layer
behave hour by hour around each. All fall in fold 3's test period (2025+).
Add new candidates here with their dates as they come up.

- **Knicks win the 2026 NBA title — 2026-06-13.** Game 5 of the Finals,
  away at San Antonio (Frost Bank Center), tip-off 20:30 ET; Knicks won
  94-90 to take the series 4-1. Look at NYC the night of 06-13 into the
  early hours of 06-14 (watch parties, street celebrations, especially
  around MSG/Midtown), plus the championship parade (date not recorded
  yet — add it). Related: the two home Finals games at MSG, 2026-06-08
  (Game 3, loss) and 2026-06-10 (Game 4, win), which are in `events.parquet`.
  The clinching game itself is *not* in `events.parquet` (only games at
  NYC/Chicago-area venues are kept), so the model gets no event signal for
  that night — a good test of how it handles an unscheduled-looking surge.
- **2026 FIFA World Cup matches in the NY/NJ area** (MetLife Stadium,
  including the final), plus fan-zone / watch-party nights. Dates still to
  list. World Cup games are not in the event data yet (would need the ESPN
  `soccer/fifa.world` league added to `fetch_sports.py`; MetLife would then
  map to the Penn Station / Port Authority entry zones).
- **UN General Assembly high-level week (NYC, every September).** Heads of
  state arrive, with large street closures and motorcade "frozen zones"
  around UN headquarters (First Ave, roughly E 42nd-48th St, Turtle Bay /
  Midtown East) and across Midtown. The UNGA80 general debate opened
  2025-09-23; 2026 dates to confirm. Unlike the one-offs above, it recurs
  yearly: earlier editions fall in training (2019, 2021, 2022; 2020 was
  virtual and is excluded anyway), early-stop (2023) and calibration (2024),
  so this tests whether the model learned a recurring disruption. Not in the
  event data; the closures could be marked from NYC's permitted-event /
  street-closure records or by date.
- **Taylor Swift's wedding (NYC).** Reported to cause large street
  closures. Date and location to fill in, then pick the zones to watch.

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
- **Feature availability at forecast time (proposed "deployable" ablation).**
  The 1-hour lag only matches reality for sources published in real time.
  Observed publication delays as of the 2026-10-27 fetch: weather (METAR) and
  NWS warnings are live; MTA hourly ridership ran ~2 weeks behind; CTA daily
  ridership ~3 months; BTS flights and TLC/Chicago trip records ~2 months
  (live pickup counts only exist inside the platform itself). So
  `subway_entries_lag1`, `subway_transfers_lag1` and `cta_rides_prevday` are
  "oracle" features nobody has at forecast time, and `event_attendance` is
  reported after the game. Keep k = 1 as the core (a platform-side forecaster
  has live pickups), but run one ablation that lags each feature by its real
  publication delay (e.g. subway entries from the same hour 2 weeks earlier,
  CTA from ~90 days earlier) and swaps attendance for the venue's typical
  attendance. The F1 gap between the two runs measures how much the
  optimistic features are worth. **Result (2026-10-28, NYC fold 3):** the
  deployable model (ridership and attendance dropped) scores F1 0.838 on
  2024 vs 0.858 for the full model and 0.840 for the 4-week average, so it
  no longer beats the baseline. Subway entries act as a live-activity proxy
  worth ~2 F1 points. Report both variants. Tried next: replacing attendance
  with `expected_attendance` (each team's / venue's average from earlier
  games only, skipping 2020-21 zero-attendance games) made no difference
  (F1 0.838 -> 0.837; major-event hours 0.799 -> 0.801). Crowd size isn't
  what the model misses on event hours.
- **Surge label ("unexpected hotspots") as a second target.** The top-10% label
  is mostly static (20 zones give 56% of hotspot hours). Proposed alternative:
  pickups vs the zone's own normal-conditions baseline (8-week same-hour,
  same-weekday median, excluding extreme hours and holidays), at 1.25x / 1.5x /
  2x / 3x plus a count-aware z >= 3. Design, results and recommendation in
  `SURGE_LABEL_DESIGN.md`; labels built by `scripts/surge_label.py`. First
  model (`scripts/train_surge.py`, NYC fold 3): AUC-PR 0.46, F1 0.45 vs 0.34
  for the best baseline ("surged last hour"); the deployable model matches the
  full one. For the group to decide how far to take it after the milestone.

## Repo-level build plan

New scripts under `scripts/`, following the existing `export.py` /
`aggregate.py` DuckDB style:

- `label.py` **(done)** — decile-rank hotspot label + fold-tag column, per
  city, from `*_zone_hour.parquet`.
- `features.py` **(done)** — lag/rolling features computed on the full
  series, plus weather/NWS/event/transit/flight joins and the composite
  extreme flag. Outputs one feature table per city.
- `sample.py` **(done)** — applies the stratified day-sampling/weighting
  rule to the train split only.
- `baseline.py` **(done)** — seasonal-last-week and 4-week-average
  baselines, ranked to top decile.
- `train.py` **(done)** — LightGBM per city per fold, early-stopped,
  fixed seeds, scored on AUC-PR and top-decile precision/recall.
- `conformal.py` **(global mode done; Mondrian pending)** — conformal layer:
  fit on calibrate, sweep alpha 0.01-0.20, track singleton/full/empty
  prediction sets separately. Week 4's global-threshold version measures on
  Jul-Dec 2024 with thresholds from Jan-Jun 2024 (test untouched).
- `evaluate.py` **(partly done: intervals, calibration, per-condition and
  monthly metrics on calibrate)** — still to add: accuracy-when-confident, abstention rate,
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
  above for the numbers). Final `extreme_flag` rate: 11.9% NYC, 9.8% Chicago;
  10.9% / 8.4% after the 2026-10-28 NWS timezone + county-filter fixes.
- **Week 3 — DONE. Baselines + first model.** `sample.py`, `baseline.py`
  (both baselines) + `train.py` on NYC fold 3, scored on calibrate. LightGBM
  top-decile F1 0.857 vs 0.840 (4-week average) and 0.812 (seasonal naive);
  AUC-PR 0.940. Details, decisions and edge cases in README.md (Week 3).
- **Week 4 — Simple abstention + one subgroup + write the milestone
  report.** A single global-threshold abstain rule (full Mondrian comes
  later) and one subgroup table (normal vs. "any extreme"). **Abstention +
  subgroup tables DONE (2026-10-28)**: at alpha = 0.05 coverage 0.947 (target
  0.95) but hotspot coverage only 0.79 and 7.0% of true hotspots confidently
  missed; on major-event hours coverage drops to 0.75 with 15% abstention.
  Details in README.md (Week 4). Milestone report still to write. In parallel,
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
