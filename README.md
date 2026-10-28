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

**Modeling — Week 3 (baselines + first model, NYC fold 3, done 2026-10-28):**
- Scripts: `sample.py` (training-day sampling and weights), `baseline.py`
  (both baselines plus the shared scoring code), `train.py` (LightGBM), run
  together by `sbatch --account=$ACCOUNT slurm/09_train.sbatch --city nyc
  --fold fold3` (re-runs `features.py` first).
- Results on the calibrate year (2024), 2.31M zone-hours, 10.2% hotspots:

  Precision / recall / F1 are for the top-decile hotspot call; normal and
  extreme hours are split by `extreme_flag`.

  | Model | Hours | Precision | Recall | F1 | AUC-PR |
  |---|---|---|---|---|---|
  | Seasonal naive (same hour last week) | all | 0.812 | 0.812 | 0.812 | 0.671 |
  | | normal | 0.811 | 0.812 | 0.811 | 0.667 |
  | | extreme | 0.823 | 0.807 | 0.815 | 0.744 |
  | 4-week same-hour average | all | 0.838 | 0.842 | 0.840 | 0.695 |
  | | normal | 0.837 | 0.843 | 0.840 | 0.691 |
  | | extreme | 0.847 | 0.836 | 0.841 | 0.767 |
  | **LightGBM** | all | **0.855** | **0.861** | **0.858** | **0.940** |
  | | normal | 0.854 | 0.861 | 0.857 | 0.940 |
  | | extreme | 0.864 | 0.867 | 0.865 | 0.947 |

  Precision ≈ recall by construction: the model calls exactly the top 10% of
  zones each hour, and ~10% of zone-hours are true hotspots. (Numbers are
  after the transit-flag fix below; extreme = 5.2% of calibrate zone-hours.)

  On the early-stop year (2023) LightGBM's F1 is 0.869, so it drops ~1.1
  points one year further out: a first, mild sign of drift.
- Run: 4.83M training rows, best iteration 2,250 of 3,000, fit 6.1 min,
  whole job 8 min on 32 cores.

Decisions:
- **Day sampling is zone-day, not city-day.** A zone's rows are always kept on
  a day it had any extreme hour. A city-day rule would keep nearly every day,
  because major events and flight delays fire in some zone almost daily.
- **Model inputs are an explicit whitelist.** The feature table also holds
  same-hour columns (`pickups`, `dropoffs`, `median_*`), the label, fold tags
  and the extreme flags; any of them would leak. `event_flag` is an input
  despite its name, because it comes from schedules.
- **Calendar features are derived from `hour`** (hour of day, weekday, month,
  weekend, US + state holiday). They're known in advance and weren't in the
  feature table.
- **Early stopping on log loss**, not AUC-PR, because the conformal layer
  needs reasonable probabilities.
- **Hotspot call = top decile of scores per hour**, the same quantile rule as
  the label. Calls and true hotspots are about the same count, so precision ≈
  recall by construction.
- **Test is predicted but not scored yet.** Predictions are saved in
  `data/processed_ml/nyc_fold3_pred.parquet` for the conformal and
  evaluation steps.
- **Reproducible:** the sampled days come from md5(date || seed), and
  LightGBM uses seed 42 with `deterministic=True`.

Edge cases and observations for the write-up:
- **Sampling kept 81% of training rows, not ~65%.** Citywide extremes
  (weather, NWS, subway) mark every zone on those days, so whole days are
  kept. 6.1% of kept rows are extreme. (86% / 8.0% before the transit-flag
  fix.)
- **The best iteration is close to the 3,000-round cap** (2,250 now, 2,641
  before the fix). Raise the cap or the learning rate when tuning.
- **AUC-PR understates the baselines.** Their scores are raw counts with many
  ties, which average precision penalizes. Top-decile F1 is the fairer
  comparison (0.840 vs 0.858).
- **Extreme hours still score slightly better than normal hours (F1 0.865 vs
  0.857).** The composite mixes conditions of very different difficulty
  (flight-delay zones are trivially hot; major events are hard), so the
  composite alone can make the model look robust. Evaluation must break
  results down per component (see below).
- **The model is almost entirely autoregressive.** Zone plus demand lags and
  rolling means carry ~87% of the gain; weather, NWS, flights and events
  together < 2%; subway ~2.6%. Under shift, conditions barely move its
  predictions, so the abstention layer has to carry the reliability story.
- **Subway features are NULL before 2020-07**, inside the training years.
  LightGBM handles the NaNs, but the feature's availability itself shifts
  over time.

**Modeling — Week 3 evaluation (`evaluate.py`, NYC fold 3, calibrate year):**

Metrics added, and why each one:
- **Day-block bootstrap 95% intervals** (1,000 resamples of whole days) for
  precision / recall / F1 and for the paired model-minus-baseline F1 gap. A
  point estimate can't show the gain is real; whole days are resampled
  because zone-hours within a day are strongly correlated.
- **Brier score and a reliability table.** The conformal layer is built on
  these probabilities, and training up-weights extreme rows 3x.
- **Per-condition breakdown with counts** (each component flag on its own).
  The composite flag made extreme hours look easier.
- **Month-by-month F1:** a simple drift curve across the year.
- Left out on purpose: plain accuracy (90% by calling nothing a hotspot),
  ROC-AUC (flattering at 10% prevalence), MCC (redundant since precision ≈
  recall ≈ F1).

Decision: **NYC transit flag tightened to ">= 3 subway suspension updates in
the hour"** (was ">= 3 suspension updates OR >= 10 routes affected"). The old
rule fired in 8.3% of hours (9.9% of the 2024 calibrate year), mostly through
the routes condition, which isn't extreme and isn't comparable over time:
MTA's alert system changed in 2020-04 and the newer alerts list more routes,
so ">= 10 routes" swings from 0.8% of hours in 2019 to 14.1% in 2020. The new
rule flags 2.3% of hours (~98.5th percentile), in line with severe weather.
Composite `extreme_flag` for NYC: 10.9% -> **5.1%** of hours (Chicago
unchanged at 8.4%; it uses the CTA anomaly). Edge case: 2019 still flags
less (0.6% of hours) because the 2012-2020 alert archive's labels were
coarser.

Results (after the transit fix):
- **The gain is real.** LightGBM F1 0.858 (95% interval 0.855-0.860). Gap over
  the 4-week average +1.8 points (1.4-2.2); over seasonal naive +4.6
  (4.1-5.1).
- **Overconfident in the middle range.** Brier 0.022, ECE 0.012 (small
  because most predictions sit near 0 or 1). But where the model says ~55%,
  hotspots happen 41% of the time; at ~85%, 72%. Likely causes: the 3x
  extreme weighting and drift from 2023 to 2024. Conformal calibration
  should absorb this, but report it.
- **By condition (calibrate year):**

  | Condition | LightGBM F1 | 4-week avg F1 | Share of zone-hours | Hotspot rate |
  |---|---|---|---|---|
  | Normal | 0.857 | 0.840 | 94.8% | 10.1% |
  | Transit disruption | 0.868 | 0.854 | 2.9% | 10.2% |
  | Severe weather | 0.846 | 0.803 | 1.8% | 10.2% |
  | NWS warning | 0.853 | 0.826 | 0.6% | 10.2% |
  | Major event | 0.809 | 0.746 | 0.1% | 33.8% |
  | Flight delays | 0.988 | 0.987 | 0.1% | 56.1% |

  Major events and severe weather are the genuinely hard conditions, and
  where LightGBM beats the baseline most (+6.3 and +4.3 points).
- **Drift within the year:** monthly F1 slides from ~0.865 (spring 2024) to
  0.840 (December). The 4-week average beats LightGBM in August and October,
  so the model's edge is not uniform.

**Deployable ablation** (`train.py --variant deployable`): drop the inputs
nobody has at forecast time — subway entries/transfers (MTA publishes ~2 weeks
late), CTA rides (~3 months late) and event attendance (reported after the
game). Same sample and settings.

| Model | F1, 2023 (early stop) | F1, 2024 (calibrate) | AUC-PR, 2024 |
|---|---|---|---|
| LightGBM, full | 0.869 | **0.858** | 0.940 |
| LightGBM, deployable | 0.859 | 0.838 | 0.927 |
| 4-week average | 0.841 | 0.840 | 0.695 |

- **Without the oracle ridership features, LightGBM no longer beats the
  4-week average on 2024 (F1 0.838 vs 0.840)**, though it still does on 2023
  and on extreme hours (0.848 vs 0.841; normal hours 0.837 vs 0.840).
- Subway entries at h-1 carry only ~2.6% of the full model's gain, yet are
  worth ~2 F1 points: they act as a live proxy for how busy a zone is right
  now. That signal isn't available to a real forecaster.
- The deployable model also drifts twice as fast (-2.1 points 2023 -> 2024 vs
  -1.1 for the full model).
- Implication for the write-up: "beats the baseline" depends on the
  deployment assumption. Report both variants. A deployable live-activity
  signal (e.g. the platform's own live request counts) is what would close
  the gap. Not yet in `evaluate.py` (no intervals for the deployable model).

**Next (Week 4)**: a simple abstention rule (one global threshold), the
normal vs extreme subgroup table, and the milestone report. Chicago fold 3
runs with the same command and `--city chicago`. See `MODELING_PLAN.md` for
the full plan, fold definitions, and threshold justifications.
