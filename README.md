Rideshare hotspot forecasting (Chicago + NYC) — data pipeline and modeling.

Raw trip/weather/transit/event data is fetched, aggregated to zone-hour
grain, and published to a Hugging Face dataset by the pipeline in
`scripts/`. The modeling plan built on top of that dataset is in
`MODELING_PLAN.md`.

## Event categories, named days, and remote trip endpoints

Broad contexts are **sports, concerts, federal holidays, elections, cultural
days (currently Halloween), and road races (currently NYC Marathon)**. These
are overlapping labels, not an event-type prediction target. The model predicts
surge hotspots conditional on the category and time. Sports/concert windows
default to 3 hours before, the scheduled start through estimated end, and
3 hours after. An hour overlapping a boundary may have two phases. The cached
concert pipeline assumes 19:30–23:00; use more reliable times before making
precise departure claims. US Open counts are matches, not distinct crowds.

Build context for every hour in the cached demand panels:

```sh
source .venv-torch/bin/activate
python scripts/build_calendar.py --city nyc chicago
python scripts/build_event_context.py --city nyc chicago
python scripts/evaluate_event_context.py --city nyc --zone-summary
```

The calendar builder appends `calendar_*` annotations to existing local feature
files atomically, and writes daily sidecars even for a city without built
features. It leaves labels and model predictor allowlists as they are.
Rules are audited for 2018–2026. Federal holidays distinguish actual and
weekday observed dates ([OPM](https://www.opm.gov/policy-data-oversight/pay-leave/federal-holidays/));
Halloween, Thanksgiving/adjacent days, Christmas/New Year's eves, marathon
and adjacent days, and DST changes are flagged. The in-person 2020 marathon
is [canceled](https://www.nyrr.org/tcsnycmarathon/pages/2020-cancellation), not flagged active.
Elections cover annual November general elections in NYC, even-year November
general elections in Chicago, and verified Chicago 2019/2023 municipal rounds.
Primaries, special elections, and early voting are not covered.
The sourced catalog and `calendar_metadata.json` record scope and provenance.

`*_event_context.parquet` contains citywide category counts by phase and all
named-day flags, with no target-hour demand or actual attendance added.
`*_event_exclusions.parquet` masks the **union of all mapped active sports/concert
venue zones**, including before/after windows across midnight. Transit entry
zones remain candidates. Outside-city/unmapped venues have no known city zone
to mask; inspect `*_event_coverage.csv`. Calendar contexts are citywide and
have no venue mask. The marathon route has not been mapped/excluded yet.
The event catalog is a retrospective snapshot, not a record of what schedule
information was published at each historical forecast time.

The frozen evaluator reuses saved calls and scores on early-stop/calibration
only. `data/results/nyc_fold3_event_context_metrics.csv` includes overall,
ordinary hours, categories, windows, and individual calendar flags, both with
and without venue masking. It includes negatives, date/hour counts, confusion
counts, precision/recall/F1/AP, and differences from the same-scope overall
metric. Groups overlap and holiday/election samples can have just one date.
`*_zones.csv` ranks mean model scores and reports marginal pickups/dropoffs;
these are descriptive zone totals, not linked trips or causal event effects.

For NYC fold 3, the existing improved full Torch model on **2024 off-venue
eligible pickup-surge rows** has F1 0.445 overall, 0.414 on sports windows,
0.471 on concert windows, and 0.745 on federal holidays. Election Day is
0.162 and Marathon Day 0.212, each on one date; the marathon also overlaps
DST fall-back, where the source merges two 01:00 hours. These are diagnostics,
not category-specific guarantees. Test rows remain unscored.

Train a separate pickup-surge experiment using event context on the Mac GPU:

```sh
python scripts/train_surge_torch.py --city nyc --device mps \
  --variant deployable --event-context --off-venue --run-name events_offvenue
python scripts/evaluate_event_context.py --city nyc --pred \
  data/processed_ml/nyc_fold3_surge150_torch_deployable_events_offvenue_pred.parquet \
  data/processed_ml/nyc_fold3_surge150_torch_improved_pred.parquet \
  data/processed_ml/nyc_fold3_surge150_pred.parquet \
  data/processed_ml/nyc_fold3_surge150_deployable_pred.parquet
```

`--off-venue` excludes venue targets from both training and saved predictions;
the category inputs stay citywide. This remains the existing **pickup surge**
target. It does not yet train a dropoff or venue-linked endpoint classifier.
The first MPS run restored epoch 25 (about one minute fit) and selected its
threshold on 2023 only. On the same 2024 off-venue rows, F1 is 0.448 overall,
0.418 for sports and 0.480 for concerts, compared with 0.445/0.414/0.471 for
the earlier improved full Torch model. This is a new feature/availability/mask
configuration, not an isolated test of the category features. Some groups got
worse (Halloween F1 0.434 versus 0.512); the one-date marathon result is 0.581
versus 0.212. These observations do not establish a general improvement.
Checkpoints carry the feature names and context/mask settings. For inference,
rebuild context for the target schedule, call `event_context.attach()` on
calendar/enhanced predictor rows, then the existing restore/predict helpers
and saved threshold policy. `deployable` still assumes access to timely
pickup counts and retains the documented flight-publication caveat.

For actual trip links, use the **cluster** data without copying raw trips to
the Mac. `aggregate_event_trips.py` streams one monthly Parquet at a time:

- Arrivals: non-venue pickup zone → venue, before/during the event.
- Departures: venue → non-venue dropoff zone, during/after the event.
- Only the remote endpoint is a candidate hotspot. Other active mapped venues
  are excluded there too. Overlapping windows in the same category/phase/venue
  are unioned before joining, so one trip is counted once within that group.
- The same windows 1–4 weeks earlier provide weekday/time controls. Controls
  can themselves contain events; treat them as observational comparisons and
  check contamination before constructing uplift labels.

Start with a one-month pilot. From the cluster repo, substitute the actual
converted NYC directory; the glob must include December 2023 for January's
controls. Original TLC monthly `fhvhv_tripdata_YYYY-MM.parquet` also works.
Chicago requires the converted monthly schema from `convert.py`.

The downloaded `MLCapstoneRideshare/rideshare-trips` dataset also works locally.
It has the converted schemas above. The current Mac setup links `data/trips`
to cached revision `b1bcdccf26660bafa3a66042f6a16a0d9bd618a8`; no second copy
is needed. Keep this link under the ignored `data/` directory and leave the
cached source files read only. Run a local pilot with:

```sh
.venv-torch/bin/python scripts/aggregate_event_trips.py --city nyc \
  --trip-glob 'data/trips/nyc_fhvhv/*.parquet' \
  --start-date 2024-01-01 --end-date 2024-01-31 \
  --out data/event_trips/nyc_2024jan_pilot \
  --threads 8 --memory-limit 8GB --temp-dir data/duckdb_tmp/event_trips
```

Use a fresh `--out` if that pilot directory already exists. On this Mac, the
January pilot finished in 2.77 seconds with about 1.24 GB maximum resident
memory. It produced 264,973 aggregate rows in about 0.37 MB of Parquet, including
exposure metadata. It read December/January and the February boundary needed
for late-January spillover. These are measured pilot results, not a promise of
the same throughput on a multi-year run. `pilot_audit.json` records the checks;
no invalid zones, nonpositive counts, masked venue targets, or missing required
month files were found. These checks do not prove that every day of source data
is complete. For Chicago, the local glob is
`data/trips/chicago_trips/*/*.parquet` (quote the glob).

For the cluster instead:

```sh
mkdir -p logs
sbatch slurm/10_event_trips.sbatch --city nyc \
  --trip-glob '/cluster/path/data/processed/nyc_fhvhv/*.parquet' \
  --start-date 2024-01-01 --end-date 2024-01-31 \
  --out data/event_trips/nyc_2024jan_pilot
```

The job uses the existing `rideshare` conda environment and `normal_q` partition,
8 CPU cores, a 16 GB allocation, and an 8 GB DuckDB budget. No GPU is needed for
aggregation. Run the same Python command directly inside an appropriate compute
allocation if Slurm configuration differs; `--temp-dir` can use cluster scratch.
The pilot runtime/output size must be measured before choosing a full-run budget.

Output is additive `part-YYYY-MM.parquet` counts keyed by `reference_hour`,
`category`, `phase`, `direction`, `period` (event/control week), and `remote_zone`.
Sum shards by these keys because arrivals can cross input-month boundaries.
`exposures.parquet` records actual observed hours, reference event hours,
window minutes per venue, and whether required month files are present.
Month-file presence does not establish day-level completeness; audit that
against the cluster data manifest. Missing coverage must not become a zero
training label. Only consume a run when `manifest.json` has `status: complete`.
The manifest fingerprints the event catalog and records input file metadata.
Use a fresh output directory for each run.

Transfer only the aggregate directory back. The next modeling stage is to join
remote endpoint counts/coverage to category, zone and known-ahead features,
form separate arrival-origin and departure-destination targets, and evaluate
with chronological splits and all eligible non-venue zones (including zeros).
These venue-associated trips do not prove attendance. Citywide holidays and
elections use their own date cohorts; there is no single venue to anchor them.

## Apple GPU neural-network comparison

`scripts/train_torch.py` adds a tabular PyTorch classifier alongside LightGBM.
It uses Apple's GPU through MPS (`--device mps`), with CPU and CUDA options too.
The model combines an 8-value learned zone embedding with standardized numeric
features and missing-value indicators, then passes them through 128- and
64-unit hidden layers with ReLU/dropout and a binary output. This is a different
model family, so GPU utilization alone does not imply better speed or accuracy.

Both trainers share `scripts/model_data.py`: the same predictor allowlist,
calendar features, existing training sample/weights, and chronological folds.
The neural trainer fits imputation/scaling on training rows only, minimizes
weighted binary cross-entropy with AdamW, and restores the epoch with lowest
unweighted log loss on `early_stop`. It scores `early_stop` and `calibrate` with
the existing top-decile rule; test predictions are saved without test metrics.
Features entirely absent during training stay disabled at inference even if
observations become available later (notably MTA ridership in fold 1).
`full` and `deployable` have the same feature-availability assumptions as
LightGBM, including the outstanding flight-data publication-delay limitation.

Use a separate Python 3.12 environment (the existing Python 3.9 environment can
remain available for LightGBM):

```sh
python3.12 -m venv .venv-torch
source .venv-torch/bin/activate
python -m pip install -r requirements-torch.txt
```

Build `*_labeled.parquet`, `*_features.parquet`, and `*_fold3_train.parquet`
with `label.py`, `features.py`, and `sample.py` first. Existing files from the
LightGBM run are reusable. Then:

```sh
python scripts/train_torch.py --city nyc --device mps
python scripts/evaluate.py --city nyc --model torch
python scripts/conformal.py --city nyc --model torch
```

For the deployable variant, add `--variant deployable` to all three commands.
Default training is at most 50 epochs with patience 5 and batches of 8,192;
adjust `--epochs`, `--batch-size`, `--lr`, `--hidden`, or `--dropout` to tune on
`early_stop`. Do not select hyperparameters on calibration/test performance.
Training batches stream to GPU by default. `--cache-device` keeps the training
and validation tensors there for fewer transfers, at a higher memory cost.
The script prints the selected device and epoch timings; its history CSV
records validation loss and throughput. Seeds are fixed, but bitwise GPU
reproducibility is not guaranteed. Numeric features/weights use float32.

Artifacts have their own tag, e.g. `nyc_fold3_torch` or
`nyc_fold3_torch_deployable`: a `.pt` checkpoint in `data/models/`, a prediction
Parquet in `data/processed_ml/`, and model metrics/training history CSVs in
`data/results/`. The checkpoint includes CPU-loadable weights, preprocessing,
architecture, feature names and configuration; `train_torch.restore()` loads
it for inference (derive calendar columns with `model_data.add_calendar()`
before calling `train_torch.predict()`). Neural models do not have LightGBM's
split-gain feature importance, so no equivalent importance file is emitted.
Existing LightGBM filenames and default evaluation commands are preserved.

Checks: `.venv-torch/bin/python -m unittest discover -s tests -v` exercises
training-only preprocessing, both feature variants, checkpoint reloads,
evaluation/conformal outputs, and MPS training when available.

## Apple GPU surge model and onset analysis

`scripts/train_surge_torch.py` trains the **surge_150** target (pickups >=
1.5 times the zone's past-only normal baseline, baseline >= 10 rides/hour).
It reuses the existing surge labels and training-day sample, plus exactly the
LightGBM surge model's predictors: the shared whitelist and `surge_baseline`,
`ratio_lag1`, `surge_lag1`, `surge_lag168` in `--features basic` mode. The MLP, train-only scaling and
weighted GPU training loop are shared with `train_torch.py`, including the
validation-loss learning-rate scheduler. Test predictions are saved, never scored.
The enhanced feature set and guarded joint threshold policy described below are
now the defaults; basic/global mode reproduces the original experiment.

```sh
source .venv-torch/bin/activate
python scripts/train_surge_torch.py --city nyc --device mps
python scripts/train_surge_torch.py --city nyc --device mps --variant deployable
```

The script automatically reports overall performance, conditions, and these
prior-state groups for the neural model, persistence baselines, and saved
LightGBM surge predictions when available:

| Group | Eligible evaluation rows | Positive outcome | Negative outcome |
|---|---|---|---|
| onset | previous hour was not a surge | a surge starts now | stays non-surge (a positive call is a false alarm) |
| continuation | previous hour was a surge | surge continues | surge ends (a positive call is a false alarm) |
| unknown_history | missing prior label or a local clock-hour gap | current surge | current non-surge |

These are risk sets, not positive-only subsets. Precision, recall, F1,
average precision, confusion counts, positive counts and false alarms per
1,000 negative zone-hours are recorded. Undefined metrics (e.g. precision
when no positives are predicted) are blank, rather than a claimed perfect
score. History is computed before volume filtering or training sampling;
clock-hour gaps, including spring DST jumps, stay unknown. The existing
row-based predictor lags are retained to match the LightGBM experiment.
Fall-back merged hours follow the dataset's local-hour convention.

This is phase evaluation of a model trained on all surge-eligible hours. It
does not yet train a separate onset classifier or predict multiple hours
before onset. The strict `persistence_label` baseline necessarily has zero
onset recall; the model must improve on that while controlling false alarms.
The older `surged_last_hour` baseline is retained for comparison: it thresholds
the previous ratio without the previous hour's volume floor, so its calls can
differ from persistence of the actual surge label.

Outputs use `nyc_fold3_surge150_torch[_deployable][_{run_name}]` (the current
default run suffix is `improved`): model/preprocessing and
cutoff in `data/models/*.pt`, all held-out predictions/history metadata in
`data/processed_ml/*_pred.parquet`, and `*_metrics.csv`, `*_phases.csv`,
`*_history.csv` in `data/results/`. CPU surge and original hotspot artifacts
keep their existing names. Use `--run-name experiment1` to give a tuning run
its own filenames. `--cache-device` can reduce batch transfers at a higher
memory cost. As with the CPU model, "deployable" retains the documented
flight-data availability assumptions; GPU training does not resolve them.

Re-evaluate saved predictions without retraining:

```sh
python scripts/evaluate_surge.py --city nyc --model torch --run-name improved
python scripts/evaluate_surge.py --city nyc --model lightgbm --variant deployable
```

Pass the same `--run-name` when analyzing a named neural run. Saved LightGBM
calls are reused without refitting a cutoff, and are enriched with prior-hour
labels from the surge table. Analysis files get an `_analysis` suffix.
Use 2023/`early_stop` for model selection; the reported calibration metrics
are preliminary evaluation and should not be used to select hyperparameters.

### Improved features and guarded threshold selection

The default command now uses `--features enhanced --threshold-mode auto
--run-name improved`, preserving the original unsuffixed run. Enhanced inputs
add 2/3/24-hour lagged ratios, their recent mean and changes, pickup growth,
the four-week same-hour pickup mean, last-hour pickups relative to the target
hour's known baseline, log counts, and sine/cosine calendar encodings. All
windows are computed on the full series before sampling; no current pickup,
current ratio or current surge label becomes a predictor. `--features basic`
retains the original feature contract for ablation comparisons.

Joint decision thresholds optimize **pooled overall F1**, not the average of
onset and continuation F1. By default their fit-time precision must be at least
that of the best global F1 threshold. `--min-precision 0.50` instead requests
an explicit validation precision floor. This can reduce recall/F1, and it is
not a guarantee of future precision.

In `--threshold-mode auto`, the first half of the validation hours fits a
global rule and a joint rule. The later half checks both. A joint rule is
selected only if it improves pooled F1 by at least 0.001 without reducing
precision; otherwise the global rule is retained. The chosen kind of rule is
then refitted on all `early_stop` rows. This is a tuning safeguard within the
year already used for model early stopping, not an independent test result.
Unknown history always uses the global fallback. Calibrate/test labels never
select a rule. `--threshold-mode global` disables phase thresholds;
`--threshold-mode joint` explicitly bypasses the temporal safeguard for an
exploratory fit. Both respect an explicit precision floor when supplied.

```sh
python scripts/train_surge_torch.py --city nyc --device mps
python scripts/train_surge_torch.py --city nyc --device mps --min-precision 0.50 --run-name precision50
python scripts/train_surge_torch.py --city nyc --device mps --features basic --threshold-mode global --run-name basic_control
python scripts/evaluate_surge.py --city nyc --model torch --run-name improved
```

The saved checkpoint contains `threshold_policy`; apply it with
`surge_thresholds.apply_policy(frame, checkpoint['threshold_policy'])` after
producing `score` and providing the past-known `previous_surge` metadata.
Do not apply the reference scalar `cutoff` to all rows when the selected rule
is joint. Predictions record each row's `decision_threshold`, and a JSON
policy file plus `*_threshold_selection.csv` audit file record the selection
and later-validation checks. The console now shows validation metrics before
calibration metrics. The evaluator reuses saved calls; it never refits them.

First enhanced/joint run (NYC fold 3, fixed configuration, 2026-10-28):
the temporal threshold check selected the joint rule: on the later 2023
validation hours, pooled F1 rose 0.4819 -> 0.4843 and precision 0.4693 ->
0.5046 compared with a global rule fitted on the earlier hours. Refitting
on all validation hours gave onset/continuation/fallback cutoffs
0.2444 / 0.1839 / 0.2226. Best epoch 37; fit about 1.1 minutes on MPS.

The subsequent 2024 calibration results (not used for selection) were:

| Model | Average precision | Precision | Recall | F1 |
|---|---|---|---|---|
| Original Torch reference | 0.4491 | 0.4500 | 0.4271 | 0.4382 |
| Enhanced Torch + guarded joint rule | 0.4664 | 0.4951 | 0.4147 | 0.4513 |
| LightGBM full reference | 0.4616 | 0.4713 | 0.4232 | 0.4459 |

This improves pooled precision/F1 in one run, not every subgroup metric:
onset precision improves 0.3105 -> 0.3789, but onset recall falls
0.2009 -> 0.1747 and onset F1 changes 0.2440 -> 0.2392. Its onset average
precision rises 0.2141 -> 0.2329, indicating a ranking improvement despite
the more conservative chosen onset calls. Phase-specific tradeoffs remain
visible in `*_phases.csv`. Repeated-seed/paired-interval checks are pending;
do not treat small differences from LightGBM as established superiority.

Initial NYC fold-3 GPU run (full variant, fixed defaults, 2026-10-28):
best epoch 44, cutoff 0.2069 selected on 2023, fit about 1.1 minutes on the
Apple GPU. On 2024, neural overall F1 is 0.4391 versus 0.4459 for the saved
full LightGBM model. By prior state:

| Model | Onset precision / recall / F1 | Continuation precision / recall / F1 |
|---|---|---|
| PyTorch full | 0.3255 / 0.1879 / 0.2383 | 0.5704 / 0.8533 / 0.6837 |
| LightGBM full | 0.3281 / 0.2085 / 0.2549 | 0.5957 / 0.8372 / 0.6961 |
| LightGBM deployable | 0.3252 / 0.2144 / 0.2584 | 0.5897 / 0.8465 / 0.6952 |

There are 42,568 true onset hours and 21,965 continuation hours, plus 226
positive hours with unknown history. These results identify new-surge onset
as the harder problem; they do not establish GPU accuracy gains. Test remains
unscored. Confidence intervals for phase-specific differences are still pending.

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

**Expected attendance (deployable variant, 2026-10-28):** the deployable model
swapped actual attendance (reported after the game) for `expected_attendance`:
the average attendance of the same home team's earlier games (or the venue's,
for concerts / neutral sites), over calendar days strictly before the event,
skipping zero-attendance 2020-21 games. Result: no real change (F1 0.838 ->
0.837 overall, 0.799 -> 0.801 on major-event hours; every condition within
±0.002), and it isn't among the top 10 features. Crowd size isn't what the
model is missing on event hours; the scheduled `event_flag` already carries
that an event is on. The errors are more likely about *when and where* people
leave. Edge case: concerts and US Open matches have no attendance in the
sources, so their expected attendance is always NULL.

**Modeling — Week 4 (simple abstention rule, NYC fold 3, 2026-10-28):**
- Script: `scripts/conformal.py --mode global` (reads the saved predictions,
  runs in seconds).
- Method: split conformal with one global threshold. Calibration score
  s = 1 - p(true class); q = the ceil((n+1)(1-alpha))-th smallest score. A row
  with hotspot probability p gets {hotspot} if p >= 1-q and {not} if p <= q;
  exactly one -> confident call, both or neither -> abstain.

Decisions:
- **Test stays untouched.** Thresholds are fit on the first half of the
  calibrate year (Jan-Jun 2024, 1.15M rows) and evaluated on the second half
  (Jul-Dec 2024, 1.16M rows). The second half also carries the within-year
  drift seen in Week 3, so it doubles as a small shift test.
- **One global threshold first** (as planned); the Mondrian version with
  separate thresholds per class x condition is Week 5.
- **Full model is the headline; deployable is reported alongside.**

Results, full model, Jul-Dec 2024:

| alpha | Target coverage | Coverage | Hotspot coverage | Abstain (empty / full) | Accuracy when confident | Confident-miss rate |
|---|---|---|---|---|---|---|
| 0.01 | 0.99 | 0.990 | 0.958 | 6.3% (0 / 6.3%) | 0.989 | 4.2% |
| 0.02 | 0.98 | 0.980 | 0.916 | 2.6% (0 / 2.6%) | 0.979 | 8.4% |
| 0.05 | 0.95 | 0.947 | 0.794 | 3.6% (3.6% / 0) | 0.982 | 7.0% |
| 0.10 | 0.90 | 0.895 | 0.638 | 10.0% (10.0% / 0) | 0.995 | 2.0% |
| 0.15 | 0.85 | 0.844 | 0.503 | 15.5% (15.5% / 0) | 0.999 | 0.6% |
| 0.20 | 0.80 | 0.792 | 0.384 | 20.7% (20.7% / 0) | 1.000 | 0.2% |

(Confident-miss rate = share of true hotspots confidently called "not".)

By condition at alpha = 0.05:

| Condition | Coverage | Hotspot coverage | Abstain | Accuracy when confident | Confident-miss rate |
|---|---|---|---|---|---|
| Normal | 0.947 | 0.793 | 3.6% | 0.982 | 7.1% |
| Extreme (any) | 0.942 | 0.820 | 3.9% | 0.981 | 6.0% |
| Severe weather | 0.944 | 0.789 | 3.8% | 0.981 | 6.7% |
| NWS warning | 0.948 | 0.884 | 3.3% | 0.980 | 4.8% |
| Transit disruption | 0.949 | 0.814 | 3.5% | 0.984 | 6.0% |
| **Major event** | **0.753** | 0.774 | **15.2%** | **0.888** | **10.2%** |
| Flight delays | 0.988 | 0.987 | 0.9% | 0.997 | 0.4% |

Findings and edge cases for the write-up:
- **Overall coverage roughly holds but runs slightly under target** on the
  second half of 2024 (0.947 vs 0.95, 0.895 vs 0.90, 0.792 vs 0.80): the
  within-year drift already erodes the guarantee a little.
- **The hotspot class is badly under-covered:** 0.79 at alpha = 0.05 and only
  0.38 at alpha = 0.20. A single threshold is set mostly by the 90% "not" rows,
  so it protects the majority class. This is the case for Week 5's
  class-conditional (Mondrian) thresholds.
- **The kind of abstention flips with alpha.** For alpha <= 0.02, q >= 0.5 and
  abstentions are full sets (both answers plausible); for alpha >= 0.05,
  q < 0.5 and they're empty sets (neither plausible enough). That's why the
  confident-miss rate isn't monotone in alpha (8.4% at 0.02, 7.0% at 0.05,
  2.0% at 0.10): with empty sets, mid-probability hotspots get abstained on
  instead of confidently called "not".
- **The guarantee breaks on major events:** coverage 0.75 against a promised
  0.95, abstention 4x the normal rate (15.2%), accuracy when confident 0.89
  and 10% of true hotspots confidently missed. The model does abstain more
  there, but not on most cases, so the failure criterion isn't triggered at
  this alpha. These are the hours conditional thresholds need to fix.
- **Flight-delay hours are trivially easy** (airport zones are nearly always
  hotspots), so they shouldn't be read as "robust under disruption".
- **Deployable model** (no ridership): same coverage overall (0.946) but
  hotspot coverage 0.72 and confident-miss rate 12.2% at alpha = 0.05 (vs 0.79
  and 7.0%). A weaker model shows up as more confident misses at the same
  abstention rate.

**Next**: the milestone report (Week 4), then the Mondrian version with
thresholds per class x condition and the alpha sweep (Week 5). Chicago fold 3
runs with the same commands and `--city chicago`. See `MODELING_PLAN.md` for
the full plan, fold definitions, and threshold justifications.
