# Surge label ("unexpected hotspots"): design note

Status: proposed second label, alongside the top-10% `hotspot` label (which stays
the milestone label). Implemented in `scripts/surge_label.py`; no existing script
was changed. For discussion with the group before any training on it.

## Why a second label

The top-10% label is mostly static. In NYC in 2024:
- 20 of 263 zones (7.6%) produce 56% of all hotspot hours; 8 zones (3%) produce 26%.
- A zone's hotspot status at a given hour differs from the same hour last week in
  only 3.8% of zone-hours, and only 18.8% of hotspots weren't hotspots the week before.

So the task is largely "remember which zones are always busy", which is why even
the seasonal-naive baseline scores F1 0.81. A driver doesn't need a model to know
Midtown and the airports are busy. The useful, and research-relevant, question is
**where is demand unusually high right now**, which is exactly what anomalous
conditions (events, storms, outages) should cause.

## Definition

**Baseline** = expected pickups for that zone and hour under normal conditions:
- the **median** of pickups at the **same hour on the same weekday** over the
  **previous 8 weeks**;
- ignoring hours flagged extreme (any component) and public holidays, so a past
  storm, game or Thanksgiving doesn't define "normal";
- past data only; NULL if fewer than 4 clean weeks are available.

What the baseline is, and isn't, conditioned on:

| Factor | In the baseline? | Why |
|---|---|---|
| Hour of day, day of week | yes (same hour, same weekday) | rush hours and weekend nights are the normal pattern the baseline must absorb |
| Season, long-term trend | yes, via a rolling 8-week window | demand rose a lot 2021-2024; an all-time average would drift |
| Holidays | excluded from the window | a holiday shouldn't set next week's "normal" |
| Weather, events, outages | **no** | these are the anomalies to detect; if the baseline expected a Knicks game, the game would stop looking like a surge |

Why not a zone's all-hours average: it flags 20% of morning-rush and 26% of
evening-rush zone-hours as "surges", which is just the daily cycle. Same-hour,
same-weekday baselines bring that down to ~1-5% at every time of day. The median
beats the mean because one past spike (a game the same hour 3 weeks ago) doesn't
move it: false surges in the morning rush drop from 2.8% (4-week mean) to 1.2%
(8-week median).

**Labels** (all describe hour h itself, like `hotspot`):

| Label | Rule | Notes |
|---|---|---|
| `surge_125` | pickups >= 1.25x baseline, baseline >= 10/h | mild; closest to "busier than usual" |
| `surge_150` | >= 1.5x | the prototype threshold |
| `surge_200` | >= 2x | clearly unusual |
| `surge_300` | >= 3x | rare, event-scale |
| `surge_z3` | (pickups - baseline) / sqrt(baseline) >= 3, baseline >= 5/h | count-aware alternative |

- **Volume floor (10 expected rides/h)** for the ratio labels, so a zone going
  from 1 to 3 rides doesn't count as +200%. Zones below it can never surge
  (that share is reported).
- **Count-aware alternative (`surge_z3`)**: a Poisson-style z-score, how many
  "natural noise" units above expectation. Small, noisy zones need a bigger
  relative jump; big zones can surge with a smaller one. It addresses the higher
  night-time surge rate seen with ratio labels (small counts bouncing near the
  floor).
- **Fixed thresholds, not a top-k per hour**: a fixed rule lets the surge rate rise
  during events and storms, which is the effect to study. Top-k would force the
  same rate every hour and hide it.

## Using it (not done yet)

- **Baselines**: the 4-week / same-hour average *is* the reference point, so it
  would never predict a surge. Fair baselines: "surging at the same hour last
  week" and each zone's historical surge rate.
- **Model**: the same LightGBM pipeline with the surge label as target (a new
  training script or a `--label` option; existing files stay as they are until
  the group agrees). The baseline itself (lagged, past-only) and its ratio to the
  latest hour are natural inputs.
- **Expectations**: much rarer and harder than the top-10% label, so lower F1 and
  AUC-PR; conditions features (events, weather) should matter far more than for
  the static label; the conformal layer will be strained on the minority class,
  which is itself a finding.
- **Edge cases**: DST-merged fall-back hours have doubled counts (flagged
  `dst_merged_hour` in the city-hour table); the first 4+ weeks of each zone's
  series have no baseline; holidays are excluded from baselines but can still
  surge themselves.

## Results (first run, 2026-10-28)

Output: `data/processed_ml/{city}_surge.parquet` (zone, hour, pickups, baseline,
ratio, surge_z, the five labels, fold tags, extreme flags, holiday) and
`data/results/{city}_surge_label_summary.csv`. 29 s for both cities.

Coverage: a baseline exists for 98.6% (NYC) / 97.8% (Chicago) of zone-hours, and
is >= 10 rides/h for 84.5% / 85.0%; the rest can never surge on the ratio labels.

Surge rate (share of zone-hours with a baseline), NYC / Chicago:

| Label | All | Normal hours | Major event | Severe weather | NWS warning | Transit disruption |
|---|---|---|---|---|---|---|
| `surge_125` | 11.6% / 12.0% | 11.4% / 11.9% | 39.7% / 47.7% | 19.5% / 13.6% | 17.3% / 12.3% | 11.1% / 10.8% |
| `surge_150` | 3.4% / 3.5% | 3.4% / 3.4% | **28.4% / 30.5%** | 6.3% / 3.9% | 5.9% / 3.5% | 3.2% / 6.1% |
| `surge_200` | 0.70% / 0.69% | 0.68% / 0.63% | 18.1% / 16.7% | 0.98% / 0.68% | 1.19% / 0.66% | 0.61% / 2.91% |
| `surge_300` | 0.15% / 0.16% | 0.14% / 0.13% | 10.0% / 6.7% | 0.14% / 0.15% | 0.24% / 0.15% | 0.10% / 1.27% |
| `surge_z3` | 6.3% / 5.3% | 6.0% / 5.1% | 44.2% / 59.5% | 13.4% / 6.8% | 12.9% / 6.4% | 6.3% / 7.9% |

Zone spread and overlap with the top-10% label:

| Label | Zones with any surge (NYC / Chicago) | Highest single-zone surge rate | Share of surges that are also top-10% hotspots |
|---|---|---|---|
| `surge_125` | 252 / 77 | 19.7% / 23.8% | 14.1% / 12.4% |
| `surge_150` | 252 / 77 | 9.0% / 11.5% | 15.6% / 17.3% |
| `surge_200` | 251 / 77 | 5.7% / 3.9% | 21.6% / 33.8% |
| `surge_300` | 250 / 75 | 3.3% / 1.3% | 31.0% / 47.2% |
| `surge_z3` | 256 / 77 | 27.1% / 29.5% | 32.5% / 45.1% |

By time of day (NYC; Chicago is similar):

| Label | 00-05 | 06-09 | 10-15 | 16-19 | 20-23 |
|---|---|---|---|---|---|
| `surge_150` | 5.3% | 1.8% | 2.3% | 2.9% | 4.5% |
| `surge_z3` | 5.8% | 3.9% | 5.5% | 7.7% | 9.0% |

Fold-3 splits (NYC, `surge_150`): train 3.0%, early stop 2.9%, calibrate 2.8%,
test 2.6% — stable, slowly declining.

Findings:
- **Not static.** Surges happen in almost every zone, no zone surges more than
  ~9-12% of the time at 1.5x, and only ~16% of `surge_150` zone-hours are also
  top-10% hotspots. It measures something different from the current label.
- **Strongly tied to events.** At 1.5x, major-event hours surge 8-9x as often as
  normal hours (28-31% vs 3.4%); at 2x and 3x the enrichment is 25-70x. Severe
  weather and NWS warnings roughly double the NYC surge rate; Chicago's weather
  effect is weaker.
- **Transit disruption differs by city.** NYC subway suspensions barely change
  the surge rate (3.2% vs 3.4%); Chicago's CTA-anomaly days nearly double it
  (6.1%), up to 10x at 3x. Plausibly riders switch to rideshare when the 'L'
  is disrupted, but the CTA flag is daily, so treat this as tentative.
- **Night-time bias in the ratio labels.** `surge_150` is ~3x more common at
  night (5.3%) than in the morning rush (1.8%): small night-time counts near the
  volume floor are noisier. `surge_z3` removes that bias but tilts toward big,
  busy zones (32-45% overlap with the top-10% label) and evenings.
- **2x and 3x are too rare to train or calibrate on** (0.7% and 0.15% of
  zone-hours; the conformal layer needs enough positives per class x
  condition). They're useful as severity tiers for evaluation.

Recommendation: `surge_150` as the main surge label (3.4% positive, spread
across zones, strong event signal), with `surge_z3` as a sensitivity check that
corrects the night-time bias, and `surge_200` / `surge_300` reported as severity
tiers. Raising the volume floor at night (or using `surge_z3`) is the main open
tuning question.
