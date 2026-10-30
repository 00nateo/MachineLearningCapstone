# Results so far

Team reference for every model run to date. Everything here is NYC, fold 3 (train → 2022,
early stop 2023, calibrate 2024, test 2025+; test is predicted but **not scored yet**). No Chicago
models and no PyTorch runs yet. Write-ups with full reasoning: `README.md` (top-10% label,
conformal) and `SURGE_LABEL_DESIGN.md` (surge label).

Refresh after new runs with `python scripts/collect_results.py` (copies `data/results/*.csv` and
reads each saved model's params and inputs), then update the tables below by hand.

## Runs

| Run | Script | Slurm job | Code at | CSVs |
|---|---|---|---|---|
| Baselines + LightGBM top-10%, full | `train.py` + `evaluate.py` | 8257029 | `08ae89c` | `hotspot_top10/nyc_fold3_{baselines,model,importance,eval_*}` |
| LightGBM top-10%, deployable (actual attendance) | `train.py --variant deployable` | 8257030 | `08ae89c` | `hotspot_top10/nyc_fold3_deployable_conditions_before_expected_attendance` |
| LightGBM top-10%, deployable (expected attendance) | `train.py --variant deployable` | 8257619 | `3853bd2` | `hotspot_top10/nyc_fold3_deployable_{model,importance}` |
| Global conformal, full / deployable | `conformal.py --mode global` | 8257688 / 8257689 | `3853bd2` | `conformal/*_abstention_global` |
| Surge labels, both cities | `surge_label.py` | 8260054 | `bcf2dc3` | `surge/*_surge_label_summary` |
| Surge-150 LightGBM, full + deployable | `train_surge.py` | 8260107 | `040252d` | `surge/nyc_fold3_surge150_*` |

Logs: `logs/<name>_<job>.out` on the cluster. Models (`data/models/*.txt`) and predictions
(`data/processed_ml/*_pred.parquet`) stay on the cluster.

## Config

All four models share one LightGBM setup (`config/<tag>_params.json` has the full list):
binary objective, learning rate 0.05, 63 leaves, min 200 rows per leaf, feature and bagging
fraction 0.8, L2 1.0, seed 42, deterministic; up to 3,000 rounds, early stop after 100 rounds
without improvement in early-stop log loss.

Training rows (`config/sampling.json`): 20 random days per month plus every zone-day with an
extreme hour, extreme rows weighted 3x; 4.83M rows (81% kept), 6.1% extreme. Early stop,
calibrate and test keep every row.

| Model tag | Trees | Inputs | Inputs list |
|---|---|---|---|
| `nyc_fold3` (top-10%, full) | 2,250 | 23 | `config/nyc_fold3_features.txt` |
| `nyc_fold3_deployable` | 2,370 | 21 | drops subway entries/transfers and actual attendance; adds `expected_attendance` |
| `nyc_fold3_surge150` | 1,384 | 27 | full + 4 surge inputs (`surge_baseline`, `ratio_lag1`, `surge_lag1`, `surge_lag168`) |
| `nyc_fold3_surge150_deployable` | 1,689 | 25 | deployable + the same 4 |

Variants: **full** uses every whitelisted input; **deployable** drops inputs nobody has at
forecast time (MTA ridership is published ~2 weeks late, attendance after the game).

## Headline numbers (calibrate year, 2024)

**Top-10% hotspot label** (10.2% positive; F1 of the top-decile call, 95% day-block bootstrap
interval):

| Model | F1 | AUC-PR | F1 normal / extreme hours |
|---|---|---|---|
| Seasonal naive | 0.812 | 0.671 | 0.811 / 0.815 |
| 4-week average | 0.840 | 0.695 | 0.840 / 0.841 |
| LightGBM, full | **0.858** (0.855–0.860) | 0.940 | 0.857 / 0.865 |
| LightGBM, deployable | 0.837 | 0.927 | 0.837 / 0.849 |

**Surge-150 label** (pickups ≥ 1.5× the zone's 8-week same-hour median; 3.3% positive):

| Model | F1 | AUC-PR | F1 on major-event hours |
|---|---|---|---|
| Surged last hour | 0.344 | 0.296 | 0.684 |
| LightGBM, full | 0.446 | 0.461 | 0.801 |
| LightGBM, deployable | **0.448** | **0.464** | 0.792 |

**Global conformal, full top-10% model** (thresholds fit Jan–Jun 2024, evaluated Jul–Dec 2024),
alpha = 0.05:

| Condition | Coverage (target 0.95) | Hotspot coverage | Abstain | Accuracy when confident | Confident-miss rate |
|---|---|---|---|---|---|
| All | 0.947 | 0.794 | 3.6% | 0.982 | 7.0% |
| Major event | **0.753** | 0.774 | 15.2% | 0.888 | 10.2% |

## Main takeaways

- The top-10% label is mostly static: LightGBM beats the 4-week average by only 1.8 F1 points, and
  the deployable model doesn't beat it at all on 2024.
- The surge label is where the model adds value: +10 F1 points over the best baseline, deployable
  as good as full, and weather enters the top inputs.
- Conformal coverage roughly holds overall but breaks on major-event hours (0.75 vs 0.95), and
  the hotspot class is under-covered by a single global threshold. Next step: Mondrian thresholds.
- Mild drift: top-10% F1 slides from ~0.865 (spring 2024) to 0.840 (December).

## Caveats

- `hotspot_top10/nyc_fold3_deployable_conditions_before_expected_attendance.csv` is a snapshot
  of the earlier deployable run; the current deployable model uses expected attendance.
- Flight-delay hours are trivially easy (airport zones), so don't read them as robustness.
- Conditions mix citywide flags (weather, NWS, subway) and zone-level flags (events, flights).
