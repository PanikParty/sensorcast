# sensorcast

Zero-shot time-series forecasting on **real sensor data from our house** — 9 Shelly H&T
sensors reporting temperature and humidity every ~40 minutes for the last ~7 weeks.

The question: **does Mistral's Timestral (209M) actually beat the other foundation models
on our data, at our scale, on a CPU?** This repo answers it with a reproducible benchmark
rather than a leaderboard screenshot.

```
                    MASE @ 24h   (lower is better; 1.0 = "same hour yesterday")
  1. Timestral-209M   ███████████████████████████▏ 0.802
  2. Timestral-123M   ███████████████████████████▎ 0.810
  3. Timestral-5M     ███████████████████████████▍ 0.812
  4. Timestral-28M    ███████████████████████████▌ 0.816
  5. TimesFM 3.0      ███████████████████████████████ 0.905
  6. Chronos-Bolt     ████████████████████████████████ 0.923
  9. seasonal naive   ███████████████████████████████████████ 1.250
```

Three things fell out of the run (full tables in [`results/REPORT.md`](results/REPORT.md)):

1. **Timestral wins on aggregate, and it is not close at 6 h.** MASE 0.427 vs 1.111
   for seasonal naive — a 2.6× reduction in scaled error with no training.
2. **Model size is nearly free of value past 5M at 24 h.** 5M → 209M buys **1.3 %** MASE
   for **36× the CPU** (5 → 181 ms/series on this box). At 6 h the jump is worth 10.5 %.
   The 28M checkpoint is the sane default for a household forecast.
3. **The biggest lever is not the model at all.** Handing Timestral the actual outdoor
   temperature as a known-future covariate cut 24 h MASE **0.802 → 0.524 (−35 %)** — more
   than any size step, and more than any competitor managed. (Oracle experiment: the
   covariate is the true future outdoor reading, not a forecast.)

TimesFM is genuinely competitive at 6 h (best on 5 of 9 individual sensors) but its 80 %
interval only covers 70 % of outcomes — it *looks* sharper than it is on quantile loss.
Timestral's bands land at 80–83 % where 80 % was asked for.

## Why this exists

The launch claims for Timestral (first on GIFT-Eval, TIME and fev-bench; a 209M model
beating 2.5B Toto 2.0) are **self-reported and were not externally reviewed at release**.
Our sensors are the opposite kind of data: low resolution, heavy dropouts, strong diurnal
structure, one house, one season. If a foundation model is going to be useful at home —
for HVAC scheduling, freeze alerts in the shed, or just "is the garage going to cook today" —
it has to win *here*, not on a benchmark of curated series.

## What is measured

* **Zero-shot only.** No fitting, no fine-tuning, no per-sensor calibration. Context in,
  forecast out. That is the whole promise of a time-series foundation model.
* **Rolling origins.** Every 12 h across the last 14 days of each sensor's record, with a
  3-week (504 h) context and a held-out horizon of 1/6/24 h.
* **Real missing data.** The archive is only 55–85 % complete per sensor (gaps up to 73 h).
  Holes stay as `NaN` in the context instead of being interpolated away.
* **Scored only where truth exists.** A drop-out mid-horizon costs a sensor that window's
  remaining hours, not the whole window.

Metrics: `MAE`, `RMSE` (°C), `MASE` (÷ in-context seasonal-naive error — the primary
ranking), `WQL` (weighted quantile loss, GIFT-Eval style), and 80 % interval coverage/width
for the probabilistic models.

## Models

| family | checkpoints | params | notes |
|---|---|---|---|
| Mistral **Timestral** | 5M / 28M / 123M / 209M | 5M–209M | 21 quantiles, past + known-future covariates |
| Google **TimesFM 3.0** | `google/timesfm-3.0-pytorch` | 200M | 9 quantiles |
| Amazon **Chronos-Bolt** | small | ~48M | 9 quantiles, fastest |
| Baselines | persistence, seasonal naive (24 h), hour-of-day climatology, drift | — | training-free |

## Quickstart

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[models,dev]"

python scripts/run_benchmark.py                 # full run -> results/ + REPORT.md
python scripts/run_benchmark.py --horizons 24 --devices livingroom kitchen \
       --models timestral:5M baseline:seasonal_naive      # quick slice
python scripts/forecast_now.py --size 28M       # tonight's 24 h forecast, all sensors
python scripts/make_report.py                   # rebuild REPORT.md + figures from saved scores
```

`forecast_now.py` prints the practical product — the next 24 h for every sensor, with the
10–90 % band:

```
       sensor last_reading  now_C  min_C  max_C  +6h_C  +12h_C  +24h_C  p10_min_C  p90_max_C
       Bedroom  10-19 18:00   26.4   23.7   26.5   25.2    23.7    26.4       22.5       27.9
        garage  10-19 18:00   31.1   24.2   33.0   26.7    24.2    31.2       23.1       34.6
    Shed North  10-19 18:00   22.0   16.4   28.3   18.0    16.4    22.3       14.7       32.9
```

No GPU needed. The full run (every model, both horizons, plus the covariate arm) takes ~14 minutes on a 4-core Intel N150 with 11 GB of RAM; the model pass alone is ~5 minutes.
First run downloads ~1.6 GB of weights from the Hugging Face Hub into `~/.cache/huggingface`.

To benchmark a frozen copy of the data (no live DB needed):

```bash
python scripts/run_benchmark.py --snapshot results/data/shelly_hourly.csv
```

## Layout

```
src/sensorcast/
  data.py       sqlite -> hourly wide frame (NaN kept), CSV snapshot/load
  baselines.py  persistence / seasonal naive / climatology / drift
  models.py     one adapter per forecaster; quantiles resampled to a common grid
  evaluate.py   rolling-origin windows, scoring, aggregation
  metrics.py    MAE/RMSE/MASE/WQL/coverage
  plots.py      accuracy bars, lead-time curves, per-sensor heatmap, example forecast
  report.py     renders results/REPORT.md
scripts/
  run_benchmark.py   end-to-end driver (models, covariate arm, figures, report)
  make_report.py     rebuild REPORT.md + figures from the saved scores
  forecast_now.py    practical 24 h forecast for every sensor, from the live DB
  smoke_models.py    verifies each model loads and predicts
tests/          metric and baseline unit tests
results/        data snapshot, per-window scores, summary, figures, REPORT.md
```

## Data

Read live from `shelly.db` (`/opt/shelly-dash/data/shelly.db`, written by the
`shelly-collector` unit) and frozen to `results/data/shelly_hourly.csv` on every run so a
result is always reproducible after the archive moves on.

Columns are the sensor labels: `bedroom, garage, kitchen, kyan, livingroom, office,
saskia_s_room, shed, shed_north`. `shed_north` is the sun-exposed outdoor sensor and doubles
as the weather covariate in the covariate arm.

## Caveats worth repeating

* 54 days, one house, one season. Diurnal structure is well sampled; seasonal drift is not.
* Sensors in one building are correlated, so windows are not independent samples.
* Mistral's benchmark wins are self-reported; this benchmark is independent of them but far
  smaller in scope.
* The covariate arm uses the **actual** outdoor temperature across the horizon — an oracle,
  not a forecast. It bounds what weather information could buy; it is not deployable as-is.

## License

MIT for this code. Model weights keep their own licenses (Timestral weights are modified MIT;
TimesFM and Chronos-Bolt are Apache-2.0).
