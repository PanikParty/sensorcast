# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A zero-shot forecasting benchmark: Mistral Timestral (5M/28M/123M/209M) vs TimesFM 3.0, Chronos-Bolt and training-free baselines on 9 real Shelly H&T temperature sensors. Nothing is trained or fitted — context in, forecast out. CPU only. The committed `results/` directory is the published evidence, not scratch output.

## Commands

```bash
make setup     # uv venv (py3.12) + pip install -e ".[models,dev]"
make test      # pytest -q  (unit tests for metrics/baselines; no model weights needed)
.venv/bin/python -m pytest tests/test_metrics.py::test_mae_rmse_exact -q   # single test
make verify    # scripts/verify_scores.py — recompute every published number offline; exit 0 = reproduces
make smoke     # check each model loads and predicts
make quick     # 24h-only slice -> results/quick/
make bench     # full run -> results/ + REPORT.md (~14 min, downloads ~1.6 GB weights first time)
python scripts/make_report.py                     # rebuild REPORT.md + figures from saved scores
python scripts/run_benchmark.py --snapshot results/data/shelly_hourly.csv   # run without the live DB
```

`run_benchmark.py` flags: `--models` (specs below), `--devices` (sensor slugs), `--horizons`, `--context` (504 h), `--eval-days` (14), `--step` (12 h), `--covariates` (`shed_north`; `''` skips the covariate arm), `--out`, `--no-report`.

Model specs parsed by `models.build_model`: `timestral:<size>`, `chronos:small`, `timesfm`, `baseline:{persistence,seasonal_naive,climatology_hod,drift}`.

## Architecture

Pipeline: `data.py` → `evaluate.py` → `models.py` → `metrics.py` → `report.py`/`plots.py`, driven by `scripts/run_benchmark.py`.

- **Data** (`data.py`): reads `shelly.db` (default `/opt/shelly-dash/data/shelly.db`, only on the collector host), resamples to an hourly wide frame (columns = sensor slugs) in local time. Gaps stay `NaN` — never interpolate them; this is a deliberate part of the protocol. Each run freezes the grid to `results/data/shelly_hourly.csv`.
- **Windows** (`evaluate.build_windows`): rolling origins every `step` hours over the last `eval-days`, each with a `context`-hour history and held-out horizon. Scoring counts only hours with ground truth.
- **Models** (`models.py`): every forecaster implements `predict(ctx (B,T) with NaNs, horizon, known_cov=, past_cov=) -> Forecast(median (B,H), quantiles (B,H,Q))`. Native quantiles are interpolated onto `metrics.COMMON_LEVELS` (0.1..0.9) so WQL compares like with like. Batches are chunked (`CHUNK=48`) to bound RAM on an 11 GB box. Baselines are wrapped by `PointForecaster` and produce no quantiles.
- **Metrics** (`metrics.py`): MASE (primary ranking) divides MAE by the in-context seasonal-naive (lag-24) error; WQL, coverage80, width80 for probabilistic models. Module docstring holds the canonical definitions.
- **Covariate arm**: Timestral (209M, 28M) given the true future `shed_north` reading as a known-future covariate — an oracle upper bound, not deployable. Runs on Timestral's slow covariate path and is stored separately.

## Reproducibility contract (don't break it)

The repo's claim is that a third party can re-derive every number without model weights. Keep these in sync when changing scoring, data handling, or output formats:

- `results/forecasts.npz` holds every prediction. Main-table keys are `h{h}_...`; covariate-arm keys are `cov_h{h}_...`, described by `forecasts_manifest.json` and `covariate_manifest.json` respectively. Written by `scripts/dump_forecasts.py` / `scripts/dump_covariate.py`.
- `scripts/recompute_scores.py` derives `per_window_scores.csv`, `summary.csv`, `covariate_arm.csv` and `step_errors.npz` from the npz; `scripts/verify_scores.py` checks them (and rebuilds the hourly grid from `results/data/shelly_raw_samples.csv`) to `TOL=1e-6`. Run `make verify` after any change to `metrics.py`, `evaluate.py`, or `data.py` resampling.
- `scripts/export_raw.py` (`make data`) re-exports raw samples + `CHECKSUMS.txt`. The cutoff must be the moment the run read the DB (timestamp of the first line in `results/run.log`, currently `--cutoff "2026-10-19 18:24:45-07:00"`), not `t_end`: `t_end` is the start of the last, partial hour. It refuses to overwrite unless the export rebuilds `shelly_hourly.csv` exactly and matches `raw_samples` in the manifest. The live DB is on g9 (`ssh alex@g9`, `/opt/shelly-dash/data/shelly.db`); copy it locally and pass `--db`. Keep the committed `devices.csv` — a fresh export only changes its `last_seen` columns.
- `scripts/verify_scores.py`'s data check compares every cell, NaN-strictly; `tests/test_verify.py` guards it. Statistical claims in REPORT.md come from `report._paired_ci` (bootstrap over forecast origins, not windows — sensors are correlated).
- `scripts/pin_env.py` (`make pin`) records dependency versions and Hugging Face snapshot commits into `run_manifest.json`; `requirements.lock.txt` freezes deps (incl. the Timestral git commit). Adapters do not pass an HF `revision`, so weights resolve to `main`.
- The README's headline numbers and `results/REPORT.md` (generated by `report.build`) must match `summary.csv`.
