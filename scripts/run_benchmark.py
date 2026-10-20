#!/usr/bin/env python
"""Run the sensorcast benchmark end to end and write results/ + REPORT.md.

    python scripts/run_benchmark.py --models timestral:209M chronos:small baseline:seasonal_naive

Every model is zero-shot; nothing is trained.  Results land in ``results/``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sensorcast import data, evaluate, plots, report
from sensorcast.models import build_model

DEFAULT_MODELS = [
    "timestral:5M",
    "timestral:28M",
    "timestral:123M",
    "timestral:209M",
    "timesfm",
    "chronos:small",
    "baseline:persistence",
    "baseline:seasonal_naive",
    "baseline:climatology_hod",
    "baseline:drift",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(data.DEFAULT_DB))
    ap.add_argument("--snapshot", default=None, help="read a frozen CSV instead of the live DB")
    ap.add_argument("--context", type=int, default=504)
    ap.add_argument("--horizons", type=int, nargs="+", default=[24, 6])
    ap.add_argument("--eval-days", type=int, default=14)
    ap.add_argument("--step", type=int, default=12, help="hours between rolling origins")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--devices", nargs="*", default=None, help="subset of sensor slugs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--covariates", default="shed_north", help="sensor slug used as oracle weather; '' to skip")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.snapshot:
        wide = data.load_snapshot(args.snapshot)
        raw_n = 0
    else:
        raw = data.load_raw(args.db)
        raw_n = len(raw)
        wide = data.load_hourly(args.db)
    cols = args.devices or list(wide.columns)
    wide = wide[cols]
    cov = data.coverage(wide)
    log(f"data: {wide.shape[1]} sensors x {wide.shape[0]} hours, NaN {np.isnan(wide.to_numpy()).mean():.1%}")
    log("coverage: " + ", ".join(f"{k}={v:.0%}" for k, v in cov.items()))

    if not args.snapshot:
        snap = data.save_snapshot(wide, out / "data" / "shelly_hourly.csv")
        log(f"snapshot -> {snap}")

    names = {c: data.device_names(args.db).get(c, c) for c in cols} if not args.snapshot else {c: c for c in cols}

    # Windows are built once per horizon and reused by every model.
    windows = {}
    for h in args.horizons:
        windows[h] = evaluate.build_windows(
            wide, horizon=h, context_len=args.context, eval_days=args.eval_days, step_hours=args.step
        )
        log(f"horizon {h:>3}h: {len(windows[h])} windows across {len(set(windows[h].device))} sensors")

    frames, step_err, lat = [], {}, []
    for spec in args.models:
        t0 = time.time()
        try:
            model = build_model(spec)
        except Exception as e:  # missing optional dependency, bad spec, ...
            log(f"SKIP {spec}: {type(e).__name__}: {e}")
            continue
        log(f"{model.name}: loaded in {time.time()-t0:.1f}s")
        for h in args.horizons:
            ws = windows[h]
            t0 = time.time()
            df, err = evaluate.run_model(model, ws)
            if df.empty:
                log(f"  {model.name} @ {h}h: no scores")
                continue
            frames.append(df)
            step_err[(model.name, h)] = err
            m = df.iloc[0]
            log(
                f"  {model.name:20s} @ {h:>2}h  MAE {df['mae'].mean():5.2f}  MASE {df['mase'].mean():5.3f}  "
                f"{df['ms_per_series'].iloc[0]:6.0f} ms/series  ({time.time()-t0:.1f}s)"
            )
        del model

    if not frames:
        raise SystemExit("no model produced scores")

    per_window = pd.concat(frames, ignore_index=True)
    per_window.to_csv(out / "per_window_scores.csv", index=False)
    if step_err:
        np.savez_compressed(
            out / "step_errors.npz", **{f"{m}|{h}": v for (m, h), v in step_err.items()}
        )
        log(f"step errors -> {out/'step_errors.npz'}")
    agg = evaluate.aggregate(per_window)
    agg.to_csv(out / "summary.csv", index=False)
    log("\n" + agg.to_string(index=False))

    # ---- covariate arm: oracle outdoor temperature as a known-future covariate
    cov_agg = None
    if args.covariates and args.covariates in cols and len(cols) > 1:
        targets = [c for c in cols if c != args.covariates]
        weather = wide[args.covariates].to_numpy(dtype=float)
        rows = []
        for size in ("209M", "28M"):
            if f"timestral:{size}" not in args.models:
                continue
            try:
                m = build_model(f"timestral:{size}")
            except Exception as e:
                log(f"covariate arm skip {size}: {e}")
                continue
            for h in args.horizons:
                ws = evaluate.build_windows(
                    wide[targets], horizon=h, context_len=args.context,
                    eval_days=args.eval_days, step_hours=args.step,
                )
                # slice the outdoor sensor onto every window: past always, horizon = oracle
                known = np.stack([weather[p - args.context: p + h] for p in ws.p_idx])
                past = np.stack([weather[p - args.context: p] for p in ws.p_idx])
                df, _ = evaluate.run_model(m, ws, known_cov=known[:, None, :], past_cov=past[:, None, :])
                if df.empty:
                    continue
                r = df.mean(numeric_only=True).to_dict()
                r.update(model=f"{m.name}+oracle-weather", horizon=h, n_windows=len(df))
                rows.append(r)
                log(f"  {r['model']:32s} @ {h:>2}h  MAE {r['mae']:5.2f}  MASE {r['mase']:5.3f}")
            del m
        cov_agg = pd.DataFrame(rows) if rows else None
        if cov_agg is not None:
            cov_agg.to_csv(out / "covariate_arm.csv", index=False)

    # ---- figures
    figures = [plots.accuracy_bars(agg, out, "mase"), plots.accuracy_bars(agg, out, "mae")]
    for h in args.horizons:
        figures.append(plots.leadtime_curves({k[0]: v for k, v in step_err.items() if k[1] == h}, out, h))
        figures.append(plots.device_heatmap(per_window, out, h))

    # example window: the most recent forecast for one representative sensor
    if frames:
        from sensorcast import baselines

        h = max(args.horizons)
        ws = windows[h]
        dev = "livingroom" if "livingroom" in ws.device else ws.device[0]
        idxs = [j for j, d in enumerate(ws.device) if d == dev]
        j = idxs[-1]
        series_map, band = {}, None
        for spec in ("timestral:209M", "timesfm"):
            try:
                mdl = build_model(spec)
            except Exception:
                continue
            f = mdl.predict(ws.ctx[j : j + 1], h)
            series_map[mdl.name] = f.median[0]
            if f.quantiles is not None and band is None:
                band = (f.quantiles[0, :, 0], f.quantiles[0, :, -1])
            del mdl
        base = baselines.seasonal_naive(ws.ctx[j].astype(float), h)
        figures.append(
            plots.example_forecast(
                wide, dev, ws.origin[j], series_map, ws.y[j], band, out,
                args.context, baseline=base,
            )
        )

    cfg = {
        "raw_samples": raw_n,
        "span_days": (wide.index[-1] - wide.index[0]).total_seconds() / 86400,
        "t_start": str(wide.index[0]),
        "t_end": str(wide.index[-1]),
        "freq": "1h",
        "n_hours": len(wide),
        "n_devices": len(wide.columns),
        "nan_frac": float(np.isnan(wide.to_numpy()).mean()),
        "context_len": args.context,
        "horizons": args.horizons,
        "eval_days": args.eval_days,
        "step_hours": args.step,
        "models": args.models,
        "devices": cols,
        "torch_threads": int(__import__("torch").get_num_threads()),
    }
    report.write_manifest(out, cfg)
    if not args.no_report:
        p = report.build(out, agg, per_window, cfg, figures, cov_agg, coverage=cov)
        log(f"report -> {p}")


if __name__ == "__main__":
    main()
