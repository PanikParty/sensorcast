#!/usr/bin/env python
"""Rebuild REPORT.md (and the example-forecast figure) from saved result files.

Does not re-run the benchmark: reads results/summary.csv, per_window_scores.csv,
covariate_arm.csv and run_manifest.json.  Only the single example window needs a
model call (~5 s).

    python scripts/make_report.py --results results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sensorcast import baselines, data, evaluate, plots, report
from sensorcast.models import build_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--example-device", default="livingroom")
    ap.add_argument("--horizon", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.results)
    agg = pd.read_csv(out / "summary.csv")
    per_window = pd.read_csv(out / "per_window_scores.csv")
    cov_agg = None
    if (out / "covariate_arm.csv").exists():
        cov_agg = pd.read_csv(out / "covariate_arm.csv")
    cfg = json.loads((out / "run_manifest.json").read_text())
    snap = args.snapshot or cfg.get("snapshot") or str(out / "data" / "shelly_hourly.csv")
    wide = data.load_snapshot(snap)
    coverage = data.coverage(wide)

    # Explicit figure list -- never glob, or stale charts from earlier runs sneak
    # into the report as broken links.
    figures: list[Path] = []

    # regenerate the aggregate figures from the saved scores (no model calls)
    figures += [plots.accuracy_bars(agg, out, "mase"), plots.accuracy_bars(agg, out, "mae")]
    for h in sorted(per_window["horizon"].unique()):
        figures.append(plots.device_heatmap(per_window, out, h))

    npz = out / "step_errors.npz"
    for h in sorted(per_window["horizon"].unique()):
        if npz.exists():
            with np.load(npz) as z:
                errs = {k.split("|")[0]: z[k] for k in z.files if k.endswith(f"|{h}")}
            if errs:
                figures.append(plots.leadtime_curves(errs, out, h))
                continue
        existing = out / f"fig_leadtime_{h}h.png"
        if existing.exists():  # produced by the run itself, not regenerated here
            figures.append(existing)
    figures = sorted(set(figures))

    # refresh the example window chart
    h = args.horizon or max(cfg["horizons"])
    dev = args.example_device
    if dev in wide.columns:
        ws = evaluate.build_windows(
            wide[[dev]], horizon=h, context_len=cfg["context_len"],
            eval_days=cfg["eval_days"], step_hours=cfg["step_hours"],
        )
        j = len(ws) - 1
        series_map, band = {}, None
        for spec in ("timestral:209M", "timesfm"):
            try:
                mdl = build_model(spec)
            except Exception as exc:  # optional dependency missing
                print(f"skip {spec}: {exc}")
                continue
            f = mdl.predict(ws.ctx[j : j + 1], h)
            series_map[mdl.name] = f.median[0]
            if f.quantiles is not None and band is None:
                band = (f.quantiles[0, :, 0], f.quantiles[0, :, -1])
            del mdl
        base = baselines.seasonal_naive(ws.ctx[j].astype(float), h)
        p = plots.example_forecast(
            wide, dev, ws.origin[j], series_map, ws.y[j], band, out, cfg["context_len"], baseline=base
        )
        print(f"example figure -> {p}")
        figures = sorted(set(figures) | {p})

    cfg.setdefault("snapshot", snap)
    p = report.build(out, agg, per_window, cfg, figures, cov_agg, coverage=coverage)
    print(f"report -> {p}")


if __name__ == "__main__":
    main()
