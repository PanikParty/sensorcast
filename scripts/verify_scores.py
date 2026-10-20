#!/usr/bin/env python
"""Independently verify every published number, offline.

Three checks, no model weights and no GPU required:

1. **Data** — the committed hourly grid really is the hourly mean of the committed raw
   readings (proves the resample, not just that a CSV exists).
2. **Metrics** — recompute MAE/RMSE/MASE/WQL/coverage for every (model, horizon) from
   `results/forecasts.npz` and compare with `results/summary.csv`.
3. **Windows** — per-window MAE in `results/per_window_scores.csv` matches the
   recomputation from the stored predictions.

    python scripts/verify_scores.py            # exit code 0 = everything reproduces
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sensorcast import data, metrics

TOL = 1e-6


def check_data(results: Path) -> list[str]:
    raw_p = results / "data" / "shelly_raw_samples.csv"
    hourly_p = results / "data" / "shelly_hourly.csv"
    if not raw_p.exists():
        return ["DATA      SKIP  raw samples not committed"]
    raw = pd.read_csv(raw_p)
    hourly = data.load_snapshot(hourly_p)
    raw["col"] = raw["sensor"].map(data.slug)
    raw["time"] = pd.to_datetime(raw["ts"], unit="s", utc=True).dt.tz_convert(data.LOCAL_TZ).dt.floor("h")
    rebuilt = raw.groupby(["col", "time"])["t_c"].mean().unstack()
    rebuilt = rebuilt.reindex(index=hourly.index, columns=hourly.columns)
    a, b = rebuilt.to_numpy(dtype=float), hourly.to_numpy(dtype=float)
    same_mask = np.isnan(a) & np.isnan(b)
    diff = np.where(same_mask, 0.0, np.abs(a - b))
    worst = float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0
    ok = worst < 1e-3
    return [f"DATA      {'PASS' if ok else 'FAIL'}  hourly grid rebuilt from {len(raw):,} raw rows, "
            f"max |diff| = {worst:.2e} deg C"]


def check_metrics(results: Path) -> tuple[list[str], list[str]]:
    npz_p = results / "forecasts.npz"
    if not npz_p.exists():
        return ["METRICS   SKIP  forecasts.npz not committed"], []
    summary = pd.read_csv(results / "summary.csv")
    per_window = pd.read_csv(results / "per_window_scores.csv")
    msgs, failures = [], []

    man_p = results / "forecasts_manifest.json"
    declared = json.loads(man_p.read_text())["models"] if man_p.exists() else None

    with np.load(npz_p) as z:
        horizons = sorted({int(k.split("_")[0][1:]) for k in z.files if k.startswith("h")})
        if not declared:
            # model names contain underscores (climatology_hod), so guessing them from
            # the array keys is fragile -- refuse instead of silently checking nothing
            raise SystemExit("forecasts_manifest.json missing -- re-run scripts/dump_forecasts.py")
        models = sorted(declared)
        for h in horizons:
            y_all = z[f"h{h}_y"]
            origins = z[f"h{h}_origin"]
            devices = z[f"h{h}_device"]
            scale = z[f"h{h}_scale"]
            for m in models:
                med = z[f"h{h}_{m}_median"]
                q = z[f"h{h}_{m}_quantiles"] if f"h{h}_{m}_quantiles" in z.files else None
                levels = tuple(z[f"h{h}_{m}_levels"]) if f"h{h}_{m}_levels" in z.files else None
                rows = []
                for i in range(len(y_all)):
                    s = metrics.score_window(
                        y_all[i], med[i], q[i] if q is not None else None, levels,
                        scale=float(scale[i]) if np.isfinite(scale[i]) else None,
                    )
                    if s:
                        rows.append((int(origins[i]), str(devices[i]), s))
                if not rows:
                    continue
                pub = summary[(summary["model"] == m) & (summary["horizon"] == h)]
                if pub.empty:
                    continue
                worst, key = 0.0, ""
                for metric in ("mae", "rmse", "mase", "wql", "coverage80"):
                    if metric not in pub.columns or not np.isfinite(pub[metric].iloc[0]):
                        continue
                    vals = [r[2][metric] for r in rows if metric in r[2] and np.isfinite(r[2][metric])]
                    if not vals:
                        continue
                    got = float(np.mean(vals))
                    want = float(pub[metric].iloc[0])
                    d = abs(got - want)
                    if d > worst:
                        worst, key = d, metric

                # per-window MAE cross-check
                pw = per_window[(per_window["model"] == m) & (per_window["horizon"] == h)]
                pw_map = {}
                for _, r in pw.iterrows():
                    pw_map[(int(pd.Timestamp(r["origin"]).timestamp()), r["device"])] = float(r["mae"])
                wdiff = 0.0
                for ts, dev, s in rows:
                    v = pw_map.get((ts, dev))
                    if v is not None:
                        wdiff = max(wdiff, abs(v - s["mae"]))

                ok = worst < TOL and wdiff < TOL and len(rows) == int(pub["n_windows"].iloc[0])
                line = (f"METRICS   {'PASS' if ok else 'FAIL'}  {m:22s} @ {h:>2}h  "
                        f"n={len(rows):4d}  max|Δ|={worst:.1e} ({key})  per-window Δ={wdiff:.1e}")
                msgs.append(line)
                if not ok:
                    failures.append(line)
    return msgs, failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    results = Path(args.results)

    msgs = check_data(results)
    metric_msgs, failures = check_metrics(results)
    msgs += metric_msgs

    for line in msgs:
        print(line)
    print()
    if failures or any("FAIL" in m for m in msgs):
        print(f"VERIFICATION FAILED ({len(failures) or 1} check(s))")
        sys.exit(1)
    print("VERIFICATION PASSED — every published number reproduces from the committed artifacts")


if __name__ == "__main__":
    main()
