#!/usr/bin/env python
"""Recompute every published score from results/forecasts.npz.

This is what makes the repo auditable: `per_window_scores.csv`, `summary.csv` and the
lead-time error arrays are derived from the committed predictions, so a third party can
reproduce them exactly (floating point included) without loading a single model.

    python scripts/recompute_scores.py && python scripts/verify_scores.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sensorcast import evaluate, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--latency-from", default=None,
                    help="earlier summary.csv to copy measured ms_per_series from; the "
                         "forecast dump stores predictions, not timings")
    args = ap.parse_args()
    out = Path(args.results)

    man = json.loads((out / "forecasts_manifest.json").read_text())
    rows, step_err = [], {}

    with np.load(out / "forecasts.npz") as z:
        for h in man["horizons"]:
            y = z[f"h{h}_y"]
            origins = z[f"h{h}_origin"]
            devices = z[f"h{h}_device"]
            ctx_cov = z[f"h{h}_ctx_cov"]
            y_cov = z[f"h{h}_y_cov"]
            scale = z[f"h{h}_scale"]
            for m in man["models"]:
                key = f"h{h}_{m}_median"
                if key not in z.files:
                    continue
                med = z[key]
                q = z[f"h{h}_{m}_quantiles"] if f"h{h}_{m}_quantiles" in z.files else None
                levels = tuple(z[f"h{h}_{m}_levels"]) if f"h{h}_{m}_levels" in z.files else None
                err = np.abs(med - y)
                step_err[f"{m}|{h}"] = err
                for i in range(len(y)):
                    s = metrics.score_window(
                        y[i], med[i], q[i] if q is not None else None, levels,
                        scale=float(scale[i]) if np.isfinite(scale[i]) else None,
                    )
                    if not s:
                        continue
                    s.update(
                        model=m, horizon=int(h), device=str(devices[i]),
                        origin=pd.Timestamp(int(origins[i]), unit="s", tz="UTC")
                        .tz_convert("America/Los_Angeles").isoformat(),
                        ctx_cov=float(ctx_cov[i]), y_cov=float(y_cov[i]),
                        scale=float(scale[i]) if np.isfinite(scale[i]) else np.nan,
                    )
                    rows.append(s)

    per_window = pd.DataFrame(rows)
    if args.latency_from:
        old = pd.read_csv(args.latency_from)
        lat = old.drop_duplicates(["model", "horizon"]).set_index(["model", "horizon"])["ms_per_series"]
        per_window["ms_per_series"] = [
            float(lat.get((m, h), np.nan)) for m, h in zip(per_window["model"], per_window["horizon"])
        ]
    per_window.to_csv(out / "per_window_scores.csv", index=False)
    agg = evaluate.aggregate(per_window)
    agg.to_csv(out / "summary.csv", index=False)
    np.savez_compressed(out / "step_errors.npz", **step_err)

    print(f"per-window rows : {len(per_window):,} -> {out/'per_window_scores.csv'}")
    print(f"aggregates      : {len(agg)} (model, horizon) pairs -> {out/'summary.csv'}")
    print(f"step errors     : {len(step_err)} arrays -> {out/'step_errors.npz'}")
    print()
    print(agg.sort_values(["horizon", "mase"]).to_string(index=False))


if __name__ == "__main__":
    main()
