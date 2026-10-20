#!/usr/bin/env python
"""Recompute every published score from results/forecasts.npz.

This is what makes the repo auditable: `per_window_scores.csv`, `summary.csv`,
`covariate_arm.csv` and the lead-time error arrays are derived from the committed
predictions, so a third party can reproduce them exactly (floating point included)
without loading a single model.

Two arms live in the archive: the main table (keys ``h{h}_...``) and the covariate
arm (keys ``cov_h{h}_...``), each described by its own manifest.

    python scripts/recompute_scores.py --latency-from <earlier summary.csv>
    python scripts/verify_scores.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sensorcast import evaluate, metrics


def arm_rows(z, manifest, prefix: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Score every (model, horizon) in one arm of the archive."""
    rows, step_err = [], {}
    for h in manifest["horizons"]:
        y = z[f"{prefix}h{h}_y"]
        origins = z[f"{prefix}h{h}_origin"]
        devices = z[f"{prefix}h{h}_device"]
        ctx_cov = z[f"{prefix}h{h}_ctx_cov"]
        y_cov = z[f"{prefix}h{h}_y_cov"]
        scale = z[f"{prefix}h{h}_scale"]
        for m in manifest["models"]:
            key = f"{prefix}h{h}_{m}_median"
            if key not in z.files:
                continue
            med = z[key]
            qkey = f"{prefix}h{h}_{m}_quantiles"
            lkey = f"{prefix}h{h}_{m}_levels"
            q = z[qkey] if qkey in z.files else None
            levels = tuple(z[lkey]) if lkey in z.files else None
            step_err[f"{m}|{h}"] = np.abs(med - y)
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
    return pd.DataFrame(rows), step_err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--latency-from", default=None,
                    help="earlier summary.csv to copy measured ms_per_series from; the "
                         "forecast dump stores predictions, not timings")
    args = ap.parse_args()
    out = Path(args.results)

    main_man = json.loads((out / "forecasts_manifest.json").read_text())
    with np.load(out / "forecasts.npz") as z:
        per_window, step_err = arm_rows(z, main_man, "")
        cov_rows = None
        cov_man_p = out / "covariate_manifest.json"
        if cov_man_p.exists():
            cov_rows, _ = arm_rows(z, json.loads(cov_man_p.read_text()), "cov_")

    if args.latency_from and Path(args.latency_from).exists():
        old = pd.read_csv(args.latency_from)
        if "ms_per_series" in old.columns:
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

    if cov_rows is not None and len(cov_rows):
        cov_agg = (
            cov_rows.groupby(["horizon", "model"])[
                ["mae", "rmse", "mase", "wql", "coverage80", "width80", "bias"]
            ]
            .mean(numeric_only=True)
            .reset_index()
        )
        cov_agg["n_windows"] = cov_rows.groupby(["horizon", "model"]).size().to_numpy()
        cov_agg.to_csv(out / "covariate_arm.csv", index=False)
        print(f"covariate arm   : {len(cov_agg)} (model, horizon) pairs -> {out/'covariate_arm.csv'}")

    print()
    print(agg.sort_values(["horizon", "mase"]).to_string(index=False))


if __name__ == "__main__":
    main()
