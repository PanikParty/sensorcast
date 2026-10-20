#!/usr/bin/env python
"""Dump the covariate-arm predictions into results/forecasts.npz.

The covariate arm feeds Timestral the *real* future outdoor temperature as a
known-future covariate (an oracle, not a forecast).  It runs on Timestral's slow
covariate path, so it is dumped separately from the main table and merged into the
same archive:

    python scripts/dump_covariate.py            # appends to results/forecasts.npz

Keys are prefixed ``cov_`` so the two arms cannot be confused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sensorcast import data, evaluate
from sensorcast.models import TimestralForecaster

SIZES = ("209M", "28M")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="results/data/shelly_hourly.csv")
    ap.add_argument("--results", default="results")
    ap.add_argument("--covariate", default="shed_north",
                    help="sensor slug used as the outdoor/weather covariate")
    ap.add_argument("--sizes", nargs="+", default=list(SIZES))
    args = ap.parse_args()

    out = Path(args.results)
    cfg = json.loads((out / "run_manifest.json").read_text())
    wide = data.load_snapshot(args.snapshot)
    if args.covariate not in wide.columns:
        raise SystemExit(f"covariate sensor {args.covariate!r} not in snapshot")

    targets = [c for c in wide.columns if c != args.covariate]
    weather = wide[args.covariate].to_numpy(dtype=float)
    arrays: dict[str, np.ndarray] = {}
    manifest = {"covariate": args.covariate, "targets": targets,
                "horizons": cfg["horizons"], "models": [], "window_meta": {}}

    for h in cfg["horizons"]:
        ws = evaluate.build_windows(
            wide[targets], horizon=h, context_len=cfg["context_len"],
            eval_days=cfg["eval_days"], step_hours=cfg["step_hours"],
        )
        arrays[f"cov_h{h}_y"] = ws.y
        arrays[f"cov_h{h}_device"] = np.array(ws.device, dtype=str)
        arrays[f"cov_h{h}_origin"] = np.array([int(o.timestamp()) for o in ws.origin], dtype=np.int64)
        arrays[f"cov_h{h}_ctx_cov"] = ws.ctx_cov
        arrays[f"cov_h{h}_y_cov"] = ws.y_cov
        arrays[f"cov_h{h}_scale"] = ws.scale
        manifest["window_meta"][str(h)] = {"n": len(ws), "context_len": cfg["context_len"]}
        print(f"h={h}: {len(ws)} windows over {len(targets)} targets", flush=True)

        known = np.stack([weather[p - cfg["context_len"]: p + h] for p in ws.p_idx])
        past = np.stack([weather[p - cfg["context_len"]: p] for p in ws.p_idx])

        for size in args.sizes:
            model = TimestralForecaster(size)
            name = f"{model.name}+oracle-weather"
            pred = model.predict(
                ws.ctx, h,
                known_cov=known[:, None, :].astype(np.float32),
                past_cov=past[:, None, :].astype(np.float32),
            )
            arrays[f"cov_h{h}_{name}_median"] = pred.median
            arrays[f"cov_h{h}_{name}_quantiles"] = pred.quantiles
            arrays[f"cov_h{h}_{name}_levels"] = np.array(pred.levels, dtype=float)
            if name not in manifest["models"]:
                manifest["models"].append(name)
            print(f"  {name:34s} median {pred.median.shape}", flush=True)
            del model

    npz_p = out / "forecasts.npz"
    merged: dict[str, np.ndarray] = {}
    if npz_p.exists():
        with np.load(npz_p) as z:
            merged = {k: z[k] for k in z.files}  # keep the main-table arrays
    merged.update(arrays)
    np.savez_compressed(npz_p, **merged)
    (out / "covariate_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n-> {npz_p}  ({npz_p.stat().st_size/1e6:.1f} MB, {len(merged)} arrays)")


if __name__ == "__main__":
    main()
