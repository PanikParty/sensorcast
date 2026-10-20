#!/usr/bin/env python
"""Re-run the models and store every prediction, so scores can be audited offline.

Writes results/forecasts.npz containing, for each horizon, the evaluation windows
(context truth, held-out truth, coverage, MASE scale) and, for each model, the median
and quantile forecasts.  `scripts/verify_scores.py` recomputes the published metrics
from that file alone -- no model weights, no GPU, no re-download.

    python scripts/dump_forecasts.py --snapshot results/data/shelly_hourly.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sensorcast import data, evaluate
from sensorcast.models import build_model
from sensorcast.report import PRETTY  # noqa: F401  (kept for parity of model names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="results/data/shelly_hourly.csv")
    ap.add_argument("--results", default="results")
    ap.add_argument("--models", nargs="+", default=None,
                    help="defaults to the model list recorded in run_manifest.json")
    args = ap.parse_args()

    out = Path(args.results)
    cfg = json.loads((out / "run_manifest.json").read_text())
    specs = args.models or cfg["models"]
    wide = data.load_snapshot(args.snapshot)

    arrays: dict[str, np.ndarray] = {}
    manifest = {"horizons": cfg["horizons"], "models": [], "window_meta": {}}

    for h in cfg["horizons"]:
        ws = evaluate.build_windows(
            wide, horizon=h, context_len=cfg["context_len"],
            eval_days=cfg["eval_days"], step_hours=cfg["step_hours"],
        )
        arrays[f"h{h}_y"] = ws.y
        arrays[f"h{h}_device"] = np.array(ws.device, dtype=str)
        arrays[f"h{h}_origin"] = np.array([int(o.timestamp()) for o in ws.origin], dtype=np.int64)
        arrays[f"h{h}_ctx_cov"] = ws.ctx_cov
        arrays[f"h{h}_y_cov"] = ws.y_cov
        arrays[f"h{h}_scale"] = ws.scale
        manifest["window_meta"][str(h)] = {"n": len(ws), "context_len": cfg["context_len"]}
        print(f"h={h}: {len(ws)} windows", flush=True)

        for spec in specs:
            try:
                model = build_model(spec)
            except Exception as exc:
                print(f"  skip {spec}: {exc}", flush=True)
                continue
            pred = model.predict(ws.ctx, h)
            arrays[f"h{h}_{model.name}_median"] = pred.median
            if pred.quantiles is not None:
                arrays[f"h{h}_{model.name}_quantiles"] = pred.quantiles
                arrays[f"h{h}_{model.name}_levels"] = np.array(pred.levels, dtype=float)
            if model.name not in manifest["models"]:
                manifest["models"].append(model.name)
            print(f"  {model.name:22s} median {pred.median.shape} "
                  f"quantiles {'yes' if pred.quantiles is not None else 'no'}", flush=True)
            del model

    p = out / "forecasts.npz"
    np.savez_compressed(p, **arrays)
    (out / "forecasts_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n-> {p}  ({p.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
