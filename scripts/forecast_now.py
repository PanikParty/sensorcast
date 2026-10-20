#!/usr/bin/env python
"""Print the next 24 h of Timestral temperature for every sensor, from the live archive.

    python scripts/forecast_now.py                  # all sensors, 209M
    python scripts/forecast_now.py --size 28M --json

This is the practical by-product of the benchmark: the 28M model is ~5x faster than
209M for ~2% worse MASE (see results/REPORT.md), so it is the default choice when you
only want a household number in a second.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from sensorcast import data
from sensorcast.models import TimestralForecaster


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="28M", choices=["5M", "28M", "123M", "209M"])
    ap.add_argument("--context", type=int, default=504)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--devices", nargs="*", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    wide = data.load_hourly()
    cols = args.devices or list(wide.columns)
    ctx = np.stack(
        [wide[c].to_numpy(dtype=np.float32)[-args.context:] for c in cols]
    )
    last_obs = {c: wide[c].dropna().index[-1] for c in cols}

    model = TimestralForecaster(args.size)
    f = model.predict(ctx, args.horizon)
    names = data.device_names()
    start = wide.index[-1] + pd.Timedelta(hours=1)

    rows = []
    for i, c in enumerate(cols):
        q = f.quantiles[i]
        med = f.median[i]
        rows.append(
            {
                "sensor": names.get(c, c),
                "last_reading": f"{last_obs[c]:%m-%d %H:%M}",
                "now_C": round(float(ctx[i][~np.isnan(ctx[i])][-1]), 1),
                "min_C": round(float(np.min(med)), 1),
                "max_C": round(float(np.max(med)), 1),
                "+6h_C": round(float(med[5]), 1),
                "+12h_C": round(float(med[11]), 1),
                "+24h_C": round(float(med[-1]), 1),
                "p10_min_C": round(float(np.min(q[:, 0])), 1),
                "p90_max_C": round(float(np.max(q[:, -1])), 1),
            }
        )

    if args.json:
        print(json.dumps({"issued": str(start), "model": model.name, "rows": rows}, indent=2))
        return

    df = pd.DataFrame(rows)
    print(f"\ntimestral-{args.size}   forecast issued {start:%Y-%m-%d %H:%M} local"
          f"   ({args.horizon} h horizon, {args.context} h context)\n")
    print(df.to_string(index=False))
    print("\n  now_C = last observed   p10/p90 = 10-90% band extremes over the horizon\n")


if __name__ == "__main__":
    main()
