#!/usr/bin/env python
"""Export the raw sensor archive (unresampled) alongside the frozen hourly snapshot.

The hourly CSV is the benchmark input; this adds the underlying readings so a third
party can re-derive the hourly grid themselves instead of trusting our resample.

The export is **pinned to the benchmark's data cutoff** (``t_end`` in
``results/run_manifest.json``).  The live database keeps growing; without the cutoff
the "same" export would describe a different dataset than the one the numbers came
from.  If the rebuild disagrees with the committed snapshot, this script refuses to
overwrite it and says so.

    python scripts/export_raw.py --out results/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from sensorcast import data


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rebuild_hourly(raw: pd.DataFrame, index: pd.DatetimeIndex, columns) -> pd.DataFrame:
    """The hourly grid as a pure function of the raw rows."""
    raw = raw.copy()
    raw["col"] = raw["sensor"].map(data.slug)
    raw["time"] = (
        pd.to_datetime(raw["ts"], unit="s", utc=True).dt.tz_convert(data.LOCAL_TZ).dt.floor("h")
    )
    wide = raw.groupby(["col", "time"])["t_c"].mean().unstack("col")
    return wide.reindex(index=index, columns=columns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(data.DEFAULT_DB))
    ap.add_argument("--out", default="results/data")
    ap.add_argument("--cutoff", default=None,
                    help="ISO local time the benchmark read the DB (see results/run.log); "
                         "defaults to t_end in results/run_manifest.json")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cutoff = pd.Timestamp(args.cutoff) if args.cutoff else None
    expected_rows = None
    man = Path("results/run_manifest.json")
    if man.exists():
        manifest = json.loads(man.read_text())
        expected_rows = manifest.get("raw_samples")
        if cutoff is None:
            # t_end is the start of the last (possibly partial) hour; samples inside it count
            cutoff = pd.Timestamp(manifest["t_end"])
    cutoff_epoch = int(cutoff.timestamp()) if cutoff is not None else None
    if cutoff is not None:
        print(f"cutoff: {cutoff}  (samples after this are excluded)")

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    raw = pd.read_sql_query(
        """
        SELECT s.device_id, COALESCE(d.name, s.device_id) AS sensor,
               s.ts, s.t_c, s.rh, s.bat_pct, s.rssi
          FROM samples s LEFT JOIN devices d ON d.device_id = s.device_id
         ORDER BY s.device_id, s.ts
        """,
        con,
    )
    devs = pd.read_sql_query("SELECT * FROM devices ORDER BY name", con)
    con.close()

    if cutoff_epoch is not None:
        before = len(raw)
        raw = raw[raw["ts"] <= cutoff_epoch].reset_index(drop=True)
        print(f"excluded {before - len(raw):,} samples newer than the cutoff")

    raw["time_local"] = (
        pd.to_datetime(raw["ts"], unit="s", utc=True)
        .dt.tz_convert(data.LOCAL_TZ)
        .dt.strftime("%Y-%m-%d %H:%M:%S%z")
    )
    raw = raw[["device_id", "sensor", "ts", "time_local", "t_c", "rh", "bat_pct", "rssi"]]

    p_raw = out / "shelly_raw_samples.csv"
    raw.to_csv(p_raw, index=False, float_format="%.3f")
    p_dev = out / "devices.csv"
    devs.to_csv(p_dev, index=False)

    p_hourly = out / "shelly_hourly.csv"
    frozen = data.load_snapshot(p_hourly)
    rebuilt = rebuild_hourly(raw, frozen.index, frozen.columns)
    a, b = rebuilt.to_numpy(dtype=float), frozen.to_numpy(dtype=float)
    # a cell present on one side and missing on the other is a mismatch, not a skip
    nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
    both = ~np.isnan(a) & ~np.isnan(b)
    # the snapshot is written at 3 dp, so allow half a unit in the last place
    worst = float(np.abs(a - b)[both].max()) if both.any() else float("inf")
    if nan_mismatch or worst > 5e-4 + 1e-9:
        raise SystemExit(
            f"REFUSING TO OVERWRITE: the raw export rebuilds a different hourly grid "
            f"(max |diff| = {worst:.3e} deg C, {nan_mismatch} missing/present mismatches). The "
            f"cutoff must be the moment the benchmark read the DB (see results/run.log), not the "
            f"start of the last hour -- pass --cutoff explicitly or re-run the benchmark."
        )
    if expected_rows is not None and len(raw) != expected_rows:
        raise SystemExit(f"REFUSING TO OVERWRITE: {len(raw):,} raw rows, but the run read "
                         f"{expected_rows:,} (raw_samples in run_manifest.json)")
    print(f"hourly grid verified against the frozen snapshot (max |diff| = {worst:.2e} deg C, "
          f"{len(raw):,} rows)")

    lines = [
        "# sha256 of the committed data files",
        f"{sha256(p_raw)}  {p_raw.name}",
        f"{sha256(p_hourly)}  {p_hourly.name}",
        f"{sha256(p_dev)}  {p_dev.name}",
    ]
    if cutoff is not None:
        lines.append(f"# data cutoff (local): {cutoff}")
    (out / "CHECKSUMS.txt").write_text("\n".join(lines) + "\n")

    print(f"raw samples   : {len(raw):,} rows -> {p_raw}")
    print(f"devices       : {len(devs)} -> {p_dev}")
    print(f"hourly grid   : {frozen.shape[0]} hours x {frozen.shape[1]} sensors (frozen)")
    print(f"checksums     : {out/'CHECKSUMS.txt'}")


if __name__ == "__main__":
    main()
