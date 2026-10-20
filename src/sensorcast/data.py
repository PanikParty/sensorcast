"""Load the Shelly H&T sensor archive into an hourly, wide temperature frame.

The family's sensors report at irregular intervals (25-95 min median, gaps up to
~3 days).  We resample to a fixed hourly grid in local time and *keep* the holes
as NaN: handling missing observations is part of what these forecasters claim to
do well, so the holes are signal, not noise to be papered over.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DB = Path("/opt/shelly-dash/data/shelly.db")
LOCAL_TZ = "America/Los_Angeles"


def slug(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower()
    return s or "sensor"


def load_raw(db_path: str | Path = DEFAULT_DB) -> pd.DataFrame:
    """Read every sample joined to its device label."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(
            """
            SELECT s.device_id,
                   COALESCE(d.name, s.device_id) AS name,
                   s.ts, s.t_c, s.rh
              FROM samples s
              LEFT JOIN devices d ON d.device_id = s.device_id
            """,
            con,
        )
    finally:
        con.close()
    if df.empty:
        raise SystemExit(f"no samples in {db_path}")
    return df


def load_hourly(
    db_path: str | Path = DEFAULT_DB,
    value: str = "t_c",
    tz: str = LOCAL_TZ,
    freq: str = "1h",
) -> pd.DataFrame:
    """Return (time x sensor) frame on a regular local-time grid, NaN where missing.

    Columns are ASCII slugs; call :func:`device_names` for the display labels.
    """
    if value not in {"t_c", "rh"}:
        raise ValueError("value must be 't_c' or 'rh'")
    df = load_raw(db_path)
    df["time"] = (
        pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(tz).dt.floor(freq)
    )
    df["col"] = df["name"].map(slug)
    grouped = df.groupby(["col", "time"])[value].mean().reset_index()
    wide = grouped.pivot(index="time", columns="col", values=value).sort_index()
    grid = pd.date_range(wide.index.min(), wide.index.max(), freq=freq, tz=tz)
    wide = wide.reindex(grid)
    wide.index.name = "time"
    return wide.astype("float64")


def device_names(db_path: str | Path = DEFAULT_DB) -> dict[str, str]:
    """slug -> human label."""
    df = load_raw(db_path)
    return {slug(n): n for n in df["name"].unique()}


def coverage(wide: pd.DataFrame) -> pd.Series:
    """Fraction of hours present per sensor."""
    return wide.notna().mean().sort_values(ascending=False)


def series(wide: pd.DataFrame, col: str) -> np.ndarray:
    return wide[col].to_numpy(dtype="float32")


def save_snapshot(wide: pd.DataFrame, path: str | Path) -> Path:
    """Freeze the dataset so the benchmark is reproducible without the live DB.

    Timestamps are written as naive *local wall-clock* time; ``load_snapshot``
    re-attaches the local timezone.  Writing tz-aware strings would come back as a
    fixed UTC offset instead of the named zone.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = wide.copy()
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.to_csv(path, float_format="%.3f")
    return path


def load_snapshot(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    wide = pd.read_csv(path, index_col=0, parse_dates=True)
    if wide.index.tz is None:
        wide.index = wide.index.tz_localize(LOCAL_TZ)
    else:
        wide.index = wide.index.tz_convert(LOCAL_TZ)
    return wide


if __name__ == "__main__":  # quick look
    w = load_hourly()
    cov = coverage(w)
    print(f"{w.shape[0]} hours  {w.index.min():%Y-%m-%d} -> {w.index.max():%Y-%m-%d}")
    for col, c in cov.items():
        x = w[col].dropna()
        print(f"  {col:14s} coverage={c:5.1%}  n={len(x):5d}  {x.min():5.1f}-{x.max():5.1f} C")
    print(np.isnan(w.to_numpy()).mean().round(3), "overall NaN fraction")
