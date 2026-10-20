"""Rolling-origin evaluation harness.

Protocol
--------
* Grid: hourly, local time, NaNs preserved.
* For every sensor, cast origins backwards from the end of the record every
  ``step_hours`` (default 12 h) across the last ``eval_days`` (default 14 d).
* Each window gets a ``context_len`` (default 504 h = 3 weeks) history and a
  horizon of H hours of *held-out* truth.
* A window is kept only if the context has >= ``min_ctx_cov`` valid hours and the
  held-out horizon has >= ``min_y_cov`` valid hours.  Scores are computed only on
  the hours where the truth exists, so a sensor that drops out mid-horizon is
  still scored on the part it reported.

Nothing is fitted on this data: every model here is zero-shot, so there is no
train/test leakage beyond the deliberate choice of the context window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics


@dataclass
class WindowSet:
    horizon: int
    context_len: int
    ctx: np.ndarray  # (N, T)
    y: np.ndarray  # (N, H)
    hods: np.ndarray  # hour-of-day of the first horizon step
    device: list[str]
    origin: list[pd.Timestamp]
    ctx_cov: np.ndarray
    y_cov: np.ndarray
    scale: np.ndarray = field(default_factory=lambda: np.array([]))
    p_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    def __len__(self) -> int:
        return len(self.device)


def build_windows(
    wide: pd.DataFrame,
    horizon: int,
    context_len: int = 504,
    eval_days: int = 14,
    step_hours: int = 12,
    min_ctx_cov: float = 0.3,
    min_y_cov: float = 0.5,
    columns: list[str] | None = None,
) -> WindowSet:
    cols = columns or list(wide.columns)
    ctxs, ys, hods, devices, origins, ccov, ycov, scales, pidx = [], [], [], [], [], [], [], [], []
    for col in cols:
        arr = wide[col].to_numpy(dtype=float)
        n = len(arr)
        start_floor = n - 1 - eval_days * 24
        p = n - horizon
        while p > context_len and p >= max(start_floor, context_len):
            ctx = arr[p - context_len : p]
            y = arr[p : p + horizon]
            cc = float(np.mean(~np.isnan(ctx)))
            yc = float(np.mean(~np.isnan(y)))
            if cc >= min_ctx_cov and yc >= min_y_cov:
                ctxs.append(ctx.astype(np.float32))
                ys.append(y)
                hods.append(wide.index[p].hour)
                devices.append(col)
                origins.append(wide.index[p])
                ccov.append(cc)
                ycov.append(yc)
                scales.append(metrics.seasonal_scale(ctx))
                pidx.append(p)
            p -= step_hours
    if not ctxs:
        raise SystemExit("no usable evaluation windows -- loosen thresholds")
    return WindowSet(
        horizon=horizon,
        context_len=context_len,
        ctx=np.vstack(ctxs),
        y=np.vstack(ys),
        hods=np.array(hods),
        device=devices,
        origin=origins,
        ctx_cov=np.array(ccov),
        y_cov=np.array(ycov),
        scale=np.array(scales, dtype=float),
        p_idx=np.array(pidx, dtype=int),
    )


def score_forecasts(
    ws: WindowSet,
    model_name: str,
    median: np.ndarray,
    quantiles: np.ndarray | None,
    levels: tuple[float, ...] | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-window scores + (N,H) absolute-error matrix for lead-time curves."""
    rows, abs_err = [], np.full(ws.y.shape, np.nan)
    for i in range(len(ws)):
        abs_err[i] = np.abs(median[i] - ws.y[i])
        s = metrics.score_window(
            ws.y[i],
            median[i],
            quantiles[i] if quantiles is not None else None,
            levels,
            scale=float(ws.scale[i]) if np.isfinite(ws.scale[i]) else None,
        )
        if not s:
            continue
        s.update(
            model=model_name,
            horizon=ws.horizon,
            device=ws.device[i],
            origin=ws.origin[i].isoformat(),
            ctx_cov=float(ws.ctx_cov[i]),
            y_cov=float(ws.y_cov[i]),
            scale=float(ws.scale[i]) if np.isfinite(ws.scale[i]) else np.nan,
        )
        rows.append(s)
    return pd.DataFrame(rows), abs_err


def run_model(model, ws: WindowSet, **kw) -> tuple[pd.DataFrame, np.ndarray]:
    pred = model.predict(ws.ctx, ws.horizon, hods=ws.hods, **kw)
    df, abs_err = score_forecasts(ws, model.name, pred.median, pred.quantiles, pred.levels)
    df["ms_per_series"] = pred.ms_per_series
    df["seconds"] = pred.seconds
    return df, abs_err


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per (model, horizon)."""
    num = ["mae", "rmse", "mase", "wql", "coverage80", "width80", "bias", "ms_per_series"]
    have = [c for c in num if c in df.columns]
    g = df.groupby(["horizon", "model"])[have].mean(numeric_only=True)
    g["n_windows"] = df.groupby(["horizon", "model"]).size()
    return g.reset_index().sort_values(["horizon", "mase"])
