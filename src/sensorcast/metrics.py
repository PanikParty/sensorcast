"""Scoring functions for point and quantile forecasts.

Definitions used everywhere in the report (all NaN-aware -- a window's score only
counts the hours where the ground truth exists):

* ``mae``   mean absolute error of the median forecast, in degrees C.
* ``rmse``  root mean squared error, in degrees C.
* ``mase``  MAE divided by the in-context seasonal-naive error scale
            (mean |y_t - y_{t-24h}| over the context window). <1 beats
            "tomorrow = same hour yesterday" on that window.
* ``wql``   weighted quantile loss, the GIFT-Eval / fev-bench style metric:
            mean over quantile levels of 2*pinball_loss, divided by mean|y|.
            Scale-free, so sensors with bigger swings do not dominate.
* ``coverage80`` share of truths inside the 10-90% interval.
* ``width80``    mean width of that interval, in degrees C.
"""

from __future__ import annotations

import numpy as np

COMMON_LEVELS = tuple(np.round(np.arange(0.1, 1.0, 0.1), 2))  # 0.1 .. 0.9


def _mask(y: np.ndarray) -> np.ndarray:
    return ~np.isnan(y)


def seasonal_scale(ctx: np.ndarray, m: int = 24) -> float:
    """In-context MAE of the seasonal-naive forecast (lag m), NaN-safe."""
    ctx = np.asarray(ctx, dtype=float)
    if ctx.size <= m:
        d = np.abs(np.diff(ctx))
    else:
        d = np.abs(ctx[m:] - ctx[:-m])
    d = d[~np.isnan(d)]
    if d.size == 0:
        return np.nan
    s = float(np.mean(d))
    return s if s > 1e-6 else np.nan


def pinball(y: np.ndarray, q: np.ndarray, level: float) -> float:
    """Mean pinball (quantile) loss for one quantile level."""
    diff = y - q
    return float(np.mean(np.maximum(level * diff, (level - 1.0) * diff)))


def score_window(
    y: np.ndarray,
    median: np.ndarray,
    quantiles: np.ndarray | None = None,
    levels: tuple[float, ...] | None = None,
    scale: float | None = None,
) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    m = np.asarray(median, dtype=float)
    ok = _mask(y) & _mask(m)
    if ok.sum() == 0:
        return {}
    out = {
        "n": int(ok.sum()),
        "mae": float(np.mean(np.abs(m[ok] - y[ok]))),
        "rmse": float(np.sqrt(np.mean((m[ok] - y[ok]) ** 2))),
        "bias": float(np.mean(m[ok] - y[ok])),
    }
    if scale is not None and np.isfinite(scale):
        out["mase"] = out["mae"] / scale

    if quantiles is not None and levels is not None:
        q = np.asarray(quantiles, dtype=float)
        losses = []
        for i, lv in enumerate(levels):
            okq = _mask(y) & _mask(q[:, i])
            if okq.sum():
                losses.append(pinball(y[okq], q[okq, i], lv))
        if losses:
            denom = float(np.mean(np.abs(y[ok])))
            out["wql"] = float(np.mean(losses)) * 2.0 / denom if denom > 1e-9 else np.nan

        lo = q[:, list(levels).index(0.1)] if 0.1 in levels else None
        hi = q[:, list(levels).index(0.9)] if 0.9 in levels else None
        if lo is not None and hi is not None:
            okc = _mask(y) & _mask(lo) & _mask(hi)
            if okc.sum():
                out["coverage80"] = float(np.mean((y[okc] >= lo[okc]) & (y[okc] <= hi[okc])))
                out["width80"] = float(np.mean(hi[okc] - lo[okc]))
    return out


def mase(mae: float, scale: float) -> float:
    return mae / scale if scale and np.isfinite(scale) else float("nan")


def summarize(rows: list[dict]) -> dict[str, float]:
    """Mean of each numeric key across windows (NaN-safe)."""
    if not rows:
        return {}
    keys = {k for r in rows for k in r}
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float)) and np.isfinite(r[k])]
        if vals:
            out[k] = float(np.mean(vals))
    return out
