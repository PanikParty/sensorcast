"""Classical, training-free baselines.  These are the numbers a forecast has to beat
before any foundation model is worth its download.
"""

from __future__ import annotations

import numpy as np


def _last_valid(ctx: np.ndarray) -> float:
    finite = np.flatnonzero(~np.isnan(ctx))
    return float(ctx[finite[-1]]) if finite.size else np.nan


def persistence(ctx: np.ndarray, horizon: int, hod0: int = 0) -> np.ndarray:
    """Tomorrow = right now, held flat."""
    return np.full(horizon, _last_valid(ctx), dtype=float)


def seasonal_naive(ctx: np.ndarray, horizon: int, period: int = 24) -> np.ndarray:
    """Tomorrow = same hour yesterday, tiled forward."""
    tail = ctx[-period:] if ctx.size >= period else ctx
    v = _last_valid(tail)
    tail = np.where(np.isnan(tail), v, tail)
    reps = int(np.ceil(horizon / period))
    return np.tile(tail, reps)[:horizon].astype(float)


def climatology(ctx: np.ndarray, horizon: int, hod0: int = 0, period: int = 24) -> np.ndarray:
    """Hour-of-day mean of the context, replayed over the horizon."""
    values = np.full(period, np.nan)
    n = min(ctx.size, 24 * 14)  # last two weeks of context carry the regime
    recent = ctx[-n:]
    hods = (hod0 - n + np.arange(n)) % period
    for h in range(period):
        sel = recent[hods == h]
        sel = sel[~np.isnan(sel)]
        if sel.size:
            values[h] = float(np.mean(sel))
    fallback = float(np.nanmean(ctx)) if np.isfinite(np.nanmean(ctx)) else np.nan
    values = np.where(np.isnan(values), fallback, values)
    return values[(hod0 + np.arange(horizon)) % period]


def drift(ctx: np.ndarray, horizon: int, hod0: int = 0, period: int = 24) -> np.ndarray:
    """Seasonal naive plus the average hourly slope of the last 3 days."""
    base = seasonal_naive(ctx, horizon, period)
    n = min(ctx.size, 72)
    recent = ctx[-n:]
    if np.isfinite(recent).sum() < 6:
        return base
    x = np.arange(n)
    ok = ~np.isnan(recent)
    slope = np.polyfit(x[ok], recent[ok], 1)[0]
    return base + slope * np.arange(1, horizon + 1)


BASELINES = {
    "persistence": persistence,
    "seasonal_naive": seasonal_naive,
    "climatology_hod": climatology,
    "drift": drift,
}
