"""Unit tests for the scoring and baseline maths.  Run: pytest -q"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sensorcast import baselines, data, metrics


def test_mae_rmse_exact():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = np.array([1.0, 2.0, 3.0, 6.0])
    s = metrics.score_window(y, m)
    assert abs(s["mae"] - 0.5) < 1e-9  # only the last point is off by 2
    assert abs(s["rmse"] - 1.0) < 1e-9


def test_window_scores_ignore_nan_truth():
    y = np.array([1.0, np.nan, 3.0])
    m = np.array([1.5, 99.0, 2.0])
    s = metrics.score_window(y, m)
    assert s["n"] == 2
    assert abs(s["mae"] - 0.75) < 1e-9  # 99.0 on a NaN hour must not count


def test_mase_uses_seasonal_scale():
    # exactly periodic at 24 h -> the seasonal-naive scale is zero (degenerate, returns NaN)
    ctx = np.tile([10.0, 12.0] * 12, 2)
    scale = metrics.seasonal_scale(ctx, m=24)
    assert np.isnan(scale)
    # a daily cycle plus a slow drift -> a real, positive scale
    t = np.arange(96)
    ctx2 = 20 + 4 * np.sin(2 * np.pi * t / 24) + 0.05 * t
    s2 = metrics.seasonal_scale(ctx2, m=24)
    assert s2 > 0.05


def test_wql_perfect_forecast_is_zero():
    y = np.array([1.0, 2.0, 3.0])
    q = np.tile(y[:, None], (1, len(metrics.COMMON_LEVELS)))
    s = metrics.score_window(y, y, q, metrics.COMMON_LEVELS, scale=1.0)
    assert s["wql"] < 1e-9
    assert s["coverage80"] == 1.0


def test_persistence_repeats_last_value():
    ctx = np.array([1.0, 2.0, np.nan, 4.0])
    out = baselines.persistence(ctx, 3)
    assert np.allclose(out, 4.0)


def test_seasonal_naive_wraps():
    ctx = np.arange(48, dtype=float)
    out = baselines.seasonal_naive(ctx, 30)
    # first 24 h repeat yesterday's 24 h; beyond that the same block is tiled again
    assert np.allclose(out[:24], ctx[-24:])
    assert np.allclose(out[24:], ctx[-24:][:6])


def test_climatology_has_right_shape_and_no_nan():
    ctx = 20 + 5 * np.sin(np.arange(336) * 2 * np.pi / 24)
    out = baselines.climatology(ctx, 24, hod0=0)
    assert out.shape == (24,)
    assert np.isfinite(out).all()


def test_slug_is_ascii_and_uniqueish():
    assert data.slug("Saskia’s Room") == "saskia_s_room"
    assert data.slug("Shed North") == "shed_north"


def test_snapshot_roundtrip(tmp_path):
    idx = pd.date_range("2026-10-01", periods=5, freq="1h", tz=data.LOCAL_TZ)
    w = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, 5.0]}, index=idx)
    p = data.save_snapshot(w, tmp_path / "snap.csv")
    back = data.load_snapshot(p)
    assert back.shape == w.shape
    assert np.isnan(back["a"].iloc[1])
    assert str(back.index.tz) == data.LOCAL_TZ
