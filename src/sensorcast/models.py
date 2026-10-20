"""Adapters that put every forecaster behind one call:

    predict(ctx, horizon) -> Forecast(median=(B,H), quantiles=(B,H,len(COMMON_LEVELS)))

``ctx`` is (B, T) float32 with NaN holes.  Quantiles are always resampled onto the
common level grid so the weighted-quantile-loss comparison is apples to apples.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .metrics import COMMON_LEVELS

CHUNK = 48  # series per forward pass; keeps peak RSS sane on a 11 GB box


@dataclass
class Forecast:
    model: str
    median: np.ndarray
    quantiles: np.ndarray | None = None
    levels: tuple[float, ...] | None = None
    seconds: float = 0.0
    n_series: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def ms_per_series(self) -> float:
        return 1000.0 * self.seconds / max(self.n_series, 1)


def interp_levels(q: np.ndarray, levels_src, levels_dst=COMMON_LEVELS) -> np.ndarray:
    """(B,H,Q_src) on levels_src -> (B,H,Q_dst) by linear interpolation in quantile space."""
    q = np.asarray(q, dtype=float)
    levels_src = np.asarray(levels_src, dtype=float)
    out = np.empty(q.shape[:2] + (len(levels_dst),), dtype=float)
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            out[b, h] = np.interp(levels_dst, levels_src, q[b, h])
    return out


class BaseForecaster:
    name = "base"
    probabilistic = True

    def predict(
        self,
        ctx: np.ndarray,
        horizon: int,
        *,
        known_cov: np.ndarray | None = None,
        past_cov: np.ndarray | None = None,
    ) -> Forecast:
        raise NotImplementedError


class TimestralForecaster(BaseForecaster):
    """Mistral Timestral, 5M / 28M / 123M / 209M."""

    probabilistic = True

    def __init__(self, size: str = "209M", device: str = "cpu"):
        from timestral import TimestralForecaster as _TF

        self.name = f"timestral-{size}"
        t0 = time.time()
        self.model = _TF.from_pretrained(f"mistralai/Timestral-{size}", device=device)
        self.load_seconds = time.time() - t0

    def predict(self, ctx, horizon, *, known_cov=None, past_cov=None, hods=None) -> Forecast:
        ctx = np.asarray(ctx, dtype=np.float32)
        B = ctx.shape[0]
        med = np.empty((B, horizon))
        qs = np.empty((B, horizon, len(COMMON_LEVELS)))
        t0 = time.time()
        for i in range(0, B, CHUNK):
            sl = slice(i, min(i + CHUNK, B))
            x = ctx[sl].reshape(-1, 1, ctx.shape[1])
            kw = {}
            if past_cov is not None:
                pc = np.asarray(past_cov[sl], dtype=np.float32)
                kw["past_covariates"] = pc[:, None, :] if pc.ndim == 2 else pc
            if known_cov is not None:
                kc = np.asarray(known_cov[sl], dtype=np.float32)
                kw["known_covariates"] = kc[:, None, :] if kc.ndim == 2 else kc
            f = self.model.forecast(x, prediction_length=horizon, **kw)
            q = np.asarray(f.quantiles)[:, 0]  # (b,H,21)
            q = interp_levels(q, f.quantile_levels)
            qs[sl] = q
            med[sl] = q[:, :, COMMON_LEVELS.index(0.5)]
        return Forecast(self.name, med, qs, COMMON_LEVELS, time.time() - t0, B,
                        {"load_seconds": self.load_seconds})


class ChronosBoltForecaster(BaseForecaster):
    """Amazon Chronos-Bolt (small/base) as a second general-purpose TSFM."""

    probabilistic = True

    def __init__(self, size: str = "small"):
        import torch
        from chronos import ChronosBoltPipeline

        self.name = f"chronos-bolt-{size}"
        self.torch = torch
        t0 = time.time()
        self.pipe = ChronosBoltPipeline.from_pretrained(
            f"amazon/chronos-bolt-{size}", device_map="cpu"
        )
        self.load_seconds = time.time() - t0

    def predict(self, ctx, horizon, *, known_cov=None, past_cov=None, hods=None) -> Forecast:
        ctx = np.asarray(ctx, dtype=np.float32)
        B = ctx.shape[0]
        med = np.empty((B, horizon))
        qs = np.empty((B, horizon, len(COMMON_LEVELS)))
        t0 = time.time()
        for i in range(0, B, CHUNK):
            sl = slice(i, min(i + CHUNK, B))
            batch = [self.torch.tensor(c, dtype=self.torch.float32) for c in ctx[sl]]
            q, _mean = self.pipe.predict_quantiles(
                inputs=batch, prediction_length=horizon, quantile_levels=list(COMMON_LEVELS)
            )
            q = np.asarray(q, dtype=float)
            qs[sl] = q
            med[sl] = q[:, :, COMMON_LEVELS.index(0.5)]
        return Forecast(self.name, med, qs, COMMON_LEVELS, time.time() - t0, B,
                        {"load_seconds": self.load_seconds})


class TimesFMForecaster(BaseForecaster):
    """Google TimesFM 3.0 (PyTorch backend)."""

    probabilistic = True

    def __init__(self, repo: str = "google/timesfm-3.0-pytorch", device: str = "cpu"):
        from timesfm import TimesFM3Forecaster as _T

        self.name = "timesfm-3.0"
        t0 = time.time()
        self.model = _T.from_pretrained(repo, device=device)
        self.load_seconds = time.time() - t0
        self.levels = tuple(self.model.config.quantiles)

    def predict(self, ctx, horizon, *, known_cov=None, past_cov=None, hods=None) -> Forecast:
        ctx = np.asarray(ctx, dtype=np.float32)
        B = ctx.shape[0]
        med = np.empty((B, horizon))
        qs = np.empty((B, horizon, len(COMMON_LEVELS)))
        t0 = time.time()
        for i in range(0, B, CHUNK):
            sl = slice(i, min(i + CHUNK, B))
            outs = list(
                self.model.predict_batch([c for c in ctx[sl]], horizon=horizon, return_quantiles=True)
            )
            for j, o in enumerate(outs):
                q = np.asarray(o.quantiles, dtype=float)  # (H, Q)
                if q.ndim == 3:  # (H, 1, Q)
                    q = q[:, 0]
                qs[i + j] = interp_levels(q[None], self.levels)[0]
                med[i + j] = qs[i + j][:, COMMON_LEVELS.index(0.5)]
        return Forecast(self.name, med, qs, COMMON_LEVELS, time.time() - t0, B,
                        {"load_seconds": self.load_seconds})


class PointForecaster(BaseForecaster):
    """Wrap a classical baseline as a point-only forecaster (no quantiles)."""

    probabilistic = False

    def __init__(self, name: str, fn, period: int = 24):
        self.name = name
        self.fn = fn
        self.period = period

    def predict(self, ctx, horizon, *, known_cov=None, past_cov=None, hods=None) -> Forecast:
        ctx = np.asarray(ctx, dtype=np.float32)
        B = ctx.shape[0]
        med = np.empty((B, horizon))
        t0 = time.time()
        for i in range(B):
            hod0 = int(hods[i]) if hods is not None else 0
            try:
                med[i] = self.fn(ctx[i], horizon, hod0=hod0)
            except TypeError:
                med[i] = self.fn(ctx[i], horizon)
        return Forecast(self.name, med, None, None, time.time() - t0, B)


def build_model(spec: str) -> BaseForecaster:
    """'timestral:209M', 'chronos:small', 'timesfm', 'baseline:persistence' ..."""
    from .baselines import BASELINES

    kind, _, arg = spec.partition(":")
    if kind == "timestral":
        return TimestralForecaster(arg or "209M")
    if kind == "chronos":
        return ChronosBoltForecaster(arg or "small")
    if kind == "timesfm":
        return TimesFMForecaster()
    if kind == "baseline":
        fn = BASELINES[arg]
        return PointForecaster(arg, fn)
    raise ValueError(f"unknown model spec {spec!r}")
