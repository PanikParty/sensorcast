"""Smoke-test each model adapter with a small batch; prints timings + shapes."""
import time
import numpy as np
import torch

N_SERIES, T, H = 9, 504, 24
rng = np.random.default_rng(0)
ctx = (rng.standard_normal((N_SERIES, T)).cumsum(-1) * 0.05 + 26.0).astype(np.float32)
ctx[:, 100:130] = np.nan  # holes, like the real sensors

print("=== Chronos-Bolt (small) ===")
from chronos import ChronosBoltPipeline
t0 = time.time()
pipe = ChronosBoltPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu")
print(f"load {time.time()-t0:.1f}s")
t0 = time.time()
q, mean = pipe.predict_quantiles(
    inputs=[torch.tensor(c, dtype=torch.float32) for c in ctx],
    prediction_length=H,
    quantile_levels=[0.1, 0.5, 0.9],
)
print(f"infer 9 series: {time.time()-t0:.2f}s  q={tuple(q.shape)} mean={tuple(mean.shape)}")

print("=== TimesFM 3.0 (pytorch) ===")
from timesfm import TimesFM3Forecaster
t0 = time.time()
tf = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device="cpu")
print(f"load {time.time()-t0:.1f}s")
t0 = time.time()
outs = list(tf.predict_batch([c.astype(np.float32) for c in ctx], horizon=H, return_quantiles=True))
print(f"infer 9 series: {time.time()-t0:.2f}s n_out={len(outs)}")
o = outs[0]
print("attrs:", [a for a in dir(o) if not a.startswith("_")])
for a in ("mean", "point_forecast", "quantiles", "quantile_levels"):
    if hasattr(o, a):
        v = getattr(o, a)
        print(f"  {a}:", np.shape(v) if v is not None else None)
print("=== Timestral 209M ===")
from timestral import TimestralForecaster
t0 = time.time()
tsl = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cpu")
print(f"load {time.time()-t0:.1f}s")
t0 = time.time()
f = tsl.forecast(ctx.reshape(N_SERIES, 1, T), prediction_length=H)
print(f"infer 9 series: {time.time()-t0:.2f}s median={tuple(f.median.shape)}")
