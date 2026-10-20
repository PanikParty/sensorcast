"""Charts for the report.  Headless matplotlib only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.lines  # noqa: E402
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
    }
)

FAMILY_COLORS = {
    "timestral": "#c2410c",
    "timesfm": "#1d4ed8",
    "chronos": "#047857",
    "baseline": "#6b7280",
}


def _color(model: str) -> str:
    for k, v in FAMILY_COLORS.items():
        if model.startswith(k):
            return v
    return "#111827"


def accuracy_bars(agg: pd.DataFrame, out: Path, metric: str = "mase") -> Path:
    horizons = sorted(agg["horizon"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(5.0 * len(horizons), 3.6), squeeze=False)
    for ax, h in zip(axes[0], horizons):
        # best first in the frame, then invert the axis so rank 1 sits at the top
        d = agg[(agg["horizon"] == h) & agg[metric].notna()].sort_values(metric, ascending=True)
        labels = [f"{i + 1}. {m}" for i, m in enumerate(d["model"])]
        ax.barh(labels, d[metric], color=[_color(m) for m in d["model"]])
        ax.invert_yaxis()
        ax.axvline(1.0, color="#111827", lw=1.2, ls="--", zorder=3)
        ax.set_title(f"{metric.upper()} @ {h}h horizon")
        ax.set_xlabel(metric.upper() + "  (lower is better; 1.0 = seasonal naive)")
        for y, v in enumerate(d[metric]):
            ax.text(v, y, f" {v:.3f}", va="center", fontsize=7.5)
        ax.margins(x=0.18)
    fig.tight_layout()
    p = out / f"fig_{metric}_by_model.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# Lead-time curves overlap, so family colour is not enough: every foundation model gets its
# own hue (a colourblind-validated categorical order, fixed per model so a figure with fewer
# models never repaints the rest) plus its own marker; baselines recede as grey, told apart
# by dash pattern.
LEADTIME_STYLE = {
    "timesfm-3.0": dict(color="#2a78d6", marker="o"),
    "timestral-209M": dict(color="#eb6834", marker="s"),
    "chronos-bolt-small": dict(color="#1baf7a", marker="^"),
    "timestral-123M": dict(color="#eda100", marker="D"),
    "timestral-28M": dict(color="#e87ba4", marker="v"),
    "timestral-5M": dict(color="#008300", marker="P"),
    "persistence": dict(color="#898781", ls=(0, (6, 2))),
    "seasonal_naive": dict(color="#52514e", ls=(0, (1, 1.5))),
    "climatology_hod": dict(color="#898781", ls=(0, (6, 2, 1, 2))),
    "drift": dict(color="#52514e", ls=(0, (3, 1.5))),
}


def leadtime_curves(err: dict[str, np.ndarray], out: Path, horizon: int) -> Path:
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    order = list(LEADTIME_STYLE) + [m for m in err if m not in LEADTIME_STYLE]
    for model in order:
        arr = err.get(model)
        if arr is None or arr.shape[1] != horizon:
            continue
        rmse = np.sqrt(np.nanmean(arr**2, axis=0))
        style = LEADTIME_STYLE.get(model, dict(color=_color(model)))
        fm = "marker" in style
        ax.plot(np.arange(1, horizon + 1), rmse, label=model, lw=1.7 if fm else 1.3,
                markersize=4.5, markevery=max(1, horizon // 8), zorder=3 if fm else 2, **style)
    ax.set_xlabel("lead time (hours ahead)")
    ax.set_ylabel("RMSE (deg C)")
    ax.set_title(f"Error growth over the horizon ({horizon} h)")
    # outside the axes: the curves span the full height, so any inside spot covers one
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    p = out / f"fig_leadtime_{horizon}h.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def example_forecast(
    wide: pd.DataFrame,
    device: str,
    origin: pd.Timestamp,
    series: dict[str, np.ndarray],
    truth: np.ndarray,
    band: tuple[np.ndarray, np.ndarray] | None,
    out: Path,
    context_len: int = 504,
    history_hours: int = 48,
    baseline: np.ndarray | None = None,
) -> Path:
    """One window, zoomed to the last 48 h of context plus the held-out horizon.

    Only the best Timestral (with its 10-90% band) and two competitors are drawn:
    six nearly identical lines on one panel is decoration, not information.
    """
    i = wide.index.get_loc(origin)
    hist = wide[device].to_numpy(dtype=float)[i - context_len : i]
    ht = wide.index[i - context_len : i]
    ft = pd.date_range(origin, periods=len(truth), freq="1h")

    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.plot(ht[-history_hours:], hist[-history_hours:], color="#111827", lw=1.6, label="observed")
    ax.plot(ft, truth, color="#111827", lw=1.5, ls=":", marker=".", ms=3, label="truth (held out)")
    if band is not None:
        lo, hi = band
        ax.fill_between(ft, lo, hi, color="#c2410c", alpha=0.16, label="timestral 10-90%")
    for name, fc in series.items():
        ax.plot(ft, fc, color=_color(name), lw=1.7, marker="o", ms=2.6, label=name)
    if baseline is not None:
        ax.plot(ft, baseline, color="#6b7280", lw=1.2, ls="--", label="seasonal naive")
    ax.set_title(f"{device} @ {origin:%Y-%m-%d %H:%M} local  —  {len(truth)} h ahead")
    ax.set_ylabel("deg C")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper left")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %H:%M"))
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    p = out / f"fig_example_{device}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def device_heatmap(per_window: pd.DataFrame, out: Path, horizon: int, metric: str = "mase") -> Path:
    d = per_window[(per_window["horizon"] == horizon) & per_window[metric].notna()]
    if d.empty:
        return out / "fig_device_heatmap.png"
    piv = d.pivot_table(index="device", columns="model", values=metric, aggfunc="mean")
    piv = piv.dropna(axis=1, how="all")
    fig, ax = plt.subplots(figsize=(0.62 * len(piv.columns) + 3.2, 0.4 * len(piv) + 1.8))
    im = ax.imshow(piv.to_numpy(), cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(piv.index)), piv.index, fontsize=7)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.2)
    ax.set_title(f"{metric.upper()} by sensor @ {horizon}h")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    p = out / f"fig_device_heatmap_{horizon}h.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def forecast_vs_actual(
    wide: pd.DataFrame,
    forecasts,
    out: Path,
    horizon: int,
    per_window: pd.DataFrame | None = None,
    models: tuple[str, ...] = ("timestral-209M", "timesfm-3.0"),
) -> Path:
    """Every sensor, the whole evaluation span: observed temperature vs the stored forecasts.

    ``forecasts`` is the ``results/forecasts.npz`` archive.  Origins are 12 h apart, so a
    24 h horizon would overlap itself; only the origins on the newest origin's phase
    (newest, then every ``horizon`` hours back) are drawn, which tiles the span end to end.
    That is a fixed rule, not a pick of good windows -- the panel labels quote the MASE
    over *all* windows.  One shared y-axis, so sensors are comparable at a glance.
    """
    pre = f"h{horizon}_"
    dev = forecasts[pre + "device"].astype(str)
    org = pd.to_datetime(forecasts[pre + "origin"], unit="s", utc=True).tz_convert(wide.index.tz)
    levels = [tuple(np.round(forecasts[f"{pre}{m}_levels"], 2)) for m in models[:1]]
    lo_i, hi_i = levels[0].index(0.1), levels[0].index(0.9)
    devices = sorted(set(dev))
    span = (org.min() - pd.Timedelta(hours=24), org.max() + pd.Timedelta(hours=horizon))

    fig, axes = plt.subplots(len(devices), 1, figsize=(10.5, 1.45 * len(devices) + 1.0),
                             sharex=True, sharey=True, squeeze=False)
    ink, grid = "#0b0b0b", "#c3c2b7"
    for ax, d in zip(axes[:, 0], devices):
        idx = np.flatnonzero(dev == d)
        newest = org[idx].max()
        keep = [i for i in idx if ((newest - org[i]) / pd.Timedelta(hours=1)) % horizon == 0]
        for i in keep:
            ft = pd.date_range(org[i], periods=horizon, freq="1h")
            ax.axvline(org[i], color=grid, lw=0.6, zorder=1)
            q = forecasts[f"{pre}{models[0]}_quantiles"][i]
            ax.fill_between(ft, q[:, lo_i], q[:, hi_i], color=LEADTIME_STYLE[models[0]]["color"],
                            alpha=0.18, lw=0, zorder=2)
            for m in models:
                ax.plot(ft, forecasts[f"{pre}{m}_median"][i], color=LEADTIME_STYLE[m]["color"],
                        lw=1.3, zorder=3)
        obs = wide.loc[span[0] : span[1], d]
        ax.plot(obs.index, obs.to_numpy(), color=ink, lw=1.0, marker=".", ms=2.2, zorder=4)
        label = d
        if per_window is not None:
            s = per_window[(per_window["horizon"] == horizon) & (per_window["device"] == d)]
            ms = [f"{m} {s[s['model'] == m]['mase'].mean():.2f}" for m in models]
            label += "   MASE " + " · ".join(ms)
        ax.text(0.005, 0.96, label, transform=ax.transAxes, va="top", fontsize=7.5, color=ink,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
        ax.tick_params(labelsize=7)
    axes[-1, 0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
    axes[-1, 0].set_xlim(*span)
    fig.supylabel("deg C", fontsize=8)

    handles = [matplotlib.lines.Line2D([], [], color=ink, lw=1.0, marker=".", ms=4, label="observed")]
    handles += [matplotlib.lines.Line2D([], [], color=LEADTIME_STYLE[m]["color"], lw=1.6, label=f"{m} median")
                for m in models]
    handles += [matplotlib.patches.Patch(color=LEADTIME_STYLE[models[0]]["color"], alpha=0.18,
                                         label=f"{models[0]} 10-90%"),
                matplotlib.lines.Line2D([], [], color=grid, lw=0.8, label="forecast issued")]
    fig.suptitle(f"Forecast vs actual, every sensor — {horizon} h forecasts issued every {horizon} h "
                 f"at {newest:%H:%M}, gaps = sensor offline", fontsize=10, fontweight="bold")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=len(handles),
               frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = out / f"fig_forecast_vs_actual_{horizon}h.png"
    fig.savefig(p)
    plt.close(fig)
    return p
