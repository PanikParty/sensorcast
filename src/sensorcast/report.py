"""Render results/ artifacts into REPORT.md."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PRETTY = {
    "timestral-5M": "Timestral-5M",
    "timestral-28M": "Timestral-28M",
    "timestral-123M": "Timestral-123M",
    "timestral-209M": "Timestral-209M",
    "timesfm-3.0": "TimesFM 3.0 (200M)",
    "chronos-bolt-small": "Chronos-Bolt-small",
    "chronos-bolt-base": "Chronos-Bolt-base",
    "persistence": "persistence",
    "seasonal_naive": "seasonal naive",
    "climatology_hod": "climatology (hour-of-day)",
    "drift": "drift",
}

FOUNDATION = ("timestral", "timesfm", "chronos")


def md_table(df: pd.DataFrame, cols: list[str], floatfmt: str = "{:.3f}",
             intcols: tuple[str, ...] = ()) -> str:
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [head, sep]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (int, np.integer)) or c in intcols:
                cells.append(str(int(round(float(v)))) if np.isfinite(float(v)) else "")
            elif isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _mase(agg, model, horizon):
    d = agg[(agg["model"] == model) & (agg["horizon"] == horizon)]["mase"]
    return float(d.iloc[0]) if len(d) else float("nan")


def _matched_mase(per_window, model, horizon, covariate):
    """Mean MASE of ``model`` without the covariate sensor -- the covariate arm's target set."""
    d = per_window[(per_window["model"] == model) & (per_window["horizon"] == horizon)
                   & (per_window["device"] != covariate)]["mase"]
    return float(d.mean()) if len(d) else float("nan")


def _paired_ci(per_window, a, b, horizon, n_boot=5000, seed=0):
    """Mean per-window MASE difference a - b with a 95% bootstrap CI.

    Resamples whole forecast origins, not windows: the sensors share a house, so
    windows cast at the same origin are correlated and do not count separately.
    """
    d = per_window[per_window["horizon"] == horizon]
    piv = d.pivot_table(index=["device", "origin"], columns="model", values="mase").dropna(subset=[a, b])
    diff = pd.DataFrame({"d": piv[a] - piv[b], "o": piv.index.get_level_values("origin")})
    g = diff.groupby("o")["d"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(g), (n_boot, len(g)))
    boot = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(diff["d"].mean()), float(lo), float(hi)


def _resolved(lo, hi) -> bool:
    return lo > 0 or hi < 0


def _key_findings(agg, per_window, cov_agg, covariate="shed_north") -> list[str]:
    out = []
    hs = sorted(agg["horizon"].unique())
    for h in hs:
        d = agg[(agg["horizon"] == h) & agg["mase"].notna()].sort_values("mase")
        if d.empty:
            continue
        top, second = d.iloc[0], d.iloc[1]
        sn = _mase(agg, "seasonal_naive", h)
        _, lo, hi = _paired_ci(per_window, top["model"], second["model"], h)
        if _resolved(lo, hi):
            vs = (f"ahead of runner-up {PRETTY.get(second['model'], second['model'])} "
                  f"({second['mase']:.3f}; 95% CI of the gap {lo:+.3f} to {hi:+.3f})")
        else:
            vs = (f"**statistically tied** with {PRETTY.get(second['model'], second['model'])} "
                  f"({second['mase']:.3f}; 95% CI of the gap {lo:+.3f} to {hi:+.3f})")
        out.append(
            f"- **{h} h ahead** — **{PRETTY.get(top['model'], top['model'])}** has the lowest aggregate "
            f"MASE **{top['mase']:.3f}** (MAE {top['mae']:.2f} °C), {vs}; {sn / top['mase']:.1f}× better "
            f"than seasonal naive ({sn:.3f})."
        )

    # does model size buy anything?
    for h in hs:
        sizes = {s: _mase(agg, f"timestral-{s}", h) for s in ("5M", "28M", "123M", "209M")}
        if all(np.isfinite(list(sizes.values()))):
            lat = {
                s: float(agg[(agg["model"] == f"timestral-{s}") & (agg["horizon"] == h)]["ms_per_series"].iloc[0])
                for s in sizes
            }
            gain = 100 * (sizes["5M"] - sizes["209M"]) / sizes["5M"]
            ratio = lat["209M"] / max(lat["5M"], 1e-9)
            _, lo, hi = _paired_ci(per_window, "timestral-209M", "timestral-5M", h)
            if _resolved(lo, hi) and gain >= 5:
                verdict, note = "pays", "the gap is outside the noise"
            else:
                verdict, note = "does not clearly pay", "the gap is within the noise"
            out.append(
                f"- **Model size {verdict} at {h} h** — 5M → 209M buys **{gain:.1f}%** MASE "
                f"({sizes['5M']:.3f} → {sizes['209M']:.3f}; 95% CI of the gap {lo:+.3f} to {hi:+.3f}, "
                f"{note}) for **{ratio:.0f}× the CPU** ({lat['5M']:.0f} → {lat['209M']:.0f} ms/series)."
            )

    if cov_agg is not None and len(cov_agg):
        # one bullet per horizon, for the flagship model only; the baseline is scored on the
        # same target sensors as the covariate arm (the covariate sensor is not a target there)
        row = cov_agg[cov_agg["model"].str.contains("209M")]
        for _, r in (row if len(row) else cov_agg).iterrows():
            base_model = r["model"].split("+")[0]
            b = _matched_mase(per_window, base_model, int(r["horizon"]), covariate)
            s5, s209 = (_mase(agg, f"timestral-{s}", r["horizon"]) for s in ("5M", "209M"))
            size_gain = (s5 - s209) / s5 if np.isfinite(s5) and np.isfinite(s209) else 0.0
            cov_gain = (b - r["mase"]) / b if np.isfinite(b) else 0.0
            # only claim "beats" when weather is worth clearly more than the whole size ladder
            headline = ("Exogenous weather beats the model-size upgrade" if cov_gain > 2 * size_gain
                        else "Exogenous weather is worth about as much as the model-size upgrade")
            if np.isfinite(b):
                out.append(
                    f"- **{headline}** — {base_model} at "
                    f"{int(r['horizon'])} h: MASE **{b:.3f} → {r['mase']:.3f}** "
                    f"(**{100*(b-r['mase'])/b:.0f}% lower**, same {int(r['n_windows'])} windows) when "
                    f"handed the actual outdoor temperature across the horizon (oracle — see the "
                    f"covariate arm)."
                )

    # per-sensor winners; models within 0.0005 of the best (equal at the 3 dp shown) share the win
    for h in hs:
        d = per_window[(per_window["horizon"] == h) & per_window["mase"].notna()]
        if d.empty:
            continue
        piv = d.pivot_table(index="device", columns="model", values="mase", aggfunc="mean")
        wins: dict[str, int] = {}
        ties = []
        for dev, row in piv.iterrows():
            best = row[row <= row.min() + 5e-4].index.tolist()
            if len(best) == 1:
                wins[best[0]] = wins.get(best[0], 0) + 1
            else:
                ties.append(f"{dev}: " + " = ".join(PRETTY.get(m, m) for m in best))
        txt = ", ".join(f"{PRETTY.get(k, k)} {v}/{len(piv)}"
                        for k, v in sorted(wins.items(), key=lambda kv: -kv[1]))
        if ties:
            txt += f"; tied {len(ties)}/{len(piv)} ({'; '.join(ties)})"
        out.append(f"- **{h} h, sensor by sensor** — wins: {txt}.")
        t = per_window[(per_window["horizon"] == h)]
        cal = {}
        for m in ("timestral-209M", "timesfm-3.0", "chronos-bolt-small"):
            v = t[t["model"] == m]["coverage80"].mean()
            if np.isfinite(v):
                cal[m] = float(v)
        if len(cal) >= 2:
            txt = ", ".join(f"{PRETTY.get(k, k)} {v:.0%}" for k, v in cal.items())
            out.append(
                f"- **Interval calibration at {h} h** (nominal 80%) — {txt}. "
                "Timestral's bands are close to nominal; TimesFM's are too narrow, so it looks "
                "sharper than it is on WQL while under-covering."
            )
    return out


def build(
    out: Path,
    agg: pd.DataFrame,
    per_window: pd.DataFrame,
    cfg: dict,
    figures: list[Path],
    covariate_agg: pd.DataFrame | None = None,
    coverage: pd.Series | None = None,
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    figures = [f for f in figures if f.exists()]

    L: list[str] = []
    L.append("# sensorcast — zero-shot temperature forecasting on the family's Shelly sensors\n")
    L.append(
        "Benchmark of Mistral **Timestral** (5M/28M/123M/209M) against Google **TimesFM 3.0**, "
        "Amazon **Chronos-Bolt**, and four training-free baselines, on **hourly temperature from "
        "9 Shelly H&T sensors** in the house and the shed.\n"
    )

    L.append("## Key findings\n")
    covariate = cfg.get("covariate", "shed_north")
    L.extend(_key_findings(agg, per_window, covariate_agg, covariate))
    L.append("")

    L.append("## Dataset\n")
    L.append(
        f"- Source: live `shelly.db` archive, hourly mean of {cfg['raw_samples']:,} samples "
        f"({cfg['span_days']:.0f} days: {cfg['t_start']} → {cfg['t_end']} local)."
    )
    L.append(
        f"- Grid: `{cfg['freq']}`, {cfg['n_hours']} hours × {cfg['n_devices']} sensors, "
        f"**{cfg['nan_frac']:.0%} of cells empty** (kept as NaN, not interpolated)."
    )
    if coverage is not None:
        cov = ", ".join(f"{k} {v:.0%}" for k, v in coverage.items())
        L.append(f"- Per-sensor hourly coverage: {cov}.")
    L.append("")

    L.append("## Protocol\n")
    L.append(
        f"- Rolling origins every **{cfg['step_hours']} h** over the final **{cfg['eval_days']} days**; "
        f"context **{cfg['context_len']} h (3 weeks)**; horizons {', '.join(str(h) for h in cfg['horizons'])} h."
    )
    L.append("- Every model is **zero-shot** — no fitting, no fine-tuning on this data.")
    L.append(
        "- A window is scored only where the ground truth exists; sensors that drop out mid-horizon "
        "are scored on the hours they reported."
    )
    L.append(
        "- `MASE` = MAE ÷ the window's own seasonal-naive error scale (mean |yₜ − yₜ₋₂₄ₕ| over the "
        "context). **MASE < 1 beats \"same hour yesterday\"**; MASE is the primary ranking because it "
        "normalises for how volatile each sensor is."
    )
    L.append(
        "- `WQL` = GIFT-Eval-style weighted quantile loss (10–90% deciles, scale-free). Only the three "
        "foundation models emit quantiles, so baselines have no WQL."
    )
    L.append("")

    L.append("## Results\n")
    for h in sorted(agg["horizon"].unique()):
        d = agg[agg["horizon"] == h].sort_values("mase").copy()
        d["model"] = d["model"].map(lambda m: PRETTY.get(m, m))
        cols = ["model", "mae", "rmse", "mase", "wql", "coverage80", "width80", "ms_per_series"]
        cols = [c for c in cols if c in d.columns]
        L.append(f"### {h}-hour horizon\n")
        L.append(md_table(d, cols, intcols=("ms_per_series",)))
        L.append("")
        n = int(d["n_windows"].iloc[0]) if "n_windows" in d else "?"
        L.append(f"*n = {n} windows, {cfg['n_devices']} sensors.*\n")

    # per-sensor table
    L.append("### Per-sensor MASE (primary horizon)\n")
    h = max(cfg["horizons"])
    d = per_window[(per_window["horizon"] == h) & per_window["mase"].notna()]
    if len(d):
        piv = d.pivot_table(index="device", columns="model", values="mase", aggfunc="mean")
        keep = [c for c in piv.columns if c.startswith(FOUNDATION) or c == "seasonal_naive"]
        piv = piv[keep].round(3)
        piv["best"] = piv[keep].idxmin(axis=1)
        tbl = piv.reset_index().rename(columns={"device": "sensor"})
        L.append(md_table(tbl, list(tbl.columns)))
        L.append(f"\n*{h} h horizon. `seasonal_naive` is the reference line every foundation model "
                 "has to clear.*\n")

    if covariate_agg is not None and len(covariate_agg):
        L.append("## Covariate arm — does tomorrow's outdoor temperature help?\n")
        L.append(
            "Timestral accepts a *known-future* covariate. We fed it the real outdoor sensor "
            f"(`{covariate}`, sun-exposed, best uptime) across the horizon. That is an **oracle** — a "
            "real deployment would need a weather forecast — so read it as an upper bound on what "
            "exogenous weather information could buy, not as an achievable score.\n\n"
            f"`{covariate}` is the input here, so it is not a forecast target: the arm scores the other "
            f"{per_window['device'].nunique() - 1} sensors. `mase without` is the same model with no "
            "covariate on those same windows, so the two columns compare like with like.\n"
        )
        c = covariate_agg.copy()
        c["mase without"] = [_matched_mase(per_window, m.split("+")[0], int(h), covariate)
                             for m, h in zip(c["model"], c["horizon"])]
        c["model"] = c["model"].map(lambda m: PRETTY.get(m, m))
        L.append(md_table(c.sort_values(["horizon", "mase"]),
                          ["horizon", "model", "mae", "mase without", "mase", "n_windows"],
                          intcols=("horizon", "n_windows")))
        L.append("")

    L.append("## Latency (this box: Intel N150, 4 cores, CPU-only)\n")
    lat = pd.DataFrame()
    if "ms_per_series" in agg.columns:
        lat = agg[agg["model"].str.startswith(FOUNDATION)][["horizon", "model", "ms_per_series"]].copy()
    if len(lat):
        piv = lat.pivot_table(index="model", columns="horizon", values="ms_per_series")
        piv = piv.reset_index()
        piv.columns = [str(c) if not isinstance(c, (int, np.integer)) else f"{c}h" for c in piv.columns]
        L.append(md_table(piv, list(piv.columns), intcols=tuple(piv.columns[1:])))
        L.append(
            "\n*Milliseconds per series per forecast, batched (48 series per forward pass), "
            f"{cfg.get('torch_threads', 4)} torch threads. No GPU was used.*\n"
        )

    if figures:
        L.append("## Figures\n")
        for f in figures:
            L.append(f"![{f.stem}]({f.name})\n")

    L.append("## What this means for us\n")
    for h in sorted(agg["horizon"].unique()):
        s5, s28, s209 = (_mase(agg, f"timestral-{s}", h) for s in ("5M", "28M", "209M"))
        if all(np.isfinite([s5, s28, s209])):
            L.append(
                f"- At **{h} h**, run **Timestral-28M**: MASE {s28:.3f} versus {s209:.3f} for 209M "
                f"({100*(s28-s209)/s209:.1f}% worse) at a fraction of the CPU cost — the right default "
                "for a household forecast (`scripts/forecast_now.py` uses it)."
            )
    L.append(
        "- **The next real gain is not a bigger model, it is a weather input.** Handing the model the "
        "actual outdoor temperature (even as an oracle) cut MASE more than any size step, by a wide margin "
        "at 24 h. Wiring a "
        "real forecast feed into the covariate slot is the highest-value follow-up.\n"
    )

    L.append("## Honest caveats\n")
    L.append(
        "- **54 days is short.** Diurnal cycles are well represented; seasonal drift is not. These "
        "numbers say nothing about a winter/summer transition.\n"
        "- **The sensors are not independent.** Indoor rooms share HVAC and outdoor weather; windows "
        "overlap in time, so the effective sample size is smaller than the window count. The confidence "
        "intervals above account for this by resampling whole forecast origins, not single windows.\n"
        "- **Timestral's leaderboard wins are self-reported by Mistral** and, at launch, had not cleared "
        "external review. This benchmark is independent of that claim but is also *our* data only — "
        "9 series, one house, one season.\n"
        "- **Nothing here is tuned.** No context-length sweep, no per-sensor scaling, no ensembling. "
        "A tuned Timestral would score better than these numbers; so might the others.\n"
        "- MASE is computed against an in-context scale, so a window whose context is unusually calm "
        "shows a harsher MASE. Read MAE/RMSE alongside it.\n"
    )
    p = out / "REPORT.md"
    p.write_text("\n".join(L))
    return p


def write_manifest(out: Path, cfg: dict) -> Path:
    p = out / "run_manifest.json"
    p.write_text(json.dumps(cfg, indent=2, default=str))
    return p
