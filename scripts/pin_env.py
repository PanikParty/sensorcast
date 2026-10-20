#!/usr/bin/env python
"""Freeze the environment: requirements.lock.txt + env block in run_manifest.json.

A benchmark that does not pin its model code is not reproducible -- `timestral` is
installed straight from GitHub, so the commit hash is part of the result.

    python scripts/pin_env.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from sensorcast.provenance import versions


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    env = versions()

    # full freeze, including the git URL + commit for timestral.
    # uv venvs have no pip, so try uv first and fall back to pip.
    freeze = ""
    for cmd in (["uv", "pip", "freeze"], [sys.executable, "-m", "pip", "freeze", "--all"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except FileNotFoundError:
            continue
        if out.returncode == 0 and out.stdout.strip():
            freeze = out.stdout.strip()
            break
    if not freeze:
        raise SystemExit("could not freeze the environment (no uv or pip on PATH)")
    header = (
        "# Frozen environment for the run recorded in results/run_manifest.json\n"
        f"# python {env['python']} on {env['platform']}\n"
        "# regenerate with: python scripts/pin_env.py\n"
    )
    lock = root / "requirements.lock.txt"
    lock.write_text(header + freeze + "\n")

    # Hugging Face resolves checkpoints to a snapshot commit; record them so the
    # exact weights can be pinned with revision=<commit> later.
    weights = {}
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    for d in sorted(hub.glob("models--mistralai--Timestral-*")) + sorted(hub.glob("models--google--timesfm-*")) + sorted(hub.glob("models--amazon--chronos-bolt-*")):
        snaps = sorted(p.name for p in (d / "snapshots").glob("*")) if (d / "snapshots").exists() else []
        if snaps:
            weights[d.name.replace("models--", "").replace("--", "/")] = snaps[-1]
    env["hf_snapshot_commits"] = weights

    man_p = root / "results" / "run_manifest.json"
    if man_p.exists():
        man = json.loads(man_p.read_text())
        man["env"] = env
        man_p.write_text(json.dumps(man, indent=2, default=str))

    print(f"wrote {lock} ({len(freeze.splitlines())} pinned packages)")
    for k, v in env.items():
        print(f"  {k:18s} {v}")


if __name__ == "__main__":
    main()
