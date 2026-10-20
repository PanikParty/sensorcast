"""Record exactly what produced a result: interpreter, library versions, model code rev."""

from __future__ import annotations

import importlib
import platform
import subprocess
import sys

PACKAGES = ["numpy", "pandas", "matplotlib", "torch", "timestral", "chronos", "timesfm", "huggingface_hub"]


def _version(mod: str) -> str:
    try:
        importlib.import_module(mod)
    except Exception:
        return "absent"
    try:
        from importlib.metadata import version

        return version(mod.split(".")[0])
    except Exception:
        pass
    m = importlib.import_module(mod)
    for attr in ("__version__", "version"):
        v = getattr(m, attr, None)
        if isinstance(v, str):
            return v
    return "unknown"


def _git_rev(pkg: str) -> str:
    """Commit hash of a pip-installed package that came from a git URL.

    ``pip show`` reports only the declared version for a git install, so read the
    PEP 610 ``direct_url.json`` that pip writes for VCS installs.
    """
    import json
    from importlib.metadata import distribution

    try:
        dist = distribution(pkg)
        raw = dist.read_text("direct_url.json")
    except Exception:
        return "unknown"
    if not raw:
        return "release"
    try:
        info = json.loads(raw)
    except Exception:
        return "unknown"
    vcs = info.get("vcs_info") or {}
    commit = vcs.get("commit_id")
    url = info.get("url", "")
    if commit:
        return f"{commit[:12]} ({url})"
    return url or "release"


def versions() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    env.update({p: _version(p) for p in PACKAGES})
    env["timestral_git_rev"] = _git_rev("timestral")
    env["torch_threads"] = None
    try:
        import torch

        env["torch_threads"] = int(torch.get_num_threads())
    except Exception:
        pass
    return env
