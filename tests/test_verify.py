"""verify_scores.check_data must actually compare cells.  Run: pytest -q"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from sensorcast import data

_spec = importlib.util.spec_from_file_location(
    "verify_scores", Path(__file__).resolve().parents[1] / "scripts" / "verify_scores.py"
)
verify_scores = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_scores)


def _write(tmp_path: Path, hourly: pd.DataFrame) -> Path:
    idx = pd.date_range("2026-10-01", periods=3, freq="1h", tz=data.LOCAL_TZ)
    ts = [int(t.timestamp()) + 600 for t in idx]
    raw = pd.DataFrame({"sensor": ["Shed North"] * 3 + ["Kitchen"] * 2,
                        "ts": ts + ts[:2], "t_c": [10.0, 11.0, 12.0, 20.0, 21.0]})
    (tmp_path / "data").mkdir()
    raw.to_csv(tmp_path / "data" / "shelly_raw_samples.csv", index=False)
    hourly.index = idx
    data.save_snapshot(hourly, tmp_path / "data" / "shelly_hourly.csv")
    return tmp_path


def test_check_data_passes_on_matching_grid(tmp_path):
    good = pd.DataFrame({"kitchen": [20.0, 21.0, np.nan], "shed_north": [10.0, 11.0, 12.0]})
    msgs = verify_scores.check_data(_write(tmp_path, good))
    assert msgs[0].startswith("DATA      PASS"), msgs


def test_check_data_fails_on_wrong_value(tmp_path):
    bad = pd.DataFrame({"kitchen": [20.0, 25.0, np.nan], "shed_north": [10.0, 11.0, 12.0]})
    assert "FAIL" in verify_scores.check_data(_write(tmp_path, bad))[0]


def test_check_data_fails_on_cell_missing_from_raw(tmp_path):
    # snapshot has a reading the raw export cannot produce
    bad = pd.DataFrame({"kitchen": [20.0, 21.0, 22.0], "shed_north": [10.0, 11.0, 12.0]})
    assert "FAIL" in verify_scores.check_data(_write(tmp_path, bad))[0]
