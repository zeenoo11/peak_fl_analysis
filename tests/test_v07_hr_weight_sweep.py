"""Pytest for v07-A2 hr_weight sweep — naming logic and cell-name parser.

Regression surfaces:

1. ``_hr_suffix`` (defined in both v06 drivers) formats hr_weight values
   correctly — in particular, 1.0 → "1" (g format, no trailing zero) rather
   than "1.0", which would diverge from disk directory names.
2. ``_PAT_AUX01_HR`` — the regex pattern for cells that combine an aux suffix
   AND an hr suffix (e.g. ``V6-Dyn-B-FedAvg-aux0.1-hr0.5``) matches correctly.
3. The FL driver's ``_build_cell_name`` inverts to the expected (algo, lambda,
   hr_weight) triple when given an aux+hr cell name.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]

# Pattern: V6-Dyn-B-<Algo>-aux0.1-hr<hr>  (v07-A2 sweep fixes λ=0.1 by design;
# the regex enforces that constraint so a stray `-aux0.3-hr...` cell name does
# not slip through `_PAT_AUX01_HR.match`. ``lam`` is still exposed as a named
# group for symmetry with the aux-only regex elsewhere; it is always "0.1".)
_PAT_AUX01_HR = re.compile(
    r"^V6-Dyn-B-(FedAvg|FedProx|FedRep|Ditto|FedProto)"
    r"-aux(?P<lam>0\.1)"
    r"-hr(?P<hr>[0-9.]+)$"
)


def _load_driver(rel: str, module_name: str) -> ModuleType:
    drv_path = REPO_ROOT / rel
    spec = importlib.util.spec_from_file_location(module_name, drv_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1) _hr_suffix formatting
# ---------------------------------------------------------------------------


def test_hr_suffix_one_point_zero_no_trailing_zero():
    """_hr_suffix(1.0) must produce '-hr1', not '-hr1.0' (g format)."""
    drv = _load_driver(
        "experiments/v06_round_dynamics/02_fl_dynamics.py", "v07_hr_drv_fl_1"
    )
    assert drv._hr_suffix(1.0) == "-hr1"


def test_hr_suffix_half():
    drv = _load_driver(
        "experiments/v06_round_dynamics/02_fl_dynamics.py", "v07_hr_drv_fl_2"
    )
    assert drv._hr_suffix(0.5) == "-hr0.5"


def test_hr_suffix_small():
    drv = _load_driver(
        "experiments/v06_round_dynamics/02_fl_dynamics.py", "v07_hr_drv_fl_3"
    )
    assert drv._hr_suffix(0.05) == "-hr0.05"


# ---------------------------------------------------------------------------
# 2) _PAT_AUX01_HR regex
# ---------------------------------------------------------------------------


def test_pat_aux01_hr_matches_hr_cell():
    """_PAT_AUX01_HR must match 'V6-Dyn-B-FedAvg-aux0.1-hr0.5'."""
    m = _PAT_AUX01_HR.match("V6-Dyn-B-FedAvg-aux0.1-hr0.5")
    assert m is not None
    assert m.group(1) == "FedAvg"
    assert m.group("lam") == "0.1"
    assert m.group("hr") == "0.5"


# ---------------------------------------------------------------------------
# 3) _build_cell_name round-trip
# ---------------------------------------------------------------------------


def test_fl_build_cell_name_hr_roundtrip():
    """_build_cell_name('fedavg', 0.1, 1.0) → 'V6-Dyn-B-FedAvg-aux0.1-hr1'
    and the cell can be decomposed back to (algo='fedavg', lam=0.1, hr=1.0)."""
    drv = _load_driver(
        "experiments/v06_round_dynamics/02_fl_dynamics.py", "v07_hr_drv_fl_rt"
    )
    cell = drv._build_cell_name("fedavg", 0.1, 1.0)
    assert cell == "V6-Dyn-B-FedAvg-aux0.1-hr1"

    m = _PAT_AUX01_HR.match(cell)
    assert m is not None
    assert m.group(1) == "FedAvg"
    assert float(m.group("lam")) == 0.1
    assert float(m.group("hr")) == 1.0
