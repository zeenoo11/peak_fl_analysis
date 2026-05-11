"""Pytest for v06 driver cell-name resolution (variant 2).

Why a unit-test
---------------
The aux_lambda → cell-name suffix mapping decides which output directory a
run lands in. Off-by-one on the suffix would silently overwrite the default
λ_aux=0.3 results — that is exactly the regression the user is asking us
to prevent. We pin the mapping here:

    --aux_lambda 0.3   → V6-Dyn-{...}            (back-compat, no suffix)
    --aux_lambda 0     → V6-Dyn-{...}-MAEonly
    --aux_lambda 0.1   → V6-Dyn-{...}-aux0.1     (paper-friendly suffix)

The 0/0.0 distinction is exact-equality on a float — we do NOT collapse
near-zero values like 1e-9 to MAEonly (the user passes ``--aux_lambda 0``
intentionally for the ablation).

The drivers (``01_centralised.py`` / ``02_fl_dynamics.py``) start with a
digit, so importlib.util is used to load them by absolute path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_driver(file_name: str, module_name: str) -> ModuleType:
    drv_path = Path(__file__).resolve().parents[1] / "experiments" / "v06_round_dynamics" / file_name
    # The drivers do `sys.path.insert(0, src/)` themselves at import time,
    # so loading them is fine even when src is not on path yet.
    spec = importlib.util.spec_from_file_location(module_name, drv_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_fl_driver_cell_name_default_lambda_has_no_suffix():
    drv = _load_driver("02_fl_dynamics.py", "v06_drv_fl")
    assert drv._build_cell_name("fedavg",   0.3) == "V6-Dyn-B-FedAvg"
    assert drv._build_cell_name("fedprox",  0.3) == "V6-Dyn-B-FedProx"
    assert drv._build_cell_name("fedrep",   0.3) == "V6-Dyn-B-FedRep"
    assert drv._build_cell_name("ditto",    0.3) == "V6-Dyn-B-Ditto"
    assert drv._build_cell_name("fedproto", 0.3) == "V6-Dyn-B-FedProto"


def test_fl_driver_cell_name_zero_lambda_gets_maeonly_suffix():
    drv = _load_driver("02_fl_dynamics.py", "v06_drv_fl")
    assert drv._build_cell_name("fedavg",   0.0) == "V6-Dyn-B-FedAvg-MAEonly"
    assert drv._build_cell_name("fedavg",   0)   == "V6-Dyn-B-FedAvg-MAEonly"
    assert drv._build_cell_name("fedproto", 0.0) == "V6-Dyn-B-FedProto-MAEonly"


def test_fl_driver_cell_name_other_lambda_gets_generic_suffix():
    drv = _load_driver("02_fl_dynamics.py", "v06_drv_fl")
    assert drv._build_cell_name("fedavg", 0.1) == "V6-Dyn-B-FedAvg-aux0.1"
    assert drv._build_cell_name("ditto",  0.5) == "V6-Dyn-B-Ditto-aux0.5"


def test_centralised_driver_cell_name_default_lambda_unchanged():
    drv = _load_driver("01_centralised.py", "v06_drv_cent")
    # default aux_lambda = 0.3 — must stay at the original directory name.
    assert drv._build_cell_name(0.3) == "V6-Dyn-A_centralised"


def test_centralised_driver_cell_name_zero_lambda_gets_maeonly():
    drv = _load_driver("01_centralised.py", "v06_drv_cent")
    assert drv._build_cell_name(0.0) == "V6-Dyn-A_centralised-MAEonly"
    assert drv._build_cell_name(0)   == "V6-Dyn-A_centralised-MAEonly"


def test_centralised_driver_cell_name_other_lambda_generic():
    drv = _load_driver("01_centralised.py", "v06_drv_cent")
    assert drv._build_cell_name(0.1) == "V6-Dyn-A_centralised-aux0.1"
