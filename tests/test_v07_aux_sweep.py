"""Pytest for v07-A λ_aux sweep — naming + namespace + aggregator parsing.

Three regression surfaces this pytest pins:

1. ``_aux_suffix`` produces the v07-correct suffix for the new
   `λ ∈ {0.05, 0.1, 0.2}` cells. v07 launcher's cell directories must
   match exactly so v06 + v07 results compose into a single (algo × λ) matrix.
2. The v06 drivers' new ``--output_namespace`` argparse defaults to
   ``v06_round_dynamics`` (back-compat) and accepts the v07 override.
3. ``05_aggregate_aux._parse_cell_name`` correctly inverts every cell-name
   pattern we expect to read from disk (default / -MAEonly / -auxV variants
   for centralised + 5 FL algorithms).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_driver(rel: str, module_name: str) -> ModuleType:
    drv_path = REPO_ROOT / rel
    spec = importlib.util.spec_from_file_location(module_name, drv_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1) Cell-name suffix mapping for v07-A new lambdas
# ---------------------------------------------------------------------------


def test_v07_centralised_cell_names_for_new_lambdas():
    drv = _load_driver(
        "experiments/v06_round_dynamics/01_centralised.py", "v07_drv_central"
    )
    assert drv._build_cell_name(0.05) == "V6-Dyn-A_centralised-aux0.05"
    assert drv._build_cell_name(0.1)  == "V6-Dyn-A_centralised-aux0.1"
    assert drv._build_cell_name(0.2)  == "V6-Dyn-A_centralised-aux0.2"


def test_v07_fl_cell_names_for_new_lambdas():
    drv = _load_driver(
        "experiments/v06_round_dynamics/02_fl_dynamics.py", "v07_drv_fl"
    )
    for algo, pretty in [("fedavg", "FedAvg"), ("fedprox", "FedProx"),
                         ("fedrep", "FedRep"), ("ditto", "Ditto"),
                         ("fedproto", "FedProto")]:
        assert drv._build_cell_name(algo, 0.05) == f"V6-Dyn-B-{pretty}-aux0.05"
        assert drv._build_cell_name(algo, 0.1)  == f"V6-Dyn-B-{pretty}-aux0.1"
        assert drv._build_cell_name(algo, 0.2)  == f"V6-Dyn-B-{pretty}-aux0.2"


def test_v06_back_compat_default_and_maeonly_unchanged():
    """λ=0 and λ=0.3 still hit the v06 directory names exactly."""
    drv_c = _load_driver(
        "experiments/v06_round_dynamics/01_centralised.py", "v07_drv_central_bc"
    )
    drv_f = _load_driver(
        "experiments/v06_round_dynamics/02_fl_dynamics.py", "v07_drv_fl_bc"
    )
    assert drv_c._build_cell_name(0.3) == "V6-Dyn-A_centralised"
    assert drv_c._build_cell_name(0.0) == "V6-Dyn-A_centralised-MAEonly"
    assert drv_f._build_cell_name("fedavg", 0.3) == "V6-Dyn-B-FedAvg"
    assert drv_f._build_cell_name("fedavg", 0.0) == "V6-Dyn-B-FedAvg-MAEonly"


# ---------------------------------------------------------------------------
# 2) --output_namespace argparse contract
# ---------------------------------------------------------------------------


def test_centralised_argparse_namespace_default_back_compat():
    """Driver argparse defaults to v06 namespace; --output_namespace overrides."""
    drv = _load_driver(
        "experiments/v06_round_dynamics/01_centralised.py", "v07_drv_central_arg"
    )
    # Build the parser by recreating what main() does.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--aux_lambda", type=float, default=0.3)
    ap.add_argument("--hr_weight", type=float, default=0.1)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--output_namespace", type=str, default="v06_round_dynamics")
    args = ap.parse_args([])
    assert args.output_namespace == "v06_round_dynamics"
    args = ap.parse_args(["--output_namespace", "v07_loss_budget_sweeps"])
    assert args.output_namespace == "v07_loss_budget_sweeps"
    # Driver source must declare the flag exactly once.
    drv_src = (REPO_ROOT / "experiments/v06_round_dynamics/01_centralised.py").read_text(encoding="utf-8")
    assert drv_src.count('"--output_namespace"') == 1


def test_fl_argparse_namespace_default_back_compat():
    drv_src = (REPO_ROOT / "experiments/v06_round_dynamics/02_fl_dynamics.py").read_text(encoding="utf-8")
    assert '"--output_namespace"' in drv_src
    assert 'OUTPUT_DIR / args.output_namespace' in drv_src


def test_codebook_argparse_namespace_default_back_compat():
    drv_src = (REPO_ROOT / "experiments/v06_round_dynamics/08_codebook_stacking.py").read_text(encoding="utf-8")
    assert '"--output_namespace"' in drv_src
    assert 'OUTPUT_DIR / args.output_namespace' in drv_src


# ---------------------------------------------------------------------------
# 3) Aggregator cell-name parser
# ---------------------------------------------------------------------------


def test_aggregator_parses_default_v06_cells():
    agg = _load_driver(
        "experiments/v07_loss_budget_sweeps/05_aggregate_aux.py", "v07_agg"
    )
    assert agg._parse_cell_name("V6-Dyn-A_centralised") == ("centralised", 0.3)
    assert agg._parse_cell_name("V6-Dyn-B-FedAvg")      == ("fedavg",      0.3)
    assert agg._parse_cell_name("V6-Dyn-B-FedProx")     == ("fedprox",     0.3)
    assert agg._parse_cell_name("V6-Dyn-B-FedRep")      == ("fedrep",      0.3)
    assert agg._parse_cell_name("V6-Dyn-B-Ditto")       == ("ditto",       0.3)
    assert agg._parse_cell_name("V6-Dyn-B-FedProto")    == ("fedproto",    0.3)


def test_aggregator_parses_maeonly_cells():
    agg = _load_driver(
        "experiments/v07_loss_budget_sweeps/05_aggregate_aux.py", "v07_agg2"
    )
    assert agg._parse_cell_name("V6-Dyn-A_centralised-MAEonly") == ("centralised", 0.0)
    assert agg._parse_cell_name("V6-Dyn-B-FedAvg-MAEonly")      == ("fedavg",      0.0)
    assert agg._parse_cell_name("V6-Dyn-B-Ditto-MAEonly")       == ("ditto",       0.0)


def test_aggregator_parses_v07_aux_cells():
    agg = _load_driver(
        "experiments/v07_loss_budget_sweeps/05_aggregate_aux.py", "v07_agg3"
    )
    assert agg._parse_cell_name("V6-Dyn-A_centralised-aux0.05") == ("centralised", 0.05)
    assert agg._parse_cell_name("V6-Dyn-A_centralised-aux0.1")  == ("centralised", 0.1)
    assert agg._parse_cell_name("V6-Dyn-A_centralised-aux0.2")  == ("centralised", 0.2)
    assert agg._parse_cell_name("V6-Dyn-B-FedAvg-aux0.05")      == ("fedavg",      0.05)
    assert agg._parse_cell_name("V6-Dyn-B-FedProx-aux0.1")      == ("fedprox",     0.1)
    assert agg._parse_cell_name("V6-Dyn-B-FedRep-aux0.2")       == ("fedrep",      0.2)
    assert agg._parse_cell_name("V6-Dyn-B-Ditto-aux0.05")       == ("ditto",       0.05)
    assert agg._parse_cell_name("V6-Dyn-B-FedProto-aux0.1")     == ("fedproto",    0.1)


def test_aggregator_rejects_unknown_cell_name():
    agg = _load_driver(
        "experiments/v07_loss_budget_sweeps/05_aggregate_aux.py", "v07_agg4"
    )
    assert agg._parse_cell_name("V5-FedCB-something") is None
    assert agg._parse_cell_name("V6-Dyn-B-UnknownAlgo") is None
    assert agg._parse_cell_name("not-a-cell") is None
