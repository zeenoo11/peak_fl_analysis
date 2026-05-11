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
import re
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]


_FLAG_DECL = re.compile(r"""['"]--output_namespace['"]""")
_NS_DEFAULT = re.compile(
    r"""default\s*=\s*['"]v06_round_dynamics['"]"""
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


def _assert_namespace_contract(rel_path: str, *, require_default: bool) -> None:
    """Quote-insensitive check that the driver declares ``--output_namespace``.

    - Exactly one declaration of the flag string (single or double quoted).
    - If ``require_default``, a ``default='v06_round_dynamics'`` clause is
      present (same quote-insensitive scan).
    - The output path uses ``OUTPUT_DIR / args.output_namespace`` so the
      flag is actually consumed.
    """
    src = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    decls = _FLAG_DECL.findall(src)
    assert len(decls) == 1, (
        f"{rel_path}: '--output_namespace' declared {len(decls)} times "
        f"(expected exactly 1)"
    )
    if require_default:
        assert _NS_DEFAULT.search(src), (
            f"{rel_path}: missing default='v06_round_dynamics' near "
            f"--output_namespace"
        )
    assert "OUTPUT_DIR / args.output_namespace" in src, (
        f"{rel_path}: output path does not consume args.output_namespace"
    )


def test_centralised_argparse_namespace_default_back_compat():
    _assert_namespace_contract(
        "experiments/v06_round_dynamics/01_centralised.py", require_default=True
    )


def test_fl_argparse_namespace_default_back_compat():
    _assert_namespace_contract(
        "experiments/v06_round_dynamics/02_fl_dynamics.py", require_default=True
    )


def test_codebook_argparse_namespace_default_back_compat():
    _assert_namespace_contract(
        "experiments/v06_round_dynamics/08_codebook_stacking.py", require_default=True
    )


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
