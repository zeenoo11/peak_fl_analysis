"""v07-A — λ_aux sweep aggregator.

(한글 요약)
plan ``v07-01_loss_and_budget_sweeps.md`` §1 (v07-A) — `λ_aux ∈ {0, 0.05, 0.1, 0.2, 0.3}`
× 6 algorithms × 3 seeds 결과를 단일 ``aux_sweep_summary.json`` 으로 통합.

v07-A 의 새 결과 (`λ ∈ {0.05, 0.1, 0.2}`) 는 ``outputs/v07_loss_budget_sweeps/``
에서, v06 의 기존 결과 (`λ = 0.0` MAEonly + `λ = 0.3` default) 는
``outputs/v06_round_dynamics/`` 에서 각각 로드한다 (default; ``--no_v06_baseline``
으로 v06 read 비활성화 가능).

산출물:

    outputs/v07_loss_budget_sweeps/aux_sweep_summary.json
    {
      "schema_version": "v07-A.1",
      "seeds": [42, 123, 7],
      "lambdas": [0.0, 0.05, 0.1, 0.2, 0.3],
      "algorithms": ["centralised", "fedavg", "fedprox", "fedrep", "ditto", "fedproto"],
      "test_pape": {  # algorithm -> lambda -> {mean, std, n_seeds}
        "fedavg": {
          "0.0":  {"mean": 38.55, "std": 1.52, "n": 3, "source": "v06_round_dynamics"},
          "0.05": {"mean": 39.20, "std": 1.61, "n": 3, "source": "v07_loss_budget_sweeps"},
          ...
        },
        ...
      },
      "test_mae":  {...},  # same shape
      "val_pape":  {...},  # same shape
      "val_mae":   {...},  # same shape
    }

CLI:

    uv run python experiments/v07_loss_budget_sweeps/05_aggregate_aux.py \\
        --seeds 42 123 7
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from config import OUTPUT_DIR  # noqa: E402


V07_NAMESPACE = "v07_loss_budget_sweeps"
V06_NAMESPACE = "v06_round_dynamics"

ALGO_ORDER = ["centralised", "fedavg", "fedprox", "fedrep", "ditto", "fedproto"]
LAMBDA_ORDER = [0.0, 0.05, 0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Cell-name parsing
# ---------------------------------------------------------------------------


_CELL_RE = re.compile(
    r"^V6-Dyn-(?:A_centralised|B-(FedAvg|FedProx|FedRep|Ditto|FedProto))"
    r"(?:-MAEonly|-aux(?P<lam>[0-9.]+))?$"
)
_ALGO_FROM_PRETTY = {
    "FedAvg":   "fedavg",
    "FedProx":  "fedprox",
    "FedRep":   "fedrep",
    "Ditto":    "ditto",
    "FedProto": "fedproto",
}


def _parse_cell_name(cell: str) -> tuple[str, float] | None:
    """``V6-Dyn-A_centralised-aux0.1`` → ('centralised', 0.1).

    Returns ``None`` if the cell name does not match the v06 schema.
    """
    m = _CELL_RE.match(cell)
    if m is None:
        return None
    pretty = m.group(1)
    if pretty is None:
        algo = "centralised"
    else:
        algo = _ALGO_FROM_PRETTY[pretty]
    if "-MAEonly" in cell:
        lam = 0.0
    elif "-aux" in cell:
        lam = float(m.group("lam"))
    else:
        lam = 0.3
    return algo, lam


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------


def _scan_namespace(namespace: str, seeds: list[int]) -> dict[tuple[str, float], dict[int, dict]]:
    """Walk ``outputs/{namespace}/seed{S}/<cell>/result.json`` files.

    Returns:
        {(algorithm, lambda) -> {seed -> result.json dict}}
    """
    root = OUTPUT_DIR / namespace
    out: dict[tuple[str, float], dict[int, dict]] = {}
    if not root.exists():
        return out
    for s in seeds:
        seed_dir = root / f"seed{s}"
        if not seed_dir.is_dir():
            continue
        for cell_dir in seed_dir.iterdir():
            if not cell_dir.is_dir():
                continue
            parsed = _parse_cell_name(cell_dir.name)
            if parsed is None:
                continue
            result_path = cell_dir / "result.json"
            if not result_path.exists():
                continue
            with result_path.open() as fh:
                result = json.load(fh)
            out.setdefault(parsed, {})[s] = result
    return out


def _agg_seeds(values: list[float]) -> dict:
    """Mean / std / n_seeds aggregator (Bessel-corrected std for n >= 2)."""
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"mean": None, "std": None, "n": 0}
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return {"mean": mean, "std": std, "n": int(a.size)}


def _build_metric_table(
    by_key_v07: dict[tuple[str, float], dict[int, dict]],
    by_key_v06: dict[tuple[str, float], dict[int, dict]],
    field_path: tuple[str, str],
) -> dict:
    """Build {algorithm -> {lambda_str -> {mean, std, n, source}}} for one metric."""
    table: dict[str, dict[str, dict]] = {}
    for algo in ALGO_ORDER:
        table[algo] = {}
        for lam in LAMBDA_ORDER:
            key = (algo, lam)
            if key in by_key_v07:
                seed_results = by_key_v07[key]
                source = V07_NAMESPACE
            elif key in by_key_v06:
                seed_results = by_key_v06[key]
                source = V06_NAMESPACE
            else:
                continue
            values = []
            for seed, r in seed_results.items():
                node = r
                for f in field_path:
                    node = node.get(f) if isinstance(node, dict) else None
                if node is None:
                    continue
                values.append(float(node))
            if not values:
                continue
            agg = _agg_seeds(values)
            agg["source"] = source
            table[algo][f"{lam:g}"] = agg
    return table


def main() -> None:
    ap = argparse.ArgumentParser(
        description="v07-A λ_aux sweep aggregator (algo × λ matrix)."
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    ap.add_argument("--no_v06_baseline", action="store_true",
                    help="Do not read v06 results for λ ∈ {0, 0.3}.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Override output path (default: "
                         "outputs/v07_loss_budget_sweeps/aux_sweep_summary.json).")
    args = ap.parse_args()

    seeds = list(args.seeds)
    by_key_v07 = _scan_namespace(V07_NAMESPACE, seeds)
    by_key_v06 = (
        {} if args.no_v06_baseline
        else _scan_namespace(V06_NAMESPACE, seeds)
    )

    print(f"[v07-A agg] v07 cells found: {len(by_key_v07)}")
    print(f"[v07-A agg] v06 cells found: {len(by_key_v06)}")
    if not by_key_v07 and not by_key_v06:
        print("[v07-A agg] WARN — no cells found. Did you run 01_run_aux_sweep.py?")

    summary = {
        "schema_version": "v07-A.1",
        "seeds": seeds,
        # Use ``f"{lam:g}"`` so the keys match the metric-table keys
        # (e.g. 0.0 -> "0", not "0.0"). Figure look-ups depend on this.
        "lambdas": [f"{lam:g}" for lam in LAMBDA_ORDER],
        "algorithms": ALGO_ORDER,
        "test_pape": _build_metric_table(by_key_v07, by_key_v06, ("test_terminal", "pape_mean")),
        "test_mae":  _build_metric_table(by_key_v07, by_key_v06, ("test_terminal", "mae_mean")),
        "test_hr1":  _build_metric_table(by_key_v07, by_key_v06, ("test_terminal", "hr@1_mean")),
        "val_pape":  _build_metric_table(by_key_v07, by_key_v06, ("val_terminal", "pape_mean")),
        "val_mae":   _build_metric_table(by_key_v07, by_key_v06, ("val_terminal", "mae_mean")),
    }

    out_path = args.out or (OUTPUT_DIR / V07_NAMESPACE / "aux_sweep_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[v07-A agg] wrote {out_path}")

    # --------------- terminal pretty-print --------------------
    print()
    print(f"  {'algorithm':>12s}  ", end="")
    for lam in LAMBDA_ORDER:
        print(f"{f'λ={lam:g}':>14s}  ", end="")
    print()
    print(f"  {'-' * 12}  " + ("-" * 14 + "  ") * len(LAMBDA_ORDER))
    for algo in ALGO_ORDER:
        print(f"  {algo:>12s}  ", end="")
        for lam in LAMBDA_ORDER:
            cell = summary["test_pape"][algo].get(f"{lam:g}")
            if cell is None:
                print(f"{'--':>14s}  ", end="")
            else:
                m, s = cell["mean"], cell["std"]
                print(f"{f'{m:6.2f}+/-{s:4.2f}':>14s}  ", end="")
        print()


if __name__ == "__main__":
    main()
