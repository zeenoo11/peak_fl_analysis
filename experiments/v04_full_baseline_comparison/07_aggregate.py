"""v04 multi-seed aggregate — collect every per-seed result.json into one summary.

Reads everything under ``outputs/v04_full_baseline_comparison/seed{S}/`` and
the seed-independent G6 / G7 outputs, builds a single
``multiseed_summary.json`` with mean ± sample-std (ddof=1) across the
3 seeds — same convention as v02 07_aggregate_seeds.py.

CLI (no per-seed argument; aggregator scans all seeds present):

    uv run python experiments/v04_full_baseline_comparison/07_aggregate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from config import OUTPUT_DIR

V04_OUT_ROOT = OUTPUT_DIR / "v04_full_baseline_comparison"

# Method labels we expect to find under each seed dir, with the per-method
# subdirectory and the result JSON shape.
METHOD_DIRS = [
    # FL baselines
    "fedavg", "fedprox", "fedrep", "ditto", "local_only",
    # NF baselines
    "nf_dlinear", "nf_nhits", "nf_crossformer",
    # FM zero-shot
    "fm_chronos_bolt_small", "fm_chronos_t5_tiny", "fm_timesfm",
    # G5 cross-cell
    "peakvq_on_fedavg", "peakvq_on_fedrep",
]


def _agg(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0, "values": []}
    n = arr.size
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if n > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(n),
        "values": [float(v) for v in arr],
    }


def _load_seed_result(seed: int, method: str) -> dict | None:
    p = V04_OUT_ROOT / f"seed{seed}" / method / "result.json"
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


def _extract_cold(method: str, r: dict) -> dict | None:
    """Method-specific cold-metrics extraction.

    Most methods write a top-level ``cold_metrics`` block. The G5
    cross-cell scripts (peakvq_on_*) instead write a baseline + two
    op-points; for paper reporting we collapse those to **PAPE-aggressive**
    (the v04 default headline op-point).
    """
    if method.startswith("peakvq_on_"):
        op = (r.get("operating_points") or {}).get("PAPE-aggressive") or {}
        return op.get("metrics") if op else None
    return r.get("cold_metrics")


def _aggregate_method(method: str, seeds: list[int]) -> dict:
    """For one method, gather cold metrics across seeds and aggregate."""
    per_seed = {}
    cold_pape, cold_hr1, cold_hr2, cold_mae = [], [], [], []
    elapsed = []
    for s in seeds:
        r = _load_seed_result(s, method)
        if r is None:
            continue
        cm = _extract_cold(method, r) or {}
        per_seed[str(s)] = {
            "pape": cm.get("pape"),
            "hr@1": cm.get("hr@1"),
            "hr@2": cm.get("hr@2"),
            "mae": cm.get("mae"),
            "elapsed_seconds": r.get("elapsed_seconds"),
            "n_cold_windows": cm.get("n_cold_windows"),
        }
        if cm.get("pape") is not None and not np.isnan(cm["pape"]):
            cold_pape.append(cm["pape"])
            cold_hr1.append(cm["hr@1"])
            cold_hr2.append(cm["hr@2"])
            cold_mae.append(cm["mae"])
        if r.get("elapsed_seconds") is not None:
            elapsed.append(r["elapsed_seconds"])
    return {
        "method": method,
        "seeds_present": list(per_seed.keys()),
        "per_seed": per_seed,
        "agg": {
            "pape": _agg(cold_pape),
            "hr@1": _agg(cold_hr1),
            "hr@2": _agg(cold_hr2),
            "mae": _agg(cold_mae),
            "elapsed_seconds": _agg(elapsed),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="v04 multi-seed aggregate.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7])
    args = ap.parse_args()

    methods = []
    for m in METHOD_DIRS:
        agg = _aggregate_method(m, args.seeds)
        methods.append(agg)

    # G6 / G7 (seed-independent)
    het_path = V04_OUT_ROOT / "heterogeneity_summary.json"
    com_path = V04_OUT_ROOT / "communication_summary.json"
    G6 = json.load(open(het_path)) if het_path.exists() else None
    G7 = json.load(open(com_path)) if com_path.exists() else None

    summary = {
        "seeds_requested": args.seeds,
        "methods": methods,
        "G6_heterogeneity": (
            None if G6 is None else {
                "summary_stats": G6.get("summary_stats"),
                "correlation": G6.get("correlation"),
                "n_apts": G6.get("n_apts"),
            }
        ),
        "G7_communication": (
            None if G7 is None else {
                "n_clients": G7.get("n_clients"),
                "n_rounds_iterative_FL": G7.get("n_rounds_iterative_FL"),
                "methods": G7.get("methods"),
            }
        ),
    }

    out_path = V04_OUT_ROOT / "multiseed_summary.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[v04 agg] saved -> {out_path}")
    print()
    print(f"  {'method':<24} {'seeds':<10} {'PAPE mean±std':<18} {'HR@1 mean±std':<18}")
    print(f"  {'-'*24} {'-'*10} {'-'*18} {'-'*18}")
    for m in methods:
        if not m["seeds_present"]:
            continue
        a = m["agg"]
        if a["pape"]["n"] == 0:
            continue
        ps = f"{a['pape']['mean']:.2f} ± {a['pape']['std']:.2f}" if a["pape"]["n"] > 1 else f"{a['pape']['mean']:.2f}"
        hs = f"{a['hr@1']['mean']:.1f} ± {a['hr@1']['std']:.2f}" if a["hr@1"]["n"] > 1 else f"{a['hr@1']['mean']:.1f}"
        print(f"  {m['method']:<24} {','.join(m['seeds_present']):<10} {ps:<18} {hs:<18}")


if __name__ == "__main__":
    main()
