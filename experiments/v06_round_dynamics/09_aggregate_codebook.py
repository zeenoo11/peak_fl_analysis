"""V6 Phase 2 aggregator — multi-seed × multi-cell codebook lift summary.

(한글 요약)
``08_codebook_stacking.py`` 가 cell × seed 별로 남긴
``codebook_lift.json`` 36 개 파일을 모아서 cell 별 3-seed mean ± std (ddof=1)
요약을 만든다. 출력은 ``outputs/v06_round_dynamics/codebook_lift_summary.json``.

기존 ``06_aggregate.py`` (Phase 1) 의 schema 와 동일한 컨벤션:
    cell: {
        "present_seeds": [...],
        "test_before":          {pape_mean: {mean, std, n}, ...},
        "test_after":           {... 같은 shape ...},
        "lift":                 {pape_delta: {mean, std, n}, ...},
        "test_before_pretty":   {pape: "56.34 ± 1.41", hr@1: ...},
        "test_after_pretty":    {... 같은 shape ...},
        "lift_pretty":          {pape: "-3.12 ± 0.45", ...},
        "codebook_diag":        {utilization: {mean, std}, ...},
        "elapsed_seconds":      {mean, std, n},
    }

Per-seed argparse — 이 파일은 read-only 집계라서 ``--seeds 42 123 7`` 한 번
호출이 자연스러움 (memory: feedback_aggregator_vs_runner_seed_argparse).
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


# 12 v06 cells (Phase 1) — same canonical order as 08_codebook_stacking.py.
DEFAULT_CELL_NAMES = [
    "V6-Dyn-A_centralised",
    "V6-Dyn-A_centralised-MAEonly",
    "V6-Dyn-B-FedAvg",
    "V6-Dyn-B-FedAvg-MAEonly",
    "V6-Dyn-B-FedProx",
    "V6-Dyn-B-FedProx-MAEonly",
    "V6-Dyn-B-FedRep",
    "V6-Dyn-B-FedRep-MAEonly",
    "V6-Dyn-B-Ditto",
    "V6-Dyn-B-Ditto-MAEonly",
    "V6-Dyn-B-FedProto",
    "V6-Dyn-B-FedProto-MAEonly",
]

_TEST_KEYS = ("pape_mean", "hr@1_mean", "hr@2_mean", "mae_mean", "mse_kw2_mean")
_LIFT_KEYS = ("pape_delta", "hr@1_delta", "hr@2_delta", "mae_delta", "mse_kw2_delta")
_DIAG_KEYS = ("utilization", "perplexity", "k_min", "k_max", "n_empty_clusters",
              "stage1_mean_inertia", "stage2_inertia")


def _discover_cells(seeds: list[int]) -> list[str]:
    """Scan ``outputs/v06_round_dynamics/seed{S}/*`` for cell directories
    that contain a ``codebook_lift.json``. Returns the union across the
    requested seeds, default cells listed first (preserving order) and any
    extras appended in sorted order.
    """
    found: set[str] = set()
    for seed in seeds:
        seed_dir = OUTPUT_DIR / "v06_round_dynamics" / f"seed{seed}"
        if not seed_dir.exists():
            continue
        for sub in seed_dir.iterdir():
            if not sub.is_dir():
                continue
            if (sub / "codebook_lift.json").exists():
                found.add(sub.name)
    ordered: list[str] = [c for c in DEFAULT_CELL_NAMES if c in found]
    extras = sorted(c for c in found if c not in DEFAULT_CELL_NAMES)
    return ordered + extras


def _load_cell_seed(seed: int, cell: str) -> dict:
    p = OUTPUT_DIR / "v06_round_dynamics" / f"seed{seed}" / cell / "codebook_lift.json"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    with p.open() as fh:
        return json.load(fh)


def _agg_metric(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "std":  float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n":    int(arr.size),
    }


def _format_mean_std(values: list[float], decimals: int = 2,
                     show_sign: bool = False) -> str:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return "n/a"
    m = float(arr.mean())
    s = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    sign = "+" if (show_sign and m >= 0) else ""
    return f"{sign}{m:.{decimals}f} ± {s:.{decimals}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="v06 Phase 2 multi-seed aggregator.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    ap.add_argument("--cells", type=str, nargs="*", default=None,
                    help="Override cell list. Default: auto-discover under "
                         "outputs/v06_round_dynamics/seed*/.")
    args = ap.parse_args()

    out_root = OUTPUT_DIR / "v06_round_dynamics"
    out_root.mkdir(parents=True, exist_ok=True)

    cell_names = args.cells if args.cells else _discover_cells(args.seeds)
    if not cell_names:
        print(f"  [warn] no cell directories with codebook_lift.json found under "
              f"seeds={args.seeds}; falling back to default cell list.")
        cell_names = DEFAULT_CELL_NAMES

    summary: dict = {"seeds": args.seeds, "cells": {}}

    for cell in cell_names:
        per_seed_before: dict[str, list[float]] = {k: [] for k in _TEST_KEYS}
        per_seed_after:  dict[str, list[float]] = {k: [] for k in _TEST_KEYS}
        per_seed_lift:   dict[str, list[float]] = {k: [] for k in _LIFT_KEYS}
        per_seed_diag:   dict[str, list[float]] = {k: [] for k in _DIAG_KEYS}
        per_seed_elapsed: list[float] = []
        per_seed_n_clients: list[int] = []
        per_seed_protocol: list[str] = []
        present_seeds: list[int] = []

        for seed in args.seeds:
            try:
                blob = _load_cell_seed(seed, cell)
            except FileNotFoundError as e:
                print(f"  [skip] {cell} seed={seed}: {e}")
                continue
            present_seeds.append(seed)
            for k in _TEST_KEYS:
                per_seed_before[k].append(float(blob["test_before"][k]))
                per_seed_after[k].append(float(blob["test_after"][k]))
            for k in _LIFT_KEYS:
                per_seed_lift[k].append(float(blob["lift"][k]))
            diag = blob["codebook_diag"]
            for k in _DIAG_KEYS:
                per_seed_diag[k].append(float(diag[k]))
            per_seed_elapsed.append(float(blob["elapsed_seconds"]))
            per_seed_n_clients.append(int(blob["n_clients"]))
            per_seed_protocol.append(str(blob["protocol"]))

        if not present_seeds:
            print(f"  [warn] {cell}: no seeds available — skipping.")
            continue

        cell_summary: dict = {
            "present_seeds": present_seeds,
            "protocol": per_seed_protocol[0],   # all 3 seeds share a protocol
            "n_clients_per_seed": per_seed_n_clients,
            "test_before": {k: _agg_metric(per_seed_before[k]) for k in _TEST_KEYS},
            "test_after":  {k: _agg_metric(per_seed_after[k])  for k in _TEST_KEYS},
            "lift":        {k: _agg_metric(per_seed_lift[k])   for k in _LIFT_KEYS},
            "codebook_diag": {k: _agg_metric(per_seed_diag[k]) for k in _DIAG_KEYS},
            "elapsed_seconds": _agg_metric(per_seed_elapsed),
        }
        cell_summary["test_before_pretty"] = {
            "pape":     _format_mean_std(per_seed_before["pape_mean"]),
            "hr@1":     _format_mean_std(per_seed_before["hr@1_mean"]),
            "hr@2":     _format_mean_std(per_seed_before["hr@2_mean"]),
            "mae":      _format_mean_std(per_seed_before["mae_mean"], decimals=4),
            "mse_kw2":  _format_mean_std(per_seed_before["mse_kw2_mean"], decimals=4),
        }
        cell_summary["test_after_pretty"] = {
            "pape":     _format_mean_std(per_seed_after["pape_mean"]),
            "hr@1":     _format_mean_std(per_seed_after["hr@1_mean"]),
            "hr@2":     _format_mean_std(per_seed_after["hr@2_mean"]),
            "mae":      _format_mean_std(per_seed_after["mae_mean"], decimals=4),
            "mse_kw2":  _format_mean_std(per_seed_after["mse_kw2_mean"], decimals=4),
        }
        cell_summary["lift_pretty"] = {
            "pape":     _format_mean_std(per_seed_lift["pape_delta"], show_sign=True),
            "hr@1":     _format_mean_std(per_seed_lift["hr@1_delta"], show_sign=True),
            "hr@2":     _format_mean_std(per_seed_lift["hr@2_delta"], show_sign=True),
            "mae":      _format_mean_std(per_seed_lift["mae_delta"], decimals=4, show_sign=True),
            "mse_kw2":  _format_mean_std(per_seed_lift["mse_kw2_delta"], decimals=4, show_sign=True),
        }
        summary["cells"][cell] = cell_summary

        print(f"  {cell:>32s}  [{cell_summary['protocol']}]  "
              f"BEFORE PAPE={cell_summary['test_before_pretty']['pape']}  "
              f"AFTER PAPE={cell_summary['test_after_pretty']['pape']}  "
              f"ΔPAPE={cell_summary['lift_pretty']['pape']}  "
              f"seeds={cell_summary['present_seeds']}")

    out_path = out_root / "codebook_lift_summary.json"
    with out_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[v06 phase2 aggregate] wrote {out_path}")


if __name__ == "__main__":
    main()
