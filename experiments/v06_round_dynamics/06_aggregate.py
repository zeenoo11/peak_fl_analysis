"""V6 Phase 1 aggregator — multi-seed summary + trajectory arrays.

(한글 요약)
plan v06-01 §"Build order" step 8. 18 runs (= 6 cells × 3 seeds) 의
``round_log.jsonl`` + ``result.json`` 을 모두 읽어:
- ``multiseed_summary.json`` — per-cell terminal val/test 의 3-seed mean ± std
  (ddof=1) — conference Table (papers/conference_draft/presentation.md
  line 197-204) 와 같은 schema 의 ``pape: 56.34 ± 1.41`` 형식.
- ``trajectories.npz`` — per-cell per-seed 라운드별 array 를 stack 해서 figures
  스크립트가 바로 plot 할 수 있게.

Reads memory-only (no MLflow). Aggregates everything under
``outputs/v06_round_dynamics/seed{42,123,7}/{cell_name}/``.

Per-seed argparse — 이 파일은 read-only 집계라서 ``--seeds 42 123 7`` 한 번
호출이 자연스러움 (memory:
feedback_aggregator_vs_runner_seed_argparse).
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


DEFAULT_CELL_NAMES = [
    "V6-Dyn-A_centralised",
    "V6-Dyn-B-FedAvg",
    "V6-Dyn-B-FedProx",
    "V6-Dyn-B-FedRep",
    "V6-Dyn-B-Ditto",
    "V6-Dyn-B-FedProto",
]


def _discover_cells(seeds: list[int]) -> list[str]:
    """Scan ``outputs/v06_round_dynamics/seed{S}/*`` for cell directories
    that contain a ``result.json``. Returns the union across the requested
    seeds, with the default cells listed first (preserving order) and any
    extra cells (e.g. ``-MAEonly`` ablations) appended in sorted order.

    A directory is treated as a cell iff at least one of the requested
    seeds has a ``result.json`` under it.
    """
    found: set[str] = set()
    for seed in seeds:
        seed_dir = OUTPUT_DIR / "v06_round_dynamics" / f"seed{seed}"
        if not seed_dir.exists():
            continue
        for sub in seed_dir.iterdir():
            if not sub.is_dir():
                continue
            if (sub / "result.json").exists():
                found.add(sub.name)
    ordered: list[str] = [c for c in DEFAULT_CELL_NAMES if c in found]
    extras = sorted(c for c in found if c not in DEFAULT_CELL_NAMES)
    return ordered + extras


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _load_cell_seed(seed: int, cell: str) -> tuple[dict, list[dict]]:
    cell_dir = OUTPUT_DIR / "v06_round_dynamics" / f"seed{seed}" / cell
    result_path = cell_dir / "result.json"
    log_path = cell_dir / "round_log.jsonl"
    if not result_path.exists():
        raise FileNotFoundError(f"missing {result_path}")
    if not log_path.exists():
        raise FileNotFoundError(f"missing {log_path}")
    with result_path.open() as fh:
        result = json.load(fh)
    rows = _read_jsonl(log_path)
    return result, rows


def _agg_metric(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "std":  float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n":    int(arr.size),
    }


def _format_mean_std(values: list[float], decimals: int = 2) -> str:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return "n/a"
    m = float(arr.mean())
    s = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return f"{m:.{decimals}f} ± {s:.{decimals}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="v06 Phase 1 multi-seed aggregator.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    ap.add_argument("--cells", type=str, nargs="*", default=None,
                    help="Override the cell list. Default: auto-discover under "
                         "outputs/v06_round_dynamics/seed*/.")
    args = ap.parse_args()

    out_root = OUTPUT_DIR / "v06_round_dynamics"
    out_root.mkdir(parents=True, exist_ok=True)

    cell_names = args.cells if args.cells else _discover_cells(args.seeds)
    if not cell_names:
        print(f"  [warn] no cell directories with result.json found under seeds={args.seeds}; "
              f"falling back to default cell list.")
        cell_names = DEFAULT_CELL_NAMES

    summary: dict = {"seeds": args.seeds, "cells": {}}
    trajectories: dict = {}

    for cell in cell_names:
        cell_summary: dict = {"present_seeds": []}
        per_seed_term_val:  dict[str, list[float]] = {k: [] for k in
            ("pape_mean", "hr@1_mean", "hr@2_mean", "mae_mean", "mse_kw2_mean")}
        per_seed_term_test: dict[str, list[float]] = {k: [] for k in
            ("pape_mean", "hr@1_mean", "hr@2_mean", "mae_mean", "mse_kw2_mean")}
        per_seed_drift_mean: list[float] = []
        per_seed_upload_cum: list[int] = []
        per_seed_broadcast_cum: list[int] = []
        per_seed_elapsed: list[float] = []

        # Trajectories — store one numpy array per cell with shape (n_seeds, n_rounds).
        traj_rows = []
        n_rounds_seen: int | None = None

        for seed in args.seeds:
            try:
                result, rows = _load_cell_seed(seed, cell)
            except FileNotFoundError as e:
                print(f"  [skip] {cell} seed={seed}: {e}")
                continue
            cell_summary["present_seeds"].append(seed)

            # Terminal val/test from result.json (= last logged terminal row).
            for k in per_seed_term_val:
                per_seed_term_val[k].append(float(result["val_terminal"][k]))
                per_seed_term_test[k].append(float(result["test_terminal"][k]))
            per_seed_drift_mean.append(float(result.get("drift_l2_mean_over_rounds", 0.0)))
            per_seed_upload_cum.append(int(result["comm_total_bytes"]["upload_cum"]))
            per_seed_broadcast_cum.append(int(result["comm_total_bytes"]["broadcast_cum"]))
            per_seed_elapsed.append(float(result["elapsed_seconds"]))

            # Trajectory: in-train rows only (round >= 1). For Centralised
            # cell the round_idx is the epoch index.
            in_train = [r for r in rows if r["round"] >= 1]

            def _test_field(r: dict, key: str) -> float:
                """Read a per-round test metric. Legacy rows without a `test`
                block (older jsonl) yield NaN so figures don't break on partial
                refresh."""
                blk = r.get("test")
                if not blk or key not in blk:
                    return float("nan")
                return float(blk[key])

            def _train_loss(r: dict) -> float:
                tr = r.get("train") or {}
                v = tr.get("loss_mean_last_epoch")
                return float("nan") if v is None else float(v)

            traj_rows.append({
                "seed": seed,
                "round_idx": np.array([r["round"] for r in in_train], dtype=np.int64),
                "val_pape_mean":   np.array([r["val"]["pape_mean"]   for r in in_train], dtype=np.float64),
                "val_hr1_mean":    np.array([r["val"]["hr@1_mean"]   for r in in_train], dtype=np.float64),
                "val_hr2_mean":    np.array([r["val"]["hr@2_mean"]   for r in in_train], dtype=np.float64),
                "val_mae_mean":    np.array([r["val"]["mae_mean"]    for r in in_train], dtype=np.float64),
                "val_mse_kw2_mean": np.array([r["val"]["mse_kw2_mean"] for r in in_train], dtype=np.float64),
                # Round-level test trajectory (paper convention). Legacy rows → NaN.
                "test_pape_mean":   np.array([_test_field(r, "pape_mean")   for r in in_train], dtype=np.float64),
                "test_hr1_mean":    np.array([_test_field(r, "hr@1_mean")   for r in in_train], dtype=np.float64),
                "test_hr2_mean":    np.array([_test_field(r, "hr@2_mean")   for r in in_train], dtype=np.float64),
                "test_mae_mean":    np.array([_test_field(r, "mae_mean")    for r in in_train], dtype=np.float64),
                "test_mse_kw2_mean": np.array([_test_field(r, "mse_kw2_mean") for r in in_train], dtype=np.float64),
                # Round-averaged training main loss (FedProx Figure 2 convention).
                "train_loss_main":  np.array([_train_loss(r) for r in in_train], dtype=np.float64),
                "drift_l2":        np.array([r["drift_l2"]            for r in in_train], dtype=np.float64),
                "upload_bytes_cum": np.array([r["comm"]["upload_bytes_cum"] for r in in_train], dtype=np.int64),
                "broadcast_bytes_cum": np.array([r["comm"]["broadcast_bytes_cum"] for r in in_train], dtype=np.int64),
            })

        if not cell_summary["present_seeds"]:
            print(f"  [warn] {cell}: no seeds available — skipping aggregation.")
            continue

        # Stack trajectories into (n_seeds, n_rounds) arrays. If lengths
        # differ across seeds, pad with NaN at the tail.
        max_len = max(len(t["round_idx"]) for t in traj_rows)
        def stack_pad(key, dtype):
            arrs = []
            for t in traj_rows:
                a = t[key].astype(np.float64) if dtype is np.float64 else t[key].astype(np.int64)
                if len(a) < max_len:
                    pad_val = np.nan if dtype is np.float64 else 0
                    a = np.concatenate([a, np.full(max_len - len(a), pad_val, dtype=a.dtype)])
                arrs.append(a)
            return np.stack(arrs, axis=0)

        trajectories[f"{cell}_round_idx"]              = stack_pad("round_idx", np.int64)
        trajectories[f"{cell}_val_pape_mean"]          = stack_pad("val_pape_mean", np.float64)
        trajectories[f"{cell}_val_hr1_mean"]           = stack_pad("val_hr1_mean", np.float64)
        trajectories[f"{cell}_val_hr2_mean"]           = stack_pad("val_hr2_mean", np.float64)
        trajectories[f"{cell}_val_mae_mean"]           = stack_pad("val_mae_mean", np.float64)
        trajectories[f"{cell}_val_mse_kw2_mean"]       = stack_pad("val_mse_kw2_mean", np.float64)
        # Round-level test trajectory (1A). Legacy jsonl rows → NaN columns.
        trajectories[f"{cell}_test_pape_mean"]         = stack_pad("test_pape_mean", np.float64)
        trajectories[f"{cell}_test_hr1_mean"]          = stack_pad("test_hr1_mean", np.float64)
        trajectories[f"{cell}_test_hr2_mean"]          = stack_pad("test_hr2_mean", np.float64)
        trajectories[f"{cell}_test_mae_mean"]          = stack_pad("test_mae_mean", np.float64)
        trajectories[f"{cell}_test_mse_kw2_mean"]      = stack_pad("test_mse_kw2_mean", np.float64)
        # Round-averaged training main loss (1A — FedProx Figure 2 style).
        trajectories[f"{cell}_train_loss_main"]        = stack_pad("train_loss_main", np.float64)
        trajectories[f"{cell}_drift_l2"]               = stack_pad("drift_l2", np.float64)
        trajectories[f"{cell}_upload_bytes_cum"]       = stack_pad("upload_bytes_cum", np.int64)
        trajectories[f"{cell}_broadcast_bytes_cum"]    = stack_pad("broadcast_bytes_cum", np.int64)
        trajectories[f"{cell}_seeds"]                  = np.array(cell_summary["present_seeds"], dtype=np.int64)

        cell_summary["val_terminal"]  = {k: _agg_metric(per_seed_term_val[k])  for k in per_seed_term_val}
        cell_summary["test_terminal"] = {k: _agg_metric(per_seed_term_test[k]) for k in per_seed_term_test}
        cell_summary["val_terminal_pretty"] = {
            "pape": _format_mean_std(per_seed_term_val["pape_mean"]),
            "hr@1": _format_mean_std(per_seed_term_val["hr@1_mean"]),
            "hr@2": _format_mean_std(per_seed_term_val["hr@2_mean"]),
            "mae":  _format_mean_std(per_seed_term_val["mae_mean"], decimals=4),
            "mse_kw2": _format_mean_std(per_seed_term_val["mse_kw2_mean"], decimals=4),
        }
        cell_summary["test_terminal_pretty"] = {
            "pape": _format_mean_std(per_seed_term_test["pape_mean"]),
            "hr@1": _format_mean_std(per_seed_term_test["hr@1_mean"]),
            "hr@2": _format_mean_std(per_seed_term_test["hr@2_mean"]),
            "mae":  _format_mean_std(per_seed_term_test["mae_mean"], decimals=4),
            "mse_kw2": _format_mean_std(per_seed_term_test["mse_kw2_mean"], decimals=4),
        }
        cell_summary["drift_l2_mean_over_rounds"] = _agg_metric(per_seed_drift_mean)
        cell_summary["comm_total_bytes"] = {
            "upload_cum":    _agg_metric(per_seed_upload_cum),
            "broadcast_cum": _agg_metric(per_seed_broadcast_cum),
        }
        cell_summary["elapsed_seconds"] = _agg_metric(per_seed_elapsed)
        summary["cells"][cell] = cell_summary

        print(f"  {cell:>22s}  val.PAPE={cell_summary['val_terminal_pretty']['pape']}  "
              f"test.PAPE={cell_summary['test_terminal_pretty']['pape']}  "
              f"seeds={cell_summary['present_seeds']}")

    # Persist.
    with (out_root / "multiseed_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    np.savez_compressed(out_root / "trajectories.npz", **trajectories)
    print(f"[v06 aggregate] wrote {out_root}/multiseed_summary.json and trajectories.npz")


if __name__ == "__main__":
    main()
