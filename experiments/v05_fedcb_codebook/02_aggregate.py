"""Aggregate v05 FedCB results across seeds {42, 123, 7}.

(한글 요약)
``outputs/v05_fedcb_codebook/seed{42,123,7}/{cell}/result.json``를 모두 읽어
3-seed 평균 / 표준편차를 계산하고, plan §Go/No-go gates의 Gate 1 / 2 / 3
체크 결과를 포함한 ``multiseed_summary.json``을 쓴다.

V5-FedCB-0 (Gate 1 anchor)는 v02 §B.3에 이미 발표된 CMO row를 그대로 참조하므로,
이 스크립트는 ``outputs/v02_fl_8020_ratio/seed{S}/W_component_results.json``의
``per_operating_point.PAPE-aggressive.cells.V0`` (R0 routing, T2 backbone, α=1.5,
W1a=0)을 직접 로드하여 ``v5_fedcb_0_paper_anchor`` 라는 이름의 가상 셀로
multiseed_summary에 포함시킨다. 이 셀은 v05 result.json에서 오지 않으므로
``_load_v02_anchor()``가 별도 경로로 만든다.

스키마는 ``outputs/v04_full_baseline_comparison/multiseed_summary.json``
(09_fix_rerun 버전)을 참조해 ``mean ± std`` (ddof=1) + ``per_seed_*``
필드를 일관되게 두었다. 셀 이름은 ``--K_local K [--alpha A]``가 만들어낸
디렉토리명을 그대로 사용 (``fedcb_K4`` / ``fedcb_K2`` / ``fedcb_K8`` /
``fedcb_K4_alpha1.5`` ...). + 가상 셀 ``v5_fedcb_0_paper_anchor``.

CLI:
    uv run python experiments/v05_fedcb_codebook/02_aggregate.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import OUTPUT_DIR

V02_OUT_ROOT = OUTPUT_DIR / "v02_fl_8020_ratio"
V05_OUT_ROOT = OUTPUT_DIR / "v05_fedcb_codebook"
SEEDS = [42, 123, 7]

# Plan §Go/No-go gates thresholds.
# Gate 1: V5-FedCB-0 PAPE within ±1.5 pp of v02 §B.3 anchor 44.18.
GATE1_ANCHOR = 44.18
GATE1_TOLERANCE = 1.5
# Gate 2 / 3: V5-FedCB-* PAPE ≤ 52 %.
GATE_2_3_THRESHOLD = 52.0

# Synthetic cell name for the V5-FedCB-0 anchor row (loaded from v02 JSONs,
# not from a v05 run).
V5_FEDCB_0_CELL = "v5_fedcb_0_paper_anchor"


def _mean_std(vals: list[float]) -> dict:
    """Return mean + sample std (ddof=1) with ``min`` / ``max`` for sanity."""
    if len(vals) == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0, "n": 1, "values": list(vals)}
    return {
        "mean": float(statistics.mean(vals)),
        "std": float(statistics.stdev(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "n": len(vals),
        "values": [float(v) for v in vals],
    }


def _discover_cells() -> list[str]:
    """Find every cell directory present under at least one seed."""
    cells: set[str] = set()
    for seed in SEEDS:
        seed_dir = V05_OUT_ROOT / f"seed{seed}"
        if not seed_dir.exists():
            continue
        for sub in seed_dir.iterdir():
            if sub.is_dir() and (sub / "result.json").exists():
                cells.add(sub.name)
    return sorted(cells)


def _load_cell(cell: str) -> dict:
    """Aggregate one v05 cell across all seeds where its result.json exists.

    Returns
    -------
    {
        "cell": str,
        "seeds_present": list[int],
        "per_seed": { seed: {"fl_only": {...}, "with_codebook_cmo": {...}} },
        "agg": {
            "fl_only": {"pape": ..., "hr@1": ..., "hr@2": ..., "mae": ...},
            "with_codebook_cmo": {"pape": ..., "hr@1": ..., "hr@2": ..., "mae": ...},
        },
        "config": dict (from any one seed; assumed identical across seeds for the cell),
    }
    """
    per_seed: dict = {}
    fl_pape, fl_hr1, fl_hr2, fl_mae = [], [], [], []
    cb_pape, cb_hr1, cb_hr2, cb_mae = [], [], [], []
    config_any = None
    mode_any = None
    alpha_any = None
    for seed in SEEDS:
        rj = V05_OUT_ROOT / f"seed{seed}" / cell / "result.json"
        if not rj.exists():
            continue
        with open(rj) as fh:
            r = json.load(fh)
        per_seed[seed] = {
            "fl_only": r["fl_only"],
            "with_codebook_cmo": r["with_codebook_cmo"],
            "vq_diagnostics": r.get("vq_diagnostics", {}),
            "elapsed_seconds": r.get("elapsed_seconds", {}),
        }
        fl_pape.append(r["fl_only"]["pape"])
        fl_hr1.append(r["fl_only"]["hr@1"])
        fl_hr2.append(r["fl_only"]["hr@2"])
        fl_mae.append(r["fl_only"]["mae"])
        m = r["with_codebook_cmo"]["metrics"]
        cb_pape.append(m["pape"])
        cb_hr1.append(m["hr@1"])
        cb_hr2.append(m["hr@2"])
        cb_mae.append(m["mae"])
        if config_any is None:
            config_any = r.get("config", {})
            mode_any = r.get("mode")
            alpha_any = r.get("alpha")
    return {
        "cell": cell,
        "seeds_present": sorted(per_seed.keys()),
        "mode": mode_any,
        "alpha": alpha_any,
        "config": config_any,
        "per_seed": per_seed,
        "agg": {
            "fl_only": {
                "pape": _mean_std(fl_pape),
                "hr@1": _mean_std(fl_hr1),
                "hr@2": _mean_std(fl_hr2),
                "mae": _mean_std(fl_mae),
            },
            "with_codebook_cmo": {
                "pape": _mean_std(cb_pape),
                "hr@1": _mean_std(cb_hr1),
                "hr@2": _mean_std(cb_hr2),
                "mae": _mean_std(cb_mae),
            },
        },
    }


def _load_v02_anchor() -> dict | None:
    """Build the V5-FedCB-0 anchor row from v02 §B.3 published CMO numbers.

    For each seed we read
    ``outputs/v02_fl_8020_ratio/seed{S}/W_component_results.json`` and pull
    ``per_operating_point.PAPE-aggressive.cells.V0`` — that is the v02 R0-routed
    T2 backbone with α_v0=1.5 and α_w1=0 (CMO-only), which is exactly the
    V5-FedCB-0 anchor referenced by the plan.

    Returns ``None`` if no seed had the v02 JSON, otherwise a dict shaped like
    ``_load_cell()`` so it can sit alongside the federated cells in the
    ``cells`` map.
    """
    per_seed: dict = {}
    cb_pape, cb_hr1, cb_hr2, cb_mae = [], [], [], []
    for seed in SEEDS:
        rj = V02_OUT_ROOT / f"seed{seed}" / "W_component_results.json"
        if not rj.exists():
            continue
        try:
            with open(rj) as fh:
                obj = json.load(fh)
            v0 = obj["per_operating_point"]["PAPE-aggressive"]["cells"]["V0"]
        except (KeyError, json.JSONDecodeError) as e:
            print(f"[v05 aggregate] WARN: seed{seed} v02 anchor missing or malformed: {e}")
            continue
        per_seed[seed] = {
            # No fl_only baseline applies here — this row is a published
            # CMO number, not a v05 cold-eval pass — so we only fill the
            # with_codebook_cmo branch. fl_only stays empty for schema
            # symmetry but is not aggregated.
            "with_codebook_cmo": {"alpha": 1.5, "metrics": v0},
            "source": str(rj),
        }
        cb_pape.append(float(v0["pape"]))
        cb_hr1.append(float(v0["hr@1"]))
        cb_hr2.append(float(v0["hr@2"]))
        cb_mae.append(float(v0["mae"]))
    if not per_seed:
        return None
    return {
        "cell": V5_FEDCB_0_CELL,
        "seeds_present": sorted(per_seed.keys()),
        "mode": "centralised_paper_anchor",
        "alpha": 1.5,
        "config": {
            "source": "v02 §B.3 W_component_results.json -> PAPE-aggressive.V0",
            "routing": "R0",
            "backbone": "v02 T2",
            "alpha_v0": 1.5,
            "alpha_w1": 0.0,
            "note": "Loaded directly from v02 outputs; not re-run by v05 driver.",
        },
        "per_seed": per_seed,
        "agg": {
            # No fl_only column for this anchor.
            "fl_only": {
                "pape": {"mean": float("nan"), "std": float("nan"), "n": 0},
                "hr@1": {"mean": float("nan"), "std": float("nan"), "n": 0},
                "hr@2": {"mean": float("nan"), "std": float("nan"), "n": 0},
                "mae": {"mean": float("nan"), "std": float("nan"), "n": 0},
            },
            "with_codebook_cmo": {
                "pape": _mean_std(cb_pape),
                "hr@1": _mean_std(cb_hr1),
                "hr@2": _mean_std(cb_hr2),
                "mae": _mean_std(cb_mae),
            },
        },
    }


def _gate_block(cells: dict[str, dict]) -> dict:
    """Compute Gate 1 / 2 / 3 pass/fail flags from the per-cell aggregates.

    Gate 1 — V5-FedCB-0 anchor cell ``v5_fedcb_0_paper_anchor``: mean PAPE
             within ±1.5 pp of 44.18.

             NOTE: Because this row is **loaded directly** from v02 §B.3's
             published numbers (not re-run), Gate 1 is *trivially true by
             construction* — it effectively becomes a check that the three
             v02 W_component_results.json files were found and parsed
             correctly. If one or more seeds is missing, ``status`` will say
             ``missing_or_partial``; otherwise the mean will sit at 44.18 and
             ``pass`` will be True. The original Gate 1 ("can we reproduce
             v02's centralised CMO under v05's pipeline?") is dropped because
             the revised plan §"Experimental matrix" specifies that
             V5-FedCB-0 is data-only.

    Gate 2 — V5-FedCB-1 (cell ``fedcb_K4``, α=1.0): mean PAPE ≤ 52 %.
             Unchanged.
    Gate 3 — at least one of (cells in V5-FedCB-{2a, 2b, 3}) clears 52 %.
             Unchanged.

    Each entry is ``{pass, mean_pape, threshold, ...}``; missing cells
    are marked ``status = "missing"``.
    """
    out: dict = {}

    # Gate 1 — v02 anchor presence + mean within ±1.5 pp of 44.18.
    g1 = {
        "threshold_low": GATE1_ANCHOR - GATE1_TOLERANCE,
        "threshold_high": GATE1_ANCHOR + GATE1_TOLERANCE,
        "anchor_pape": GATE1_ANCHOR,
        "note": (
            "V5-FedCB-0 is loaded from v02 §B.3 outputs directly, so this "
            "gate is true-by-construction once all three seeds parse."
        ),
    }
    if V5_FEDCB_0_CELL in cells:
        cm = cells[V5_FEDCB_0_CELL]["agg"]["with_codebook_cmo"]["pape"]
        n_seeds = cm.get("n", 0)
        if "mean" in cm and n_seeds == len(SEEDS):
            g1["mean_pape"] = cm["mean"]
            g1["pass"] = bool(g1["threshold_low"] <= cm["mean"] <= g1["threshold_high"])
            g1["status"] = "checked"
            g1["seeds_loaded"] = n_seeds
        elif "mean" in cm and n_seeds > 0:
            g1["mean_pape"] = cm["mean"]
            g1["pass"] = bool(g1["threshold_low"] <= cm["mean"] <= g1["threshold_high"])
            g1["status"] = "missing_or_partial"
            g1["seeds_loaded"] = n_seeds
        else:
            g1["status"] = "missing_or_partial"
            g1["seeds_loaded"] = 0
    else:
        g1["status"] = "missing"
        g1["seeds_loaded"] = 0
    out["gate_1_v02_anchor_load"] = g1

    # Gate 2 — fedcb_K4 (α=1.0) ≤ 52 %.
    g2 = {"threshold": GATE_2_3_THRESHOLD}
    if "fedcb_K4" in cells:
        cm = cells["fedcb_K4"]["agg"]["with_codebook_cmo"]["pape"]
        if "mean" in cm and not (cm["mean"] != cm["mean"]):  # not NaN
            g2["mean_pape"] = cm["mean"]
            g2["pass"] = bool(cm["mean"] <= GATE_2_3_THRESHOLD)
            g2["status"] = "checked"
        else:
            g2["status"] = "missing_or_partial"
    else:
        g2["status"] = "missing"
    out["gate_2_default_K4"] = g2

    # Gate 3 — at least one of {fedcb_K2, fedcb_K8, fedcb_K4_alpha*} clears the bar.
    g3_candidates = []
    for name, c in cells.items():
        if name in (V5_FEDCB_0_CELL, "fedcb_K4"):
            continue
        if not name.startswith("fedcb_"):
            continue
        cm = c["agg"]["with_codebook_cmo"]["pape"]
        if "mean" in cm and not (cm["mean"] != cm["mean"]):
            g3_candidates.append({
                "cell": name,
                "mean_pape": cm["mean"],
                "pass": bool(cm["mean"] <= GATE_2_3_THRESHOLD),
            })
    g3 = {
        "threshold": GATE_2_3_THRESHOLD,
        "candidates": g3_candidates,
        "any_pass": bool(any(c["pass"] for c in g3_candidates)),
        "status": "checked" if g3_candidates else "missing",
    }
    out["gate_3_K_alpha_sweep"] = g3
    return out


def main() -> None:
    cells = _discover_cells()
    per_cell: dict[str, dict] = {}
    for c in cells:
        per_cell[c] = _load_cell(c)

    # Inject the V5-FedCB-0 anchor row pulled from v02 outputs (data-only).
    anchor = _load_v02_anchor()
    if anchor is not None:
        per_cell[V5_FEDCB_0_CELL] = anchor
    else:
        print(
            f"[v05 aggregate] WARN: no v02 W_component_results.json found "
            f"under {V02_OUT_ROOT}; V5-FedCB-0 anchor row will be absent "
            f"and Gate 1 will be 'missing'."
        )

    if not per_cell:
        print(f"[v05 aggregate] no cells found under {V05_OUT_ROOT} and no "
              f"v02 anchor available; nothing to aggregate.")
        return

    summary = {
        "seeds_requested": SEEDS,
        "cells": per_cell,
        "gates": _gate_block(per_cell),
    }
    out_path = V05_OUT_ROOT / "multiseed_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[v05 aggregate] written -> {out_path}")

    # Console headline table (parallel to v04 _aggregate.py print).
    print()
    print(f"=== v05 multiseed summary (seeds {SEEDS}) ===\n")
    print(f"{'Cell':<32} {'fl_only PAPE':>12} {'+/-std':>8} "
          f"{'CMO PAPE':>12} {'+/-std':>8} {'HR@1':>8} {'+/-std':>8}")
    print("-" * 90)
    for name in sorted(per_cell.keys()):
        agg = per_cell[name]["agg"]
        fl = agg["fl_only"]["pape"]
        cb = agg["with_codebook_cmo"]["pape"]
        cb_hr1 = agg["with_codebook_cmo"]["hr@1"]
        if "mean" not in cb:
            continue
        # fl_only is "n/a" for the v02 anchor row.
        fl_mean_str = (
            f"{fl['mean']:>12.3f}" if ("mean" in fl and fl.get("n", 0) > 0)
            else f"{'n/a':>12}"
        )
        fl_std_str = (
            f"{fl.get('std', 0.0):>8.3f}" if ("mean" in fl and fl.get("n", 0) > 0)
            else f"{'n/a':>8}"
        )
        print(f"{name:<32} {fl_mean_str} {fl_std_str} "
              f"{cb['mean']:>12.3f} {cb.get('std', 0.0):>8.3f} "
              f"{cb_hr1.get('mean', float('nan')):>8.3f} "
              f"{cb_hr1.get('std', 0.0):>8.3f}")
    print()
    g = summary["gates"]
    print("Gates:")
    g1 = g["gate_1_v02_anchor_load"]
    print(f"  Gate 1 (V5-FedCB-0 anchor loaded from v02 B.3; trivially "
          f"~={GATE1_ANCHOR} +/- {GATE1_TOLERANCE}): {g1.get('status')}, "
          f"seeds_loaded={g1.get('seeds_loaded', 'n/a')}/{len(SEEDS)}, "
          f"mean_pape={g1.get('mean_pape', 'n/a')}, pass={g1.get('pass', 'n/a')}")
    g2 = g["gate_2_default_K4"]
    print(f"  Gate 2 (fedcb_K4 ≤ {GATE_2_3_THRESHOLD}): {g2.get('status')}, "
          f"mean_pape={g2.get('mean_pape', 'n/a')}, pass={g2.get('pass', 'n/a')}")
    g3 = g["gate_3_K_alpha_sweep"]
    print(f"  Gate 3 (any of K_local/α sweep ≤ {GATE_2_3_THRESHOLD}): "
          f"{g3.get('status')}, any_pass={g3.get('any_pass', 'n/a')}, "
          f"n_candidates={len(g3.get('candidates', []))}")


if __name__ == "__main__":
    main()
