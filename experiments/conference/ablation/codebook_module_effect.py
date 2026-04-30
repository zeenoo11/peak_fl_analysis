"""Codebook Correction Module 효과 측정 — multi-seed aggregator.

(한글 요약)
KIIE conference 발표 (``papers/conference_draft/presentation.md``)의 §"Codebook
Correction Module 효과 측정" 표 (lines 211-218)를 그대로 재구성하는 aggregator
스크립트:

    | Method                                | PAPE (%)     | HR@1 (%)     | HR@2 (%)     | MSE (kW²)       |
    | ------------------------------------- | ------------ | ------------ | ------------ | --------------- |
    | Backbone (no correction)              | 57.32 ± 1.55 | 26.35 ± 1.67 | 37.76 ± 1.56 | 0.5300 ± 0.0314 |
    | Backbone + Codebook Correction Module | 50.17 ± 0.97 | 25.28 ± 1.30 | 37.24 ± 1.86 | 0.5060 ± 0.0326 |

각 seed에 대해 ``outputs/conference/pipeline/seed{S}/phase_c/{result.json,
cold_arrays.npz}``에서 raw cold 배열을 읽어:
    - ``Backbone (no correction)`` 행: ``y_hat_z``로부터 PAPE/HR@1/HR@2/MSE 계산.
    - ``Backbone + Codebook Correction Module`` 행: ``corrected_z``로부터 동일 4지표
      계산.

PAPE / HR@1 / HR@2는 ``eval.cold_helpers.metrics_z_to_kw``를 그대로 호출 (PAPE는
peak-amplitude % 오차, HR@k는 hit rate; ``utils.metrics``의 bit-exact port).
MSE in kW²는 본 스크립트에서 직접 계산: ``y_true_z`` / ``y_pred_z``를 per-window
``mean`` / ``std``로 de-z-norm 후 ``((y_true_kw - y_pred_kw) ** 2).mean()`` —
``experiments/v05_fedcb_codebook/04_recompute_mse.py``의 공식과 일치.

3-seed 집계는 ``mean ± std (ddof=1)`` (v04 ``07_aggregate.py`` 컨벤션 일치;
``np.std(arr, ddof=1)``).

Per-seed argparse vs aggregator distinction
-------------------------------------------
``feedback_argparse_per_seed`` 메모리는 *training/eval* 드라이버에 적용된다 — 즉
Phase A / B / C 같이 backbone을 학습/추론하는 스크립트는 ``--seed S``로 외부에서
세 번 호출된다. 본 스크립트는 *aggregator* — 이미 디스크에 있는 artefact를 읽기만
하므로 multi-seed argparse (``--seeds 42 123 7``)를 받아도 컨벤션에 어긋나지 않는다.

Output:
    - stdout: 발표 표와 동일 모양의 markdown table (paper에 그대로 paste 가능).
    - JSON: ``outputs/conference/ablation/codebook_module_effect.json`` —
      per-seed numbers + aggregates + source paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from config import OUTPUT_DIR
from eval.cold_helpers import metrics_z_to_kw

CONFERENCE_PIPELINE_ROOT = OUTPUT_DIR / "conference" / "pipeline"
CONFERENCE_ABLATION_ROOT = OUTPUT_DIR / "conference" / "ablation"


def _kw_mse(y_true_z: np.ndarray, y_pred_z: np.ndarray,
            mean_arr: np.ndarray, std_arr: np.ndarray) -> float:
    """MSE in kW² — de-z-norm and average. Matches v05 04_recompute_mse exactly."""
    t_kw = y_true_z * std_arr[:, None] + mean_arr[:, None]
    p_kw = y_pred_z * std_arr[:, None] + mean_arr[:, None]
    return float(((t_kw - p_kw) ** 2).mean())


def _per_seed_metrics(seed: int) -> dict:
    """Compute the 4 metrics × 2 rows for one seed.

    Returns
    -------
    dict shaped:
        {
            "result_json_path": str,
            "cold_arrays_path": str,
            "n_cold_windows": int,
            "n_cold_apts": int,
            "rows": {
                "backbone_no_correction":      {pape, hr@1, hr@2, mse_kw2},
                "backbone_plus_codebook_cmo":  {pape, hr@1, hr@2, mse_kw2},
            }
        }
    """
    seed_dir = CONFERENCE_PIPELINE_ROOT / f"seed{seed}" / "phase_c"
    rj = seed_dir / "result.json"
    arrs = seed_dir / "cold_arrays.npz"
    if not rj.exists():
        raise FileNotFoundError(
            f"Conference Phase C result.json missing for seed={seed}: {rj}. "
            f"Run experiments/conference/pipeline/03_phase_c_cmo_inference.py "
            f"--seed {seed}."
        )
    if not arrs.exists():
        raise FileNotFoundError(
            f"Conference Phase C cold_arrays.npz missing for seed={seed}: {arrs}. "
            f"Run experiments/conference/pipeline/03_phase_c_cmo_inference.py "
            f"--seed {seed}."
        )

    z = np.load(arrs)
    y_true_z = z["y_true_z"]
    y_hat_z = z["y_hat_z"]
    corrected_z = z["corrected_z"]
    mean_arr = z["mean"]
    std_arr = z["std"]

    # Backbone (no correction) row.
    bb = metrics_z_to_kw(y_true_z, y_hat_z, mean_arr, std_arr)
    bb["mse_kw2"] = _kw_mse(y_true_z, y_hat_z, mean_arr, std_arr)

    # Backbone + Codebook (CMO) row.
    cm = metrics_z_to_kw(y_true_z, corrected_z, mean_arr, std_arr)
    cm["mse_kw2"] = _kw_mse(y_true_z, corrected_z, mean_arr, std_arr)

    return {
        "result_json_path": str(rj),
        "cold_arrays_path": str(arrs),
        "n_cold_windows": int(y_true_z.shape[0]),
        "n_cold_apts": int(len(np.unique(z["apt"]))),
        "rows": {
            "backbone_no_correction": {
                "pape": float(bb["pape"]),
                "hr@1": float(bb["hr@1"]),
                "hr@2": float(bb["hr@2"]),
                "mse_kw2": float(bb["mse_kw2"]),
            },
            "backbone_plus_codebook_cmo": {
                "pape": float(cm["pape"]),
                "hr@1": float(cm["hr@1"]),
                "hr@2": float(cm["hr@2"]),
                "mse_kw2": float(cm["mse_kw2"]),
            },
        },
    }


def _aggregate(values: list[float]) -> dict:
    """3-seed mean ± std (ddof=1). Matches v04 07_aggregate.py convention."""
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if n > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(n),
        "values": [float(v) for v in arr],
    }


def _format_pm(agg: dict, decimals: int) -> str:
    return f"{agg['mean']:.{decimals}f} ± {agg['std']:.{decimals}f}"


def _print_markdown(agg_rows: dict[str, dict[str, dict]]) -> None:
    """Print the §'Codebook Correction Module 효과 측정' table.

    Shape mirrors presentation.md lines 213-216 exactly so the output can be
    pasted straight into the paper.
    """
    bb = agg_rows["backbone_no_correction"]
    cm = agg_rows["backbone_plus_codebook_cmo"]
    print()
    print("| Method                                | PAPE (%)     | HR@1 (%)     | HR@2 (%)     | MSE (kW²)       |")
    print("| ------------------------------------- | ------------ | ------------ | ------------ | --------------- |")
    print(
        f"| Backbone (no correction)              | "
        f"{_format_pm(bb['pape'], 2)} | "
        f"{_format_pm(bb['hr@1'], 2)} | "
        f"{_format_pm(bb['hr@2'], 2)} | "
        f"{_format_pm(bb['mse_kw2'], 4)} |"
    )
    print(
        f"| Backbone + Codebook Correction Module | "
        f"{_format_pm(cm['pape'], 2)} | "
        f"{_format_pm(cm['hr@1'], 2)} | "
        f"{_format_pm(cm['hr@2'], 2)} | "
        f"{_format_pm(cm['mse_kw2'], 4)} |"
    )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Aggregate Phase C outputs across seeds for the §'Codebook "
            "Correction Module 효과 측정' table in presentation.md. "
            "This is an aggregator over existing artefacts, not a "
            "training/eval driver — multi-seed argparse is fine."
        )
    )
    ap.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 7],
        help="Seeds to aggregate (default: 42 123 7).",
    )
    ap.add_argument(
        "--alpha", type=float, default=1.0,
        help=(
            "α used in Phase C; declared here only for the result JSON's "
            "provenance (the actual α is whatever Phase C wrote). "
            "If the per-seed Phase C result.json shows a different α, the "
            "summary keeps the per-seed values verbatim."
        ),
    )
    ap.add_argument(
        "--out", type=Path,
        default=CONFERENCE_ABLATION_ROOT / "codebook_module_effect.json",
    )
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    per_seed: dict[str, dict] = {}
    for s in args.seeds:
        per_seed[str(s)] = _per_seed_metrics(s)

    # Aggregate per row × per metric across seeds.
    rows_keys = ("backbone_no_correction", "backbone_plus_codebook_cmo")
    metric_keys = ("pape", "hr@1", "hr@2", "mse_kw2")
    agg_rows: dict[str, dict[str, dict]] = {row: {} for row in rows_keys}
    for row in rows_keys:
        for mk in metric_keys:
            vals = [per_seed[str(s)]["rows"][row][mk] for s in args.seeds]
            agg_rows[row][mk] = _aggregate(vals)

    _print_markdown(agg_rows)

    payload = {
        "seeds": list(args.seeds),
        "declared_alpha": float(args.alpha),
        "ddof_for_std": 1,
        "comment": (
            "Aggregator for presentation.md §'Codebook Correction Module 효과 "
            "측정' (lines 211-218). PAPE / HR@1 / HR@2 from "
            "eval.cold_helpers.metrics_z_to_kw on the saved cold arrays; MSE "
            "in kW² recomputed here by de-z-norming with per-window mean/std "
            "(matches experiments/v05_fedcb_codebook/04_recompute_mse.py). "
            "Aggregation = mean ± sample-std (ddof=1), matching v04 "
            "07_aggregate.py convention."
        ),
        "per_seed": per_seed,
        "aggregate": agg_rows,
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[ablation] saved -> {args.out}")


if __name__ == "__main__":
    main()
