"""Cold-side zero-shot inference under {R0, R1} routing × {HR-preserving, PAPE-aggressive}.

(한글 요약)
v02의 04번 — **메인 결과 산출 스크립트**. 한 seed에 대해 cold 가구의 zero-shot
추론을 수행한다. routing 분기(R0/R1) × operating point(HR-pres/PAPE-aggr)의
**2 × 2 = 4 cell** 평가에 baseline(보정 없음)을 더해 G1(80:20 PAPE 개선 생존)과
G2(R1 vs R0 routing)의 숫자를 모두 만든다. cold 측 학습은 일절 없으며(zero-shot,
FedHiP 프레이밍), per-apt z-norm은 cold 가구 자신의 train 구간(앞 70%)에서 추정한
warm-start 통계를 사용한다.

핵심 설계 결정:
    - **cold-side α 튜닝 금지** (plan §"Non-goals"의 v01 §5.4.1 selection bias 우려).
      operating points (σ, α_v0, α_w1)은 v01에서 그대로 carry-over 한다.
    - **routing은 backbone forward를 추가로 요구하지 않는다.**
      R0의 KEY는 입력 윈도우만으로 계산되며 (input-only, 5-d),
      R1의 ``h_g_cold``는 어차피 aux head 호출 때문에 forward에서 산출되므로
      두 routing 모두 추가 비용 0회이다 (plan §"Method").
    - cluster offset ``o_{c*}``는 03에서 z-norm space로 저장됐으므로 보정도 z-norm
      space에서 더하고 마지막에 denormalize하여 PAPE는 kW 단위로 보고한다.

멀티 seed sweep ({42, 123, 7})은 스크립트 안에 두지 않고 외부 launcher가
``--seed S``로 시드마다 한 번씩 호출한다 (memory: feedback_argparse_per_seed).

Inputs (per seed):
    outputs/v02_fl_8020_ratio/seed{S}/T2/best.pt       — frozen backbone + aux head
    outputs/v02_fl_8020_ratio/seed{S}/codebook.npz     — centroids, offsets, KEY pool, cluster_idx
    outputs/v02_fl_8020_ratio/splits/v02_8020_seed{S}.yaml  — train/cold apartments

For each cold apt:
    1. warm-start z-norm on its OWN first 70% (mirrors v01 cold protocol).
    2. sliding windows on the train-segment (stride=24, matching v01).
    3. frozen forward -> (y_hat_z, h_g, amp_pred, hr_pred_int, key).

Routing:
    R0 — KEY(x) -> StandardScaler (params from codebook.npz) -> 1-NN on
         train KEY pool -> cluster_idx of that train window.
         (v01과 동일한 routing. KEY는 입력만으로 계산 — backbone 의존성 없음.)
    R1 — argmin_c ||h_g - codebook[c]||_2 directly (×12 info, no extra fwd).
         (v02의 새로운 ablation. plan §"Open question 2": raw Euclidean이 default.)

Correction (W5 hybrid, both v01 operating points):
    g(t; h_hat, a_hat, sigma) = a_hat * exp(-(t - h_hat)^2 / (2*sigma^2))
                                normalised so g.max(axis=1) == a_hat.
    y_corr_z = y_hat_z + alpha_v0 * offsets[c*] + alpha_w1 * g

Outputs:
    outputs/v02_fl_8020_ratio/seed{S}/coldstart_R0.json
    outputs/v02_fl_8020_ratio/seed{S}/coldstart_R1.json
        — baseline + per-op-point metrics + aux diagnostics + cluster-usage stats.
        (plan §"Outputs" 트리: routing별 JSON 두 파일로 분리 저장.)
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
import torch

from config import OUTPUT_DIR, RANDOM_SEED
from dataloader.splits import load_v02_split
from eval.cold_helpers import (
    OPERATING_POINTS,
    gather_cold,
    gauss_template,
    metrics_z_to_kw,
    route_R0,
    route_R1,
)
from models.nbeatsx_aux import NBEATSxAux

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V02_OUT_ROOT = OUTPUT_DIR / "v02_fl_8020_ratio"


def run_routing(
    routing: str,
    co: dict,
    cb: dict,
) -> tuple[np.ndarray, dict]:
    """Returns (cold_cluster [N], routing_diag).

    (한글) routing 문자열 dispatch — R0(KEY 1-NN) / R1(centroid 직접). 두 경로 모두
    추가 backbone forward 없음. routing diagnostic으로 cluster usage 분포(사용된
    cluster 개수, min/max/mean usage)도 함께 만들어, 어느 한 쪽 routing이 sparse
    하게 collapse되는지(예: 한 두 cluster에 cold가 몰림)를 다음 단계 분석에서
    체크할 수 있게 한다.
    """
    if routing == "R0":
        # 03이 저장한 codebook.npz의 키들을 그대로 받아 R0 라우팅 호출.
        cold_cluster = route_R0(
            co["key"],
            cb["key_scaler_mean"],
            cb["key_scaler_scale"],
            cb["key_pool_scaled"],
            cb["cluster_idx"].astype(np.int64),
        )
    elif routing == "R1":
        # R1: cold h_g를 centroid 풀과 직접 비교 (raw Euclidean, plan default).
        cold_cluster = route_R1(co["h_g"], cb["codebook"])
    else:
        raise ValueError(f"unknown routing: {routing}")
    M = cb["codebook"].shape[0]
    # cluster usage histogram — sparse 라우팅(특정 cluster에 몰림) 진단용.
    usage_counts = np.bincount(cold_cluster, minlength=M)
    diag = {
        "n_clusters_used": int((usage_counts > 0).sum()),
        "usage_min": int(usage_counts.min()),
        "usage_max": int(usage_counts.max()),
        "usage_mean": float(usage_counts.mean()),
    }
    return cold_cluster, diag


def evaluate_routing(
    routing: str,
    co: dict,
    cb: dict,
) -> dict:
    """한 routing(R0 또는 R1)에 대해 baseline + 두 op-point 보정 결과를 모두 반환.

    (한글) **2 × 2 = 4 cell** 평가 구조의 한 행(row) — routing 하나 × op-point 둘.
    main()이 R0/R1에 대해 두 번 호출해 4 cell + baseline 2회분(각 routing마다 동일
    baseline)을 만든다.

    절차:
        1) ``run_routing``으로 cold cluster 배정 → ``offsets[cold_cluster]``로
           윈도우별 V0 보정량 ``o_{c*}`` (z-norm space)을 lookup.
        2) baseline metrics: 보정 없이 ŷ_base만 denorm 후 PAPE/HR/MAE 계산
           (G1 평가의 reference). plan §"Comparison table"의 "v01 50:50 baseline"
           대조 대상.
        3) 각 op-point에 대해 W5 hybrid 보정:
                ŷ_corr_z = ŷ_base_z + α_v0·o_{c*} + α_w1·g(t; ĥ, â, σ=3.0)
           — 보정은 z-norm space에서 더하고 마지막에 denorm (offsets/g 모두
             z-norm 단위이므로 일관). PAPE는 kW 단위로 보고.
        4) aux_diagnostics: aux head가 예측한 ĥ_int와 cold 실제 peak hour
           ``argmax(y_true_z)``의 일치율 (top-1 / ±1h / ±2h). 이는 W1a 부분이
           정확한 시각을 잡는지를 들여다보는 진단치이며 PAPE/HR 보고와는 별개.
    """
    cold_cluster, route_diag = run_routing(routing, co, cb)
    offsets = cb["offsets"]  # [M, 24] z-norm space residual offsets (03 산출).
    cluster_offset = offsets[cold_cluster]  # [N, 24] — 각 cold 윈도우의 V0 항.

    # ---- baseline (보정 없음) — 두 routing 다 동일하지만 routing별 JSON에 함께 저장 ----
    base = metrics_z_to_kw(co["y_true_z"], co["y_hat_z"], co["mean"], co["std"])

    # ---- 두 op-point 평가: (HR-preserving, PAPE-aggressive) ----
    op_results = {}
    for op_name, op in OPERATING_POINTS.items():
        # Gaussian template g(t; ĥ, â, σ) — z-norm space (â가 z-norm 단위이므로).
        g = gauss_template(co["pred_hr"], co["pred_amp"], sigma=op["sigma"])
        # W5 hybrid 보정: V0 + W1a를 z-norm space에서 합산 (이후 metric_z_to_kw에서 denorm).
        corrected_z = (
            co["y_hat_z"]
            + op["alpha_v0"] * cluster_offset
            + op["alpha_w1"] * g
        ).astype(np.float32)
        op_results[op_name] = {
            "sigma": op["sigma"],
            "alpha_v0": op["alpha_v0"],
            "alpha_w1": op["alpha_w1"],
            "metrics": metrics_z_to_kw(co["y_true_z"], corrected_z, co["mean"], co["std"]),
        }

    # aux head 진단: 예측 peak hour vs 실제 peak hour. W1a Gaussian center가 잘 잡히는지.
    cold_true_hr = co["y_true_z"].argmax(axis=1)
    aux_diag = {
        "top1": float((co["pred_hr"] == cold_true_hr).mean()),
        "within_1h": float((np.abs(co["pred_hr"] - cold_true_hr) <= 1).mean()),
        "within_2h": float((np.abs(co["pred_hr"] - cold_true_hr) <= 2).mean()),
    }
    return {
        "routing": routing,
        "n_cold_windows": int(co["y_true_z"].shape[0]),
        "n_cold_apts": int(len(np.unique(co["apt"]))),
        "baseline": base,                 # 보정 없음 — G1 reference
        "operating_points": op_results,   # HR-preserving / PAPE-aggressive 두 셀
        "routing_diagnostics": route_diag, # cluster usage 진단
        "aux_diagnostics": aux_diag,       # ĥ_int vs 실제 peak hour 일치율
    }


def main() -> None:
    """v02 04번 entrypoint — 한 seed에 대해 R0/R1 × 두 op-point 평가를 수행.

    (한글) per-seed argparse 컨벤션 (memory: feedback_argparse_per_seed) — 멀티시드
    sweep은 외부 launcher가 ``--seed 42 / 123 / 7``을 따로 호출. 스크립트 안에 시드
    루프를 넣지 않는다. 출력은 routing별로 JSON 분리(``coldstart_R0.json``,
    ``coldstart_R1.json``) — plan §"Outputs" 트리와 일치.
    """
    ap = argparse.ArgumentParser(description="Cold zero-shot evaluation: R0/R1 × HR-pres/PAPE-aggr.")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--routings", nargs="+", default=["R0", "R1"], choices=["R0", "R1"])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--stride", type=int, default=24)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # 02 (T2 backbone)와 03 (codebook) 산출이 모두 있어야 진행 가능 — 빠르면 즉시 fail.
    seed_root = V02_OUT_ROOT / f"seed{args.seed}"
    ckpt = seed_root / "T2" / "best.pt"
    cb_path = seed_root / "codebook.npz"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing {ckpt}; run 02_train_arms.py --seed {args.seed} --arms T2 first.")
    if not cb_path.exists():
        raise FileNotFoundError(f"missing {cb_path}; run 03_fit_codebook.py --seed {args.seed} first.")

    # 80:20 split의 cold 20 apts. (train 80은 03 codebook fit에서만 쓰이고 여기선 쓰지 않음.)
    cold_apts = load_v02_split(args.seed)["cold"]
    print(f"[setup] seed={args.seed}; {len(cold_apts)} cold apts; routings={args.routings}")
    print(f"[setup] device={DEVICE}; seed_root={seed_root}")

    # T2 backbone (NBEATSxAux + peak_aux head) frozen 로드 — eval 모드 / no_grad.
    # state_dict 키는 v10 b2 호환이므로 strict=True로 자동 로드 (load_state_dict default).
    model = NBEATSxAux(latent_source="h_generic").to(DEVICE).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))

    # 모든 cold 윈도우에 대해 frozen forward 1회 — h_g, ŷ_base, (â, ĥ), KEY 모두 수집.
    co = gather_cold(cold_apts, model, batch=args.batch, stride=args.stride)
    print(
        f"[data] {len(np.unique(co['apt']))} cold apts present, "
        f"{co['y_true_z'].shape[0]} cold windows"
    )

    # 03이 저장한 codebook 번들(centroids + offsets + KEY pool + scaler) 통째 로드.
    # np.savez는 dict-like한 NpzFile 반환 → 그대로 dict로 풀어 routing 함수에 넘김.
    cb_npz = np.load(cb_path)
    cb = {k: cb_npz[k] for k in cb_npz.files}

    for routing in args.routings:
        print(f"\n========== {routing} ==========")
        result = evaluate_routing(routing, co, cb)
        result["seed"] = int(args.seed)
        result["split_version"] = "v02"
        # routing별로 별도 JSON 파일로 저장 — plan §"Outputs"의 coldstart_R0.json /
        # coldstart_R1.json 트리와 정확히 일치. 07_aggregate_seeds.py가 이 두 파일을
        # 시드 × routing × op-point 차원으로 모아 multiseed_summary.json을 만든다.
        out_path = seed_root / f"coldstart_{routing}.json"
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
        base = result["baseline"]
        print(
            f"  baseline    PAPE={base['pape']:.2f}  HR@1={base['hr@1']:.1f}  "
            f"HR@2={base['hr@2']:.1f}  MAE={base['mae']:.4f}"
        )
        for op_name, op in result["operating_points"].items():
            ops_m = op["metrics"]
            ratio = ops_m["pape"] / base["pape"] if base["pape"] > 0 else float("nan")
            print(
                f"  {op_name:<16} PAPE={ops_m['pape']:.2f}  HR@1={ops_m['hr@1']:.1f}  "
                f"HR@2={ops_m['hr@2']:.1f}  MAE={ops_m['mae']:.4f}  (ratio={ratio:.3f})"
            )
        rd = result["routing_diagnostics"]
        print(
            f"  routing_diag  used={rd['n_clusters_used']}/32  "
            f"usage min/max={rd['usage_min']}/{rd['usage_max']}  mean={rd['usage_mean']:.1f}"
        )
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
