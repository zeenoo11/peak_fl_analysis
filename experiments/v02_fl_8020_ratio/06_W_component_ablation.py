"""W-component ablation @ 80:20 — V0 / W1a / W5 decomposition on T2.

Holds the backbone fixed (T2, with peak_aux) and the routing fixed (R0 KEY-NN);
varies only the cold-side correction mechanism. Reuses the codebook produced
by ``03_fit_codebook.py`` so V0 / W5 share the same offsets.

For each of the two v01 operating points (HR-preserving, PAPE-aggressive),
report PAPE / HR@k under three mechanism cells:

    V0-only   :  ŷ + α_v0 · o_{c*}                       (α_w1 = 0)
    W1a-only  :  ŷ + α_w1 · g(t; ĥ, â, σ)                (α_v0 = 0)
    W5-hybrid :  ŷ + α_v0 · o_{c*} + α_w1 · g(t; ĥ, â, σ)

Asks v01 §4.6 iter4's ranking question (W5 dominates V0 / W1a) at the v02
80:20 split. Orthogonal to E1 (which holds mechanism = V0 fixed and varies
the backbone).

Per-seed invocation:
    uv run python experiments/v02_fl_8020_ratio/06_W_component_ablation.py --seed 42

Output: outputs/v02_fl_8020_ratio/seed{S}/W_component_results.json

(한글 요약)
v02의 06번 — **W component decomposition (G4)**. backbone을 T2(peak_aux 포함)로
**고정**하고, routing도 R0(KEY 1-NN)으로 **고정**한 채, cold 측 보정(correction)
메커니즘만 {V0-only, W1a-only, W5-hybrid} 3-way로 토글한다. 두 v01 op-point
(HR-preserving / PAPE-aggressive)에 대해 각 메커니즘의 PAPE/HR@k를 보고.

핵심 질문 (plan §G4): v01 §4.6 iter4가 50:50 split에서 보였던 "W5 hybrid가
V0/W1a 단독보다 우월하다"는 랭킹이 v02의 80:20 split에서도 살아남는가?
W5 = V0 + W1a이므로 두 단독 결과의 max보다 W5가 더 좋다면 **시너지(synergy)**가
존재한다는 뜻 — 출력의 ``hybrid_synergy_kw = min(V0_PAPE, W1a_PAPE) - W5_PAPE``로
정의 (양수 → W5가 best single보다 PAPE를 더 깎음).

E1(05)과의 직교성:
    - **05 (E1)**  : mechanism = V0 고정, backbone 토글 (T0 ↔ T2). v01 §4.3
                     "+18.6 pp peak_aux 효과" 검증.
    - **06 (W comp)**: backbone = T2 고정, mechanism 토글 ({V0, W1a, W5}). v01
                     §4.6 iter4 "W5 dominance" 검증.
    두 ablation을 합치면 안 됨(어느 축이 변화의 원인인지 흐려짐). README의
    "What 05 vs 06 isolates" 박스 참조.

설계 메모 (cold split α 재튜닝 금지 — plan §"Non-goals"):
    - V0-only / W1a-only가 쓰는 α 값은 **W5 op-point의 값을 그대로 차용**.
      (V0-only는 α_v0 = op["alpha_v0"], W1a-only는 α_w1 = op["alpha_w1"].)
      cold split에서 단독 메커니즘에 대해 별도 α를 튜닝하면 v01 §5.4.1의
      selection bias가 다시 살아남.
    - σ=3.0은 두 op-point 모두에서 동일 (carry-over).

backbone = T2 only인 이유:
    W1a/W5는 aux head의 (â, ĥ) 출력이 Gaussian template 입력으로 필요하므로
    T0(no peak_aux) backbone으로는 정의 불가. → 06은 T2만 다룸.

routing = R0 only인 이유:
    plan §G4가 R0 한정으로 W5 ranking을 묻기 때문 (R0 vs R1 비교는 04에서 다룸).
    동일 codebook을 04와 공유하므로 동일 op-point의 W5 결과는 04와 비트 동일.

멀티 seed sweep ({42, 123, 7})은 스크립트 안에 두지 않고 외부 launcher가
``--seed S``로 시드마다 한 번씩 호출 (memory: feedback_argparse_per_seed).

Inputs (per seed):
    outputs/v02_fl_8020_ratio/seed{S}/T2/best.pt    — frozen T2 backbone + aux head
    outputs/v02_fl_8020_ratio/seed{S}/codebook.npz   — centroids + offsets + KEY pool + scaler
    outputs/v02_fl_8020_ratio/splits/v02_8020_seed{S}.yaml — train/cold apartments

Output:
    outputs/v02_fl_8020_ratio/seed{S}/W_component_results.json
        — baseline + 두 op-point × {V0, W1a, W5} 3-way + synergy + aux_diagnostics.
        plan §"Outputs" 트리와 일치. 07_aggregate_seeds.py가 시드별로 모아 평균/표준편차 산출.
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
)
from models.nbeatsx_aux import NBEATSxAux

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V02_OUT_ROOT = OUTPUT_DIR / "v02_fl_8020_ratio"


def main() -> None:
    """v02 06번 entrypoint — 한 seed에 대해 T2 × {V0, W1a, W5} 3-way 평가 (R0 routing, 두 op-point).

    (한글) per-seed argparse 컨벤션 (memory: feedback_argparse_per_seed) — 멀티시드
    sweep은 외부 launcher가 ``--seed 42 / 123 / 7``을 따로 호출. 스크립트 안에
    시드 루프 두지 않음.
    """
    ap = argparse.ArgumentParser(description="W-component ablation: V0 / W1a / W5 decomposition (per-seed).")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--stride", type=int, default=24)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ---- 02 (T2 backbone) + 03 (codebook) 의존성 체크 — 빠르면 즉시 fail ----
    seed_root = V02_OUT_ROOT / f"seed{args.seed}"
    ckpt = seed_root / "T2" / "best.pt"
    cb_path = seed_root / "codebook.npz"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing {ckpt}; run 02_train_arms.py --seed {args.seed} --arms T2 first.")
    if not cb_path.exists():
        raise FileNotFoundError(f"missing {cb_path}; run 03_fit_codebook.py --seed {args.seed} first.")

    # 80:20 split의 cold 20 apts. (train 80은 03 codebook fit에서만 쓰이고 06에선 미사용.)
    cold_apts = load_v02_split(args.seed)["cold"]
    print(f"[setup] seed={args.seed}  cold={len(cold_apts)}  routing=R0 (KEY-NN)")
    print(f"[setup] device={DEVICE}  seed_root={seed_root}")

    # ---- T2 backbone (NBEATSxAux + peak_aux head) frozen 로드 ----
    # state_dict 키는 v10 b2 호환이므로 strict=True (load_state_dict default)로 자동 로드.
    # backbone = T2 only — W1a/W5는 aux head의 (â, ĥ)가 필요하므로 T0(no peak_aux) 불가.
    model = NBEATSxAux(latent_source="h_generic").to(DEVICE).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))

    # 모든 cold 윈도우에 대해 frozen forward 1회 — ŷ_base, h_g, (â, ĥ), KEY 모두 수집.
    co = gather_cold(cold_apts, model, batch=args.batch, stride=args.stride)
    print(f"[data] {co['y_hat_z'].shape[0]} cold windows")

    # ---- 03이 저장한 codebook 번들 (centroids + offsets + KEY pool + scaler) 로드 ----
    # 04와 동일한 codebook을 그대로 사용 → 같은 op-point의 W5 결과는 04와 비트 동일해야 함.
    cb_npz = np.load(cb_path)
    cb = {k: cb_npz[k] for k in cb_npz.files}
    # R0 routing only (plan §G4) — cold 윈도우 → cluster index 부여.
    cold_cluster = route_R0(
        co["key"],
        cb["key_scaler_mean"],
        cb["key_scaler_scale"],
        cb["key_pool_scaled"],
        cb["cluster_idx"].astype(np.int64),
    )
    # 각 cold 윈도우에 해당 cluster의 residual offset o_{c*}를 lookup (z-norm space).
    cluster_offset = cb["offsets"][cold_cluster]  # [N, 24] z-norm

    # ---- baseline: 보정 없음 (ŷ_base만 denorm 후 metric 계산) — 시너지 비교의 reference ----
    base = metrics_z_to_kw(co["y_true_z"], co["y_hat_z"], co["mean"], co["std"])
    print(
        f"\n  baseline (no correction)      PAPE={base['pape']:.2f}  "
        f"HR@1={base['hr@1']:.1f}  HR@2={base['hr@2']:.1f}  MAE={base['mae']:.4f}"
    )

    # ---- 두 op-point × 세 메커니즘 = 6 cell 평가 ----
    out_per_op = {}
    for op_name, op in OPERATING_POINTS.items():
        sigma, av, aw = op["sigma"], op["alpha_v0"], op["alpha_w1"]
        # Gaussian template g(t; ĥ, â, σ) — z-norm space (â가 z-norm 단위).
        # σ는 op-point 무관 3.0 carry-over (재튜닝 금지).
        g = gauss_template(co["pred_hr"], co["pred_amp"], sigma=sigma)

        # 세 메커니즘 — V0-only / W1a-only / W5-hybrid. 모두 동일 op-point의 동일 α 값을 사용.
        # **핵심 설계**: V0-only / W1a-only가 W5와 별도 α를 쓰지 않음. cold split α 재튜닝 금지.
        v0_z = co["y_hat_z"] + av * cluster_offset                  # V0  : ŷ + α_v0·o_{c*}
        w1a_z = co["y_hat_z"] + aw * g                              # W1a : ŷ + α_w1·g(t; ĥ, â, σ)
        w5_z = co["y_hat_z"] + av * cluster_offset + aw * g         # W5  : V0 + W1a (additive hybrid)

        cells = {
            "V0": metrics_z_to_kw(co["y_true_z"], v0_z, co["mean"], co["std"]),
            "W1a": metrics_z_to_kw(co["y_true_z"], w1a_z, co["mean"], co["std"]),
            "W5": metrics_z_to_kw(co["y_true_z"], w5_z, co["mean"], co["std"]),
        }
        # ---- Synergy = best single component PAPE - W5 PAPE ----
        # PAPE는 낮을수록 좋으므로 best_single = min(V0, W1a). synergy>0 ⇔ W5가 두 단독의
        # 최고치보다 PAPE를 더 깎음 → "단순 합산이 아니라 진짜 hybrid 효과"로 해석.
        # README seed=42 결과 "+3.02 / +2.70 PAPE-kW"가 이 정의에서 나온 값.
        pape_v0 = cells["V0"]["pape"]
        pape_w1a = cells["W1a"]["pape"]
        pape_w5 = cells["W5"]["pape"]
        best_single = min(pape_v0, pape_w1a)
        synergy = best_single - pape_w5  # >0 => W5 beats best single component
        out_per_op[op_name] = {
            "sigma": sigma,
            "alpha_v0": av,
            "alpha_w1": aw,
            "cells": cells,
            "best_single_pape": best_single,
            "w5_pape": pape_w5,
            "hybrid_synergy_kw": synergy,
        }

        print(f"\n  --- {op_name} (σ={sigma}, α_v0={av}, α_w1={aw}) ---")
        for mech, m in cells.items():
            ratio = m["pape"] / base["pape"] if base["pape"] > 0 else float("nan")
            print(
                f"    {mech:5s} PAPE={m['pape']:.2f}  HR@1={m['hr@1']:.1f}  "
                f"HR@2={m['hr@2']:.1f}  MAE={m['mae']:.4f}  (ratio={ratio:.3f})"
            )
        print(f"    synergy (best_single - W5 in PAPE kW) = {synergy:+.2f}")

    # ---- aux head 진단: 예측 peak hour vs 실제 peak hour ----
    # W1a Gaussian center가 잘 잡히는지 들여다보는 진단치 (PAPE/HR 보고와는 별개).
    cold_true_hr = co["y_true_z"].argmax(axis=1)
    aux_diag = {
        "top1": float((co["pred_hr"] == cold_true_hr).mean()),
        "within_1h": float((np.abs(co["pred_hr"] - cold_true_hr) <= 1).mean()),
        "within_2h": float((np.abs(co["pred_hr"] - cold_true_hr) <= 2).mean()),
    }
    # ---- 시드별 결과 JSON 저장 (07_aggregate_seeds.py가 시드 모아 평균/표준편차 산출) ----
    out = {
        "seed": int(args.seed),
        "split_version": "v02",
        "routing": "R0",
        "n_cold_windows": int(co["y_hat_z"].shape[0]),
        "n_cold_apts": len(cold_apts),
        "baseline": base,
        "per_operating_point": out_per_op,
        "aux_diagnostics": aux_diag,
        "comment": (
            "T2 × {V0-only, W1a-only, W5-hybrid} on R0 routing; orthogonal to E1, "
            "asks the v01 §4.6 iter4 W5-dominance question at 80:20."
        ),
    }
    out_path = seed_root / "W_component_results.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  saved -> {out_path}")


if __name__ == "__main__":
    main()
