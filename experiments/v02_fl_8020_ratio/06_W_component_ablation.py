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
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from config import OUTPUT_DIR, RANDOM_SEED, TRAIN_RATIO
from dataloader.splits import load_v02_split
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from models.nbeatsx_aux import NBEATSxAux
from probes.peak_descriptor import extract_key
from utils.metrics import compute_hr, compute_mae, compute_pape

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V02_OUT_ROOT = OUTPUT_DIR / "v02_fl_8020_ratio"

# v01 carry-over 두 op-point. 04의 OPERATING_POINTS와 비트 동일하게 유지해야
# "06의 W5 결과 == 04의 W5 결과 (R0 routing)"가 성립. cold split에서 (σ, α_v0, α_w1)
# 재튜닝은 plan §"Non-goals"에서 명시적으로 금지 — v01 §5.4.1 selection bias.
#
# V0-only는 α_v0만 살리고 (α_w1=0), W1a-only는 α_w1만 살리고 (α_v0=0), W5는 둘 다 사용.
# **세 메커니즘 모두 동일 op-point의 동일 α 값을 사용** → 단독 vs hybrid의 시너지를
# "α 차이"가 아니라 "두 항의 합산 효과" 자체로 측정.
OPERATING_POINTS = {
    "HR-preserving": {"sigma": 3.0, "alpha_v0": 1.0, "alpha_w1": 0.1},
    "PAPE-aggressive": {"sigma": 3.0, "alpha_v0": 1.5, "alpha_w1": 0.5},
}


def gather_cold(
    apts: list[str],
    model: NBEATSxAux,
    batch: int = 256,
    stride: int = 24,
) -> dict[str, np.ndarray]:
    """Same protocol as 04_coldstart_eval.py:gather_cold (warm-start z-norm, stride=24).

    (한글) 04의 ``gather_cold``를 그대로 복제한 함수 — cold apt 각각에 대해
    자기 시계열 앞 70% (``TRAIN_RATIO``)에서 추정한 per-apt z-norm 통계로
    ``stride=24`` 슬라이딩 윈도우를 만들고 frozen T2를 한 번 통과시켜 다음을 수집:

        - ``h_g``     : ``h_generic`` ∈ ℝ^{N×64} — 본 06번에선 직접 쓰진 않으나
                        04와 동일 시그니처 유지를 위해 수집.
        - ``y_hat_z`` : z-norm space ŷ_base.
        - ``y_true_z``: ground truth (z-norm). denorm 후 PAPE/HR 계산에 사용.
        - ``pred_amp``, ``pred_hr`` : aux head (â, ĥ_int) — W1a/W5 Gaussian 입력.
        - ``key``     : 5-d KEY — R0 routing 입력.
        - ``mean``, ``std`` : per-window denorm 통계.

    설계 메모:
        - **04와 동일 프로토콜**: warm-start z-norm + stride=24 + frozen forward.
          따라서 같은 codebook + 같은 op-point에서의 W5 결과는 04의 R0 W5와 비트 동일.
        - cold-side 학습 없음 (``model.eval()`` + ``torch.no_grad()``) — FedHiP 프레이밍.
        - 04의 ``gather_cold``는 추가로 ``apt`` (윈도우 출처) 배열을 수집하지만
          06은 cold 윈도우 단위 평균만 사용하므로 생략.

    NOTE (engineer): 04의 헬퍼와 **import 공유 없이 복붙**된 상태.
    drift 위험 → src로 빼는 리팩터링은 별도 작업.
    """
    h_chunks, yhat_chunks, ytrue_chunks = [], [], []
    amp_chunks, hr_chunks, key_chunks = [], [], []
    mean_chunks, std_chunks = [], []
    for apt in apts:
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            continue
        n = len(series)
        train_end = int(n * TRAIN_RATIO)
        seg = series[:train_end]
        # per-apt z-norm: cold 자신의 train segment에서만 추정 (cold future label 미사용).
        m_ = float(seg.mean())
        s_ = float(seg.std()) if seg.std() > 1e-8 else 1.0
        # stride=24: 비중첩 — 04 / 03 codebook fit과 동일.
        ds = HouseholdDataset(seg, m_, s_, stride=stride)
        if len(ds) == 0:
            continue
        for x, y in DataLoader(ds, batch_size=batch, shuffle=False):
            x_dev = x.to(DEVICE)
            with torch.no_grad():
                # T2 forward: (ŷ_base_z, hiddens, (â, ĥ_logits)). aux head 출력 (â, ĥ)는
                # 06의 W1a/W5 Gaussian template center/amplitude로 직결.
                y_hat, hiddens, (amp_p, hr_p) = model(x_dev)
            h_chunks.append(hiddens["h_generic"].cpu().numpy())
            yhat_chunks.append(y_hat.cpu().numpy())
            ytrue_chunks.append(y.numpy())
            amp_chunks.append(amp_p.cpu().numpy().reshape(-1))
            # hr_pred: 24-class CE logits → argmax로 정수 시각 ĥ_int (Gaussian center).
            hr_chunks.append(hr_p.argmax(dim=1).cpu().numpy())
            # KEY는 입력 x로부터 직접 계산 — backbone 호출 불필요 (input-only 5-d 디스크립터).
            key_chunks.append(extract_key(x.numpy()))
            mean_chunks.append(np.full(len(y), m_, dtype=np.float32))
            std_chunks.append(np.full(len(y), s_, dtype=np.float32))
    return {
        "h_g": np.concatenate(h_chunks, axis=0).astype(np.float32),
        "y_hat_z": np.concatenate(yhat_chunks, axis=0).astype(np.float32),
        "y_true_z": np.concatenate(ytrue_chunks, axis=0).astype(np.float32),
        "pred_amp": np.concatenate(amp_chunks, axis=0).astype(np.float32),
        "pred_hr": np.concatenate(hr_chunks, axis=0).astype(np.int64),
        "key": np.concatenate(key_chunks, axis=0).astype(np.float32),
        "mean": np.concatenate(mean_chunks, axis=0),
        "std": np.concatenate(std_chunks, axis=0),
    }


def gauss_template(pred_hr: np.ndarray, pred_amp: np.ndarray, sigma: float, length: int = 24) -> np.ndarray:
    """W1a/W5의 Gaussian template ``g(t; ĥ, â, σ) = â · exp(-(t-ĥ)²/2σ²)``.

    (한글) max-normalize 후 amplitude 곱셈으로 ``g.max(axis=1) == pred_amp`` 보장
    (W family 규약 — v01 §iter4와 비트 정확). σ는 op-point 무관 3.0 carry-over.

    NOTE: 04와 동일 구현 (복붙). v01 09_iter4_mechanisms.py의 ``gauss_template``과는
    σ default만 다르고 (v01: 1.5, v02: 3.0 op-point 값) 계산 식은 동일.
    """
    # t shape: (1, length=24). pred_hr는 정수 시각 (aux head argmax).
    t = np.arange(length, dtype=np.float32)[None, :]
    # 표준 가우시안 (broadcast: B × length).
    g = np.exp(-0.5 * ((t - pred_hr.astype(np.float32)[:, None]) / sigma) ** 2)
    # max-normalize → 곱한 후 g.max == pred_amp 보장.
    g = g / g.max(axis=1, keepdims=True)
    return (g * pred_amp[:, None]).astype(np.float32)


def metrics_z_to_kw(true_z, pred_z, mean_arr, std_arr) -> dict:
    """z-norm space → kW 단위로 denorm 후 PAPE/HR@1/HR@2/MAE 계산.

    (한글) plan §"Metrics" — PAPE는 kW 기준 (v01 §4.1과 비트 정확).
    ``compute_pape``/``compute_hr``는 ``Peak_Analysis``로부터의 비트 정확 포팅이며
    수정 금지 (CLAUDE.md). 04의 동명 함수와 비트 동일.
    """
    # per-window broadcasting: (N, H) * (N, 1) + (N, 1).
    true_kw = true_z * std_arr[:, None] + mean_arr[:, None]
    pred_kw = pred_z * std_arr[:, None] + mean_arr[:, None]
    return {
        "pape": float(compute_pape(true_kw, pred_kw)),
        "hr@1": float(compute_hr(true_kw, pred_kw, tol=1)),
        "hr@2": float(compute_hr(true_kw, pred_kw, tol=2)),
        "mae": float(compute_mae(true_kw, pred_kw)),
    }


def route_R0(co_key, key_scaler_mean, key_scaler_scale, key_pool_scaled, train_cluster_idx) -> np.ndarray:
    """v01과 동일한 R0 routing (KEY 1-NN).

    (한글)
        1) cold 5-d KEY를 03이 fit/저장한 StandardScaler 파라미터로 정규화 (cold 측
           재fit 금지 — fair zero-shot).
        2) scaler-적용 train KEY 풀에서 1-NN 검색.
        3) 그 이웃 train 윈도우의 ``cluster_idx``를 cold 윈도우의 cluster로 채택.
    KEY는 input-only이므로 backbone forward 0회. 04의 ``route_R0``와 비트 동일.
    """
    # cold KEY를 train 시점 scaler로 정규화 (재fit 금지).
    co_key_scaled = (co_key - key_scaler_mean) / key_scaler_scale
    nn = NearestNeighbors(n_neighbors=1).fit(key_pool_scaled)
    _, neigh_idx = nn.kneighbors(co_key_scaled)
    # 가장 가까운 train 윈도우의 cluster index를 빌려와 cold 윈도우에 부여.
    return train_cluster_idx[neigh_idx[:, 0]]


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
