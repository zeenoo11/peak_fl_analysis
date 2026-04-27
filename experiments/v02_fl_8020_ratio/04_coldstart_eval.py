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

# v01 carry-over operating points (plans/v02-01_fl_8020_ratio.md "Non-goals").
# 두 op-point 모두 σ=3.0 고정. cold split에서 (σ, α_v0, α_w1) 재튜닝은 명시적으로
# 금지 — 그렇게 하면 v01 §5.4.1 selection bias가 v02에서 다시 살아남.
#   - HR-preserving : 약한 보정. HR@k를 baseline 근방으로 유지.
#   - PAPE-aggressive: 강한 보정. HR@k를 약간 양보하고 PAPE를 더 깎음.
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
    """Forward pass on every cold apt's train-segment (warm-start z-norm).

    (한글) 모든 cold apt에 대해 자기 시계열 앞 70%(``TRAIN_RATIO=0.7``)에서 얻은
    per-apt z-norm 통계를 사용해 ``stride=24`` 슬라이딩 윈도우를 만들고,
    frozen T2 backbone에 한 번 통과시켜 다음을 수집한다:

        - ``h_g``     : ``h_generic`` ∈ ℝ^{N×64} — R1 routing 입력 (centroid 거리).
        - ``y_hat_z`` : z-norm space 베이스 예측 ŷ_base.
        - ``y_true_z``: ground truth (z-norm space) — denorm 후 PAPE/HR 계산에 사용.
        - ``pred_amp``, ``pred_hr`` : aux head 출력 (â, ĥ_int) — W5 Gaussian template 입력.
        - ``key``     : 5-d KEY [max, argmax/96, mean, std, last24_max] — R0 routing.
        - ``mean``, ``std`` : 윈도우별 denorm 통계 (per-apt이므로 한 apt 내 동일값 broadcast).
        - ``apt``     : 윈도우 출처 apt 이름 — diagnostic / cold-apt 카운트.

    설계 메모:
        - **cold side에서 학습 없음**: ``model.eval()`` + ``torch.no_grad()``.
          backbone은 02에서 train된 frozen 상태 그대로 (FedHiP 프레이밍).
        - **z-norm 통계는 cold 자신의 train segment에서만 추정** (warm-start z-norm,
          v01 §cold protocol과 동일). cold의 future label은 통계 추정에 사용 X.
        - **stride=24** (= horizon, non-overlapping)는 v01 cold-start와 03 codebook fit
          과 동일하게 맞춰 정합성 유지.
        - aux head 출력 ``hr_p``는 24-class logits → ``argmax``로 정수 시각 ĥ_int 추출.
          ``amp_p``는 scalar amplitude â (둘 다 z-norm space 값).
    """
    h_chunks, yhat_chunks, ytrue_chunks = [], [], []
    amp_chunks, hr_chunks, key_chunks = [], [], []
    mean_chunks, std_chunks, apt_chunks = [], [], []
    for apt in apts:
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            # 데이터가 빠진 apt는 건너뛰되, 사용자가 인지할 수 있게 출력.
            print(f"  [skip] {apt}: missing")
            continue
        n = len(series)
        train_end = int(n * TRAIN_RATIO)
        seg = series[:train_end]
        # per-apt z-norm: cold의 학습 구간 통계만 사용 (std<1e-8 → 1.0 fallback).
        m_ = float(seg.mean())
        s_ = float(seg.std()) if seg.std() > 1e-8 else 1.0
        # stride=24 → 윈도우 비중첩. v01 cold-start / 03 codebook fit과 동일.
        ds = HouseholdDataset(seg, m_, s_, stride=stride)
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=batch, shuffle=False)
        for x, y in loader:
            x_dev = x.to(DEVICE)
            with torch.no_grad():
                # T2 forward 1회: (ŷ_base_z, hiddens, (â, ĥ_logits)).
                # — h_generic은 R1 routing에, aux 출력은 W5 Gaussian template에 재사용.
                # — 따라서 R0/R1 두 routing 모두 추가 backbone 호출 0회.
                y_hat, hiddens, (amp_p, hr_p) = model(x_dev)
            h_chunks.append(hiddens["h_generic"].cpu().numpy())
            yhat_chunks.append(y_hat.cpu().numpy())
            ytrue_chunks.append(y.numpy())
            amp_chunks.append(amp_p.cpu().numpy().reshape(-1))
            # hr_pred는 24-class CE logits → argmax로 정수 시각 (Gaussian center ĥ).
            hr_chunks.append(hr_p.argmax(dim=1).cpu().numpy())
            # KEY는 입력 x로부터 직접 계산 — backbone 호출 불필요 (input-only 5-d 디스크립터).
            key_chunks.append(extract_key(x.numpy()))
            mean_chunks.append(np.full(len(y), m_, dtype=np.float32))
            std_chunks.append(np.full(len(y), s_, dtype=np.float32))
            apt_chunks.append(np.array([apt] * len(y)))
    return {
        "h_g": np.concatenate(h_chunks, axis=0).astype(np.float32),
        "y_hat_z": np.concatenate(yhat_chunks, axis=0).astype(np.float32),
        "y_true_z": np.concatenate(ytrue_chunks, axis=0).astype(np.float32),
        "pred_amp": np.concatenate(amp_chunks, axis=0).astype(np.float32),
        "pred_hr": np.concatenate(hr_chunks, axis=0).astype(np.int64),
        "key": np.concatenate(key_chunks, axis=0).astype(np.float32),
        "mean": np.concatenate(mean_chunks, axis=0),
        "std": np.concatenate(std_chunks, axis=0),
        "apt": np.concatenate(apt_chunks, axis=0),
    }


def gauss_template(
    pred_hr: np.ndarray,
    pred_amp: np.ndarray,
    sigma: float,
    length: int = 24,
) -> np.ndarray:
    """Gaussian peak template, normalised so g.max(axis=1) == pred_amp.

    Mirrors experiments/v01_peak_from_latent/09_iter4_mechanisms.py:gauss_template.

    (한글) W5 hybrid 보정의 두 번째 항 ``g(t; ĥ, â, σ)``를 만든다:
        ``g(t) = â · exp(-(t - ĥ)² / 2σ²)``  단, max-normalize 후 â를 곱하므로
        실제 결과는 ``g.max(axis=1) == pred_amp``를 만족한다.

    왜 max-normalize 후 amplitude 곱 — 단순히 가우시안 × â로 두면 σ가 작을 때
    수치적으로 max값이 â보다 작거나 클 수 있다. 정규화로 "peak amplitude를 정확히
    â만큼 보존"하는 W family 규약을 강제 (v01 §iter4와 비트 정확).

    σ는 op-point에 무관하게 3.0으로 고정 (carry-over from v01). cold split에서
    재튜닝 금지 (plan §"Non-goals").
    """
    # t shape: (1, length=24). pred_hr는 정수 시각 (aux head argmax 결과).
    t = np.arange(length, dtype=np.float32)[None, :]
    # 표준 가우시안 곡선 (broadcast: B × length).
    g = np.exp(-0.5 * ((t - pred_hr.astype(np.float32)[:, None]) / sigma) ** 2)
    # max-normalize: 최고점이 1.0이 되도록 → 곱한 후 g.max == pred_amp 보장.
    g = g / g.max(axis=1, keepdims=True)
    return (g * pred_amp[:, None]).astype(np.float32)


def metrics_z_to_kw(
    true_z: np.ndarray,
    pred_z: np.ndarray,
    mean_arr: np.ndarray,
    std_arr: np.ndarray,
) -> dict:
    """z-norm space 텐서들을 per-window (mean, std)로 denormalize한 후
    PAPE / HR@1 / HR@2 / MAE를 kW 단위로 계산해서 반환.

    plan §"Metrics" — PAPE는 kW(denormalised) 기준이며, v01 §4.1과 비트 정확
    (``compute_pape`` / ``compute_hr``는 ``Peak_Analysis``로부터 비트 정확 포팅,
    수정 금지). HR@1/HR@2 모두 보고하지만 ``seven_axis_metrics``는 사용하지 않음
    (필요한 4지표만 골라 dict 빌드 — MSE는 v02 보고서 외부).
    """
    # per-window broadcasting: true_z (N, H) * std (N, 1) + mean (N, 1).
    true_kw = true_z * std_arr[:, None] + mean_arr[:, None]
    pred_kw = pred_z * std_arr[:, None] + mean_arr[:, None]
    return {
        "pape": float(compute_pape(true_kw, pred_kw)),
        "hr@1": float(compute_hr(true_kw, pred_kw, tol=1)),
        "hr@2": float(compute_hr(true_kw, pred_kw, tol=2)),
        "mae": float(compute_mae(true_kw, pred_kw)),
    }


def route_R0(
    co_key: np.ndarray,
    key_scaler_mean: np.ndarray,
    key_scaler_scale: np.ndarray,
    key_pool_scaled: np.ndarray,
    train_cluster_idx: np.ndarray,
) -> np.ndarray:
    """Cold KEY -> 1-NN on scaled train KEY pool -> train window's cluster_idx.

    (한글) v01과 동일한 R0 routing.
        1) cold KEY를 03이 fit/저장한 StandardScaler의 ``mean``/``scale``로 정규화
           (cold 측에서 scaler를 다시 fit하지 않음 — 공정한 zero-shot 보장).
        2) scaler 적용된 train KEY 풀에서 1-NN 이웃 1개 검색.
        3) 그 이웃 train 윈도우가 03 fit 단계에서 받은 ``cluster_idx``를 그대로 cold의
           cluster 배정으로 사용. (KEY 자체가 centroid로 가는 게 아니라, 가장 비슷한
           train 윈도우를 찾고 그 윈도우의 라벨을 빌려 쓰는 구조.)

    KEY는 입력만으로 계산되므로 backbone forward 0회 — README "Routings" 표 일치.
    """
    # cold KEY를 train 시점 scaler로 정규화 (재fit 금지).
    co_key_scaled = (co_key - key_scaler_mean) / key_scaler_scale
    # sklearn NearestNeighbors는 sklearn KMeans와 동일하게 default Euclidean.
    nn = NearestNeighbors(n_neighbors=1).fit(key_pool_scaled)
    _, neigh_idx = nn.kneighbors(co_key_scaled)
    # 가장 가까운 train 윈도우의 cluster index를 cold 윈도우의 cluster로 채택.
    return train_cluster_idx[neigh_idx[:, 0]]


def route_R1(co_h_g: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Cold h_g_cold -> argmin_c ||h_g - centroid_c||_2.

    (한글) v02의 새로운 ablation 라우팅.
        cold 윈도우의 ``h_g_cold`` (64-d)와 코드북 centroid M개 사이의 raw
        Euclidean ²-거리를 직접 계산해 가장 가까운 centroid 인덱스를 cluster로 사용.
        — h_g는 어차피 aux head 호출 때문에 forward에서 산출되므로 추가 backbone 호출
          0회 (R0와 동일).
        — KEY 5-d → centroid 64-d로 ×12 정보량 증가가 routing 결정을 바꾸는지가
          plan §G2의 핵심 질문.

    plan §"Open question 2": 거리 메트릭은 **raw Euclidean이 default** (현 구현).
        만약 underperform하면 StandardScaler-normalised 버전을 시도해볼 것 — 토글은
        아직 두지 않음 (현재 v02에서는 raw로 G2를 결정하기로).
    """
    # (N, 1, D) - (1, M, D) → (N, M, D). 64×M=32라 메모리 문제 없음.
    d = ((co_h_g[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1).astype(np.int64)


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
