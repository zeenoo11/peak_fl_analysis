"""E1 ablation @ 80:20 — peak_aux ON/OFF on V0 mechanism.

Mirror of v01's §4.3 / experiments/v01_peak_from_latent/15_E1_peak_aux_ablation.py:
holds the correction mechanism fixed (V0 cluster-mean offset, α=2.0 — v01's
clean-comparison choice with no aux predictions involved) and varies the
backbone training only:

    T0  = MinimalNBEATSx, MAE only         (no peak_aux)
    T2  = NBEATSxAux,    MAE + λ·peak_aux  (with peak_aux)

For each arm we fit an independent codebook on that arm's own h_g latents,
because the two backbones produce different latent spaces.

Per-seed invocation:
    uv run python experiments/v02_fl_8020_ratio/05_E1_ablation.py --seed 42

Output: outputs/v02_fl_8020_ratio/seed{S}/E1_results.json

(한글 요약)
v02 §G1 후속 — **E1 ablation**. v01 §4.3에서 보고한 "peak_aux 추가로 cold PAPE
+18.6 pp 개선" 헤드라인이 80:20 split에서도 살아남는지 검증.

직교 ablation 2축 (README "What 05 vs 06 isolates" 박스 참조):
    - 05 (이 파일) = **mechanism 고정 (V0 only) × backbone 토글 (T0 ↔ T2)**
    - 06          = backbone 고정 (T2)        × mechanism 토글 (V0 / W1a / W5)

설계 결정:
    1) **V0 mechanism만 평가** — T0는 peak_aux head가 없어서 (â, ĥ)를 만들지
       못하므로 W1a/W5의 Gaussian template을 fair하게 비교할 수 없다. 따라서
       E1은 V0(cluster-mean offset)만으로 묶는다. (v01 §4.3과 동일한 선택.
       v01 15번 원본은 T0에 self-derived aux를 끼워 W5도 비교했지만, v02 E1은
       V0-only로 더 깨끗하게 잘라낸다.)
    2) **R0 routing only** — E1의 1차 비교축은 backbone이지 routing이 아니다.
       (R0/R1 비교는 G2이며 04번에서 처리.)
    3) **arm별 독립 codebook fit** — T0와 T2는 latent space 자체가 다르므로
       동일 codebook을 공유할 수 없다. 따라서 03이 만든 ``codebook.npz``
       (T2 latent 위에서만 fit됨)는 **이 스크립트에서 사용하지 않으며**, T0/T2
       각각의 h_generic 위에 KMeans(M=32)을 새로 fit한다 (v01 §4.3과 동일).
    4) α_v0 = 2.0 — v01 §4.3의 clean-comparison 값을 그대로 carry-over.
       (04의 op-point들과 다른 값. E1은 mechanism 자체 효과만 분리하기 위해
       단일 강도로 평가.)

평가 protocol은 v01 §4.3과 비트 정확하게 동일하므로, 출력 JSON의
``peak_aux_contribution_on_V0.pape``를 v01의 +18.6 pp와 직접 비교 가능.

멀티 seed sweep ({42, 123, 7})은 외부 launcher가 ``--seed S``로 시드마다 한 번씩
호출 (memory: feedback_argparse_per_seed). 스크립트 안에 시드 루프 없음.

Inputs (per seed):
    outputs/v02_fl_8020_ratio/seed{S}/T0/best.pt   — MinimalNBEATSx (peak_aux OFF)
    outputs/v02_fl_8020_ratio/seed{S}/T2/best.pt   — NBEATSxAux (peak_aux ON)
    outputs/v02_fl_8020_ratio/splits/v02_8020_seed{S}.yaml

Outputs:
    outputs/v02_fl_8020_ratio/seed{S}/E1_results.json
        — arm별 baseline + V0 보정 metric, 그리고
          peak_aux_contribution_on_V0 = T2.V0 - T0.V0 (G1 +18.6 pp 검증치).
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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import OUTPUT_DIR, RANDOM_SEED, TRAIN_RATIO
from dataloader.splits import load_v02_split
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from models.nbeatsx import MinimalNBEATSx
from models.nbeatsx_aux import NBEATSxAux
from models.vq_kmeans import VectorQuantizerKMeans
from probes.peak_descriptor import extract_key
from utils.metrics import compute_hr, compute_mae, compute_pape

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V02_OUT_ROOT = OUTPUT_DIR / "v02_fl_8020_ratio"


def load_arm(arm: str, seed: int) -> tuple[torch.nn.Module, bool]:
    """02 산출 frozen backbone(``best.pt``) 로드. ``(model, is_aux)`` 반환.

    (한글) arm에 따라 모델 클래스가 달라지고, forward 반환 튜플 길이도 달라지므로
    ``is_aux`` 플래그를 함께 돌려준다 (gather()가 forward dispatch에 사용).

        - T0 → ``MinimalNBEATSx``       : forward → ``(y_hat, hiddens)``         (2-tuple)
        - T2 → ``NBEATSxAux(h_generic)`` : forward → ``(y_hat, hiddens, (â, ĥ))`` (3-tuple)

    state_dict 키는 v10 b2 호환이라 ``load_state_dict`` default(strict=True)로 자동 로드.
    backbone은 02에서 train된 frozen 상태 그대로 — E1은 추가 학습 일절 없음.
    """
    seed_root = V02_OUT_ROOT / f"seed{seed}"
    ckpt = seed_root / arm / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing {ckpt}; run 02_train_arms.py --seed {seed} --arms {arm} first.")
    if arm == "T0":
        # T0: peak_aux head 없음. forward 반환 2-tuple → is_aux=False.
        m = MinimalNBEATSx().to(DEVICE).eval()
        m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
        return m, False
    if arm == "T2":
        # T2: peak_aux head 부착. forward 반환 3-tuple → is_aux=True.
        # latent_source='h_generic'으로 v01/v02 표준 (h_concat은 T3 v01-only).
        m = NBEATSxAux(latent_source="h_generic").to(DEVICE).eval()
        m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
        return m, True
    raise ValueError(arm)


def gather(
    apts: list[str],
    model: torch.nn.Module,
    is_aux: bool,
    batch: int = 256,
    stride: int = 24,
) -> dict[str, np.ndarray]:
    """주어진 apt 리스트의 train segment(앞 70%)에서 frozen forward 1회.

    (한글) 04의 ``gather_cold``와 거의 동일한 구조이지만 다음 두 가지가 다르다:
        1) **train과 cold 모두에 호출**된다 (04는 cold만). E1은 train 윈도우로
           codebook을 fit하고 cold 윈도우로 평가하는 구조라, 같은 함수를 두 번
           쓰면서 ``apts``만 train_apts/cold_apts로 바꿔 부른다.
        2) **aux head 출력 (â, ĥ)을 수집하지 않는다** — V0 mechanism은
           cluster offset만 사용하므로 Gaussian template이 필요 없고, T0는 어차피
           aux head가 없어서 출력도 없다.

    z-norm은 cold-side 학습 금지 원칙을 따라 각 apt 자기 시계열의 앞 ``TRAIN_RATIO``
    구간 통계만 사용 (warm-start z-norm). std<1e-8 fallback=1.0은 모든 v01/v02
    스크립트와 동일.

    stride=24 (= horizon, non-overlapping)는 03/04와 동일하게 맞춰 train↔cold
    윈도우 카운트가 정합적이게 유지.

    forward dispatch:
        - is_aux=False (T0): ``y_hat, hiddens = model(x)``    (2-tuple)
        - is_aux=True  (T2): ``y_hat, hiddens, _ = model(x)`` (aux 출력은 V0에 불필요해 버림)
    """
    h_chunks, yhat_chunks, ytrue_chunks, key_chunks = [], [], [], []
    mean_chunks, std_chunks = [], []
    for apt in apts:
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            # 데이터 빠진 apt는 조용히 건너뜀 (04와 달리 print 없음 — 빠진 것 가정에서 OK).
            continue
        n = len(series)
        train_end = int(n * TRAIN_RATIO)
        seg = series[:train_end]
        # per-apt z-norm 통계: 자기 train 구간만 사용 (cold의 future label leak 차단).
        m_ = float(seg.mean())
        s_ = float(seg.std()) if seg.std() > 1e-8 else 1.0
        ds = HouseholdDataset(seg, m_, s_, stride=stride)
        if len(ds) == 0:
            continue
        for x, y in DataLoader(ds, batch_size=batch, shuffle=False):
            x_dev = x.to(DEVICE)
            with torch.no_grad():
                if is_aux:
                    # T2: 3-tuple. aux 출력 (â, ĥ)은 V0 보정에 불필요 → 버림.
                    y_hat, hiddens, _ = model(x_dev)
                else:
                    # T0: 2-tuple. aux head 자체가 없음.
                    y_hat, hiddens = model(x_dev)
            # h_generic은 codebook KMeans fit / R0 routing에 모두 필요.
            h_chunks.append(hiddens["h_generic"].cpu().numpy())
            yhat_chunks.append(y_hat.cpu().numpy())
            ytrue_chunks.append(y.numpy())
            # KEY는 입력 x로부터 직접 (input-only 5-d). T0/T2 모두 동일하게 계산.
            key_chunks.append(extract_key(x.numpy()))
            mean_chunks.append(np.full(len(y), m_, dtype=np.float32))
            std_chunks.append(np.full(len(y), s_, dtype=np.float32))
    return {
        "h_g": np.concatenate(h_chunks, axis=0).astype(np.float32),
        "y_hat_z": np.concatenate(yhat_chunks, axis=0).astype(np.float32),
        "y_true_z": np.concatenate(ytrue_chunks, axis=0).astype(np.float32),
        "key": np.concatenate(key_chunks, axis=0).astype(np.float32),
        "mean": np.concatenate(mean_chunks, axis=0),
        "std": np.concatenate(std_chunks, axis=0),
    }


def fit_vq_and_offsets(
    h_g: np.ndarray, residuals: np.ndarray, M: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """**arm별 독립** post-hoc 1-shot KMeans codebook fit + per-cluster residual offsets.

    (한글) **이 함수가 03 ``codebook.npz``를 사용하지 않고 새로 fit하는 이유**:
        03이 만든 codebook은 T2 backbone의 h_generic 위에서만 fit된 것이다.
        E1은 T0 backbone도 평가하는데, T0의 h_generic 공간은 T2의 그것과 완전히
        다른 분포를 가진다 (peak_aux loss 유무 때문에 latent geometry 자체가
        다름). 따라서 T0 평가에서 T2 codebook을 그대로 쓰면 routing/offset이
        모두 의미를 잃는다. 그래서 E1은 **arm마다 그 arm의 latent에 codebook을
        새로 fit**한다 (v01 §4.3 / 15번과 동일한 선택).

    절차:
        1) ``VectorQuantizerKMeans(M=32, D=h_g.shape[1])``로 1-shot KMeans++ fit
           (sklearn KMeans 1회). ``random_state``는 ``--seed``로 결정 → 시드 결정성.
        2) ``vq(h_g)`` 호출로 train 윈도우의 cluster 인덱스 산출.
        3) cluster별로 residual = (y_true_z - y_hat_z)의 평균을 offset으로 저장.
           이게 V0 보정량 ``o_{c}`` (z-norm space, shape [M, 24]).

    diag 정보 (utilization / perplexity / k_min / k_max)는 T0/T2 codebook 건강도
    비교에 사용 — README는 T0가 "codebook collapse" (k_min 매우 작음)로 V0가
    덜 작동해서 peak_aux 효과가 +24.7 pp까지 부풀려진다고 보고.
    """
    # 1-shot KMeans++ fit. embedding_dim은 h_g shape에 맞춰 자동 (h_generic은 64).
    vq = VectorQuantizerKMeans(num_embeddings=M, embedding_dim=h_g.shape[1], random_state=seed)
    diag = vq.fit(torch.from_numpy(h_g).float())
    cb = vq.codebook.cpu().numpy()
    with torch.no_grad():
        # train 윈도우 → 가장 가까운 centroid → cluster 인덱스 (R0/R1과 별개의 train-side 배정).
        _, idx_t = vq(torch.from_numpy(h_g).float())
    cluster_idx = idx_t.cpu().numpy().astype(np.int64)
    # cluster별 평균 residual = V0 보정량 o_c (z-norm space, shape [M, 24]).
    offsets = np.zeros((M, residuals.shape[1]), dtype=np.float32)
    for c in range(M):
        mask = cluster_idx == c
        if mask.any():
            offsets[c] = residuals[mask].mean(axis=0)
        # 빈 cluster는 0 offset 유지 (보정 없음 = baseline forecast 그대로).
    return cb, offsets, cluster_idx, {
        "utilization": float(diag["utilization"]),
        "perplexity": float(diag["perplexity"]),
        "k_min": int(diag["k_min"]),
        "k_max": int(diag["k_max"]),
    }


def metrics_z_to_kw(true_z, pred_z, mean_arr, std_arr) -> dict:
    """z-norm space → kW(denorm) → PAPE / HR@{1,2} / MAE.

    (한글) 04 ``metrics_z_to_kw``와 의도/구현 동일. 보정은 z-norm에서 이루어지고
    metric 보고는 kW 단위 (v01 §4.1과 비트 정확). PAPE/HR은 ``Peak_Analysis``에서
    포팅된 함수 (수정 금지).

    NOTE: 04와 코드가 똑같다 — 별도 import가 아니라 복붙되어 있음. 향후 한쪽이
    수정되면 drift 위험. (보고 §"04 헬퍼 공유" 항목 참조.)
    """
    true_kw = true_z * std_arr[:, None] + mean_arr[:, None]
    pred_kw = pred_z * std_arr[:, None] + mean_arr[:, None]
    return {
        "pape": float(compute_pape(true_kw, pred_kw)),
        "hr@1": float(compute_hr(true_kw, pred_kw, tol=1)),
        "hr@2": float(compute_hr(true_kw, pred_kw, tol=2)),
        "mae": float(compute_mae(true_kw, pred_kw)),
    }


def evaluate_arm(
    arm: str,
    train_apts: list[str],
    cold_apts: list[str],
    seed: int,
    M: int,
    alpha_v0: float,
    batch: int,
    stride: int,
) -> dict:
    """한 arm(T0 또는 T2)에 대한 E1 평가 1 행.

    (한글) E1 비교의 한 row를 만든다 (T0 한 번, T2 한 번 main()에서 호출).

    절차 (v01 §4.3 / 15번과 비트 정확하게 동일):
        1) frozen backbone 로드 (02 산출 ``best.pt``).
        2) **train apts에서 forward** → h_g 수집 (이 arm 전용 codebook을 fit하기 위해).
           **cold apts에서 forward** → 평가용 h_g/ŷ_base/y_true 수집.
           — 03 codebook.npz는 사용하지 않음. T0/T2 각각의 latent 공간이 다르므로
             이 arm의 latent에 codebook을 새로 fit해야 의미가 있음.
        3) ``fit_vq_and_offsets``: train residual = y_true_z - y_hat_z 평균을 cluster
           offset으로 → V0 보정량 ``o_c``. utilization/k_min 등 codebook 건강도 진단도 출력.
        4) **R0 routing**: cold KEY를 train KEY 풀에 StandardScaler 후 1-NN로 매핑,
           이웃 train 윈도우의 ``cluster_idx``를 cold 윈도우 cluster로 사용.
           — 04와 달리 scaler를 codebook.npz에서 읽지 않고 매 호출 새로 fit
             (T0/T2 각각 독립이므로 03 산출물을 재사용할 수 없음).
        5) baseline metric: ŷ_base만 denorm → PAPE/HR/MAE.
           V0 metric: ŷ_base + α_v0 · o_{c*}를 denorm → PAPE/HR/MAE.
        6) ``delta_v0_minus_base``: V0 보정의 효과 (mechanism 자체가 어느 정도
           작동하는지의 within-arm 진단).

    호출자(main)는 두 arm 결과의 V0를 빼서 ``peak_aux_contribution_on_V0``를 계산
    → v01 §4.3의 +18.6 pp 헤드라인과 직접 비교 가능 (G1 검증).
    """
    print(f"\n========== E1 {arm} (seed {seed}) ==========")
    # arm별 backbone 로드 (T0/T2 각각의 best.pt). is_aux로 forward 분기.
    model, is_aux = load_arm(arm, seed)

    # train과 cold 모두 forward — train은 codebook fit + R0 KEY 풀, cold는 평가.
    tr = gather(train_apts, model, is_aux, batch=batch, stride=stride)
    co = gather(cold_apts, model, is_aux, batch=batch, stride=stride)
    print(f"  windows: train={tr['h_g'].shape[0]}  cold={co['h_g'].shape[0]}")

    # ---- arm 전용 codebook fit (03 codebook.npz 사용 안 함; arm마다 latent 공간 다름) ----
    residuals = tr["y_true_z"] - tr["y_hat_z"]  # z-norm space residual → V0 offset의 원천.
    cb, offsets, cluster_idx_tr, vq_diag = fit_vq_and_offsets(tr["h_g"], residuals, M, seed)
    print(
        f"  vq diag: util={vq_diag['utilization']:.3f}  ppl={vq_diag['perplexity']:.2f}  "
        f"k_min={vq_diag['k_min']}  k_max={vq_diag['k_max']}"
    )

    # ---- R0 routing (E1은 R0 only; routing 비교는 G2/04에서 처리) ----
    # KEY → scaled 1-NN on train pool → cluster_idx of that train window.
    # T0/T2마다 KEY 풀이 같지만 (KEY는 input-only) cluster_idx_tr는 다르므로 매번 fit 필요.
    ks = StandardScaler().fit(tr["key"])
    nn = NearestNeighbors(n_neighbors=1).fit(ks.transform(tr["key"]))
    _, ni = nn.kneighbors(ks.transform(co["key"]))
    cold_cluster = cluster_idx_tr[ni[:, 0]]

    # ---- baseline (보정 없음) — peak_aux 효과 측정의 reference ----
    base = metrics_z_to_kw(co["y_true_z"], co["y_hat_z"], co["mean"], co["std"])
    # ---- V0 보정 (cluster offset만; α_v0=2.0, v01 §4.3 carry-over) ----
    v0_corrected = co["y_hat_z"] + alpha_v0 * offsets[cold_cluster]
    v0 = metrics_z_to_kw(co["y_true_z"], v0_corrected, co["mean"], co["std"])

    # 표시용: V0 / baseline PAPE 비율 (1보다 작으면 보정이 PAPE를 낮춘 것 = 좋음).
    ratio = v0["pape"] / base["pape"] if base["pape"] > 0 else float("nan")
    print(
        f"  baseline: PAPE={base['pape']:.2f}  HR@1={base['hr@1']:.1f}  HR@2={base['hr@2']:.1f}"
    )
    print(
        f"  V0 (α={alpha_v0}): PAPE={v0['pape']:.2f}  HR@1={v0['hr@1']:.1f}  HR@2={v0['hr@2']:.1f}  "
        f"(ratio={ratio:.3f}, Δ={(1 - ratio) * 100:+.1f}%)"
    )

    return {
        "arm": arm,
        "is_aux": is_aux,
        "n_train_windows": int(tr["h_g"].shape[0]),
        "n_cold_windows": int(co["h_g"].shape[0]),
        "vq_diagnostics": vq_diag,
        "alpha_v0": alpha_v0,
        "M": int(M),
        "baseline": base,                   # 보정 없음 (G1 reference)
        "V0": v0,                           # V0 보정 결과 (mechanism 효과 with this backbone)
        # within-arm 보정 효과 (V0 - baseline). 절대값이 크면 mechanism이 잘 잡혔다는 뜻.
        # T0가 T2보다 V0 효과가 작으면 → "peak_aux 없으면 V0 mechanism이 잘 작동 안함" → +18.6 pp의 출처.
        "delta_v0_minus_base": {
            "pape": v0["pape"] - base["pape"],
            "hr@1": v0["hr@1"] - base["hr@1"],
            "hr@2": v0["hr@2"] - base["hr@2"],
            "mae": v0["mae"] - base["mae"],
        },
    }


def main() -> None:
    """v02 05번 entrypoint — 한 seed에 대해 E1 ablation (T0 vs T2 on V0).

    (한글) per-seed argparse 컨벤션 (memory: feedback_argparse_per_seed). 시드
    sweep은 외부 launcher가 ``--seed 42 / 123 / 7``을 따로 호출. 출력은
    ``seed{S}/E1_results.json`` 단일 파일 (plan §"Outputs" 트리 일치).

    핵심 출력 ``peak_aux_contribution_on_V0`` = T2.V0 - T0.V0:
        - PAPE 항목이 음수일수록 (T2가 T0보다 PAPE 낮음) peak_aux 효과 큼.
        - 부호를 뒤집어 "+M.M pp"로 보고하면 v01 §4.3의 +18.6 pp 헤드라인과 직접 비교.
        - README 보고치 (3 seeds 평균 +11.9 ± 9.2 pp; per-seed swing 3.6–24.7) — 큰 σ는
          T0 codebook이 collapse하는 시드(예: seed=42에서 +24.7 pp)에서 비롯.
    """
    ap = argparse.ArgumentParser(description="E1 ablation: peak_aux ON/OFF on V0 mechanism (per-seed).")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--M", type=int, default=32)
    # α_v0=2.0은 v01 §4.3의 clean-comparison 값 carry-over. 04의 op-point들과 다른 값.
    # cold split에서 재튜닝 금지 (plan §"Non-goals", v01 §5.4.1 selection bias).
    ap.add_argument("--alpha_v0", type=float, default=2.0, help="V0 strength; v01 §4.3 used 2.0.")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--stride", type=int, default=24)
    args = ap.parse_args()

    # 결정성: torch + numpy seed 모두 고정 (KMeans/NN init도 영향).
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # 80:20 split 로드 (01번 산출). train 80은 codebook fit + R0 KEY 풀에 사용.
    split = load_v02_split(args.seed)
    train_apts, cold_apts = split["train"], split["cold"]
    seed_root = V02_OUT_ROOT / f"seed{args.seed}"
    print(f"[setup] seed={args.seed}  train={len(train_apts)}  cold={len(cold_apts)}  M={args.M}  α_v0={args.alpha_v0}")
    print(f"[setup] device={DEVICE}  out={seed_root}")

    # 두 arm을 순차 평가 (T0 → T2). 각 arm 안에서 backbone load → forward → codebook fit → R0 → V0 보정.
    results = {}
    for arm in ["T0", "T2"]:
        results[arm] = evaluate_arm(
            arm, train_apts, cold_apts,
            seed=args.seed, M=args.M, alpha_v0=args.alpha_v0,
            batch=args.batch, stride=args.stride,
        )

    # ---- E1 헤드라인 계산: peak_aux 효과 = T2.V0 - T0.V0 (V0 보정된 cold metric 기준) ----
    # README 표기 "+24.7 pp at 80:20 (seed=42)"는 정확히 이 값의 PAPE 항목을 부호 반전한 것.
    # main()의 출력 print에 "delta(T2-T0)=-24.7 (down)"로 찍히고, JSON에는 raw -24.7로 저장됨.
    contribution = {
        k: results["T2"]["V0"][k] - results["T0"]["V0"][k]
        for k in ["pape", "hr@1", "hr@2", "mae"]
    }
    print("\n========== E1 SUMMARY ==========")
    for k in ["pape", "hr@1", "hr@2"]:
        t0_v = results["T0"]["V0"][k]
        t2_v = results["T2"]["V0"][k]
        # PAPE는 작을수록 좋음(down), HR은 클수록 좋음(up). 부호 해석을 안내.
        sign = "down" if k == "pape" else "up"
        print(f"  V0 {k:6s}: T0={t0_v:7.2f}  T2={t2_v:7.2f}  delta(T2-T0)={contribution[k]:+.2f} ({sign})")

    # 결과 dict — seed/split_version/M/alpha_v0 메타 + arm별 상세 + 헤드라인 contribution.
    out = {
        "seed": int(args.seed),
        "split_version": "v02",
        "M": int(args.M),
        "alpha_v0": float(args.alpha_v0),
        # arm별 baseline + V0 metric + codebook diag — 둘 다 같은 op-point(α_v0)에서 비교 가능.
        "results_by_arm": results,
        # T2 - T0 (V0 보정 후). PAPE 항목 부호 반전이 v01 §4.3의 "+18.6 pp"에 대응.
        "peak_aux_contribution_on_V0": contribution,
        "comment": "Mirrors v01 §4.3 / 15_E1_peak_aux_ablation.py — V0 only, α=2.0.",
    }
    out_path = seed_root / "E1_results.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  saved -> {out_path}")


if __name__ == "__main__":
    main()
