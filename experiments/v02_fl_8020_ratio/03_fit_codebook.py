"""Fit the post-hoc Peak-VQ codebook on the v02 80-train-apt T2 latents (per-seed).

(한글 요약)
v02의 03번 스크립트. 한 seed에 대해 80 train apt의 T2 backbone에서 추출한
``h_generic`` (64-d latent) 위에 1-shot KMeans++ 코드북(M=32, D=64)을 fit하고,
이후 W5 hybrid correction이 cold inference 단계에서 그대로 사용할 수 있도록
다음 산출물을 함께 저장한다:

    - ``codebook``        : KMeans 중심점 (centroids) — R1 routing의 룩업 대상.
    - ``offsets``         : cluster별 residual 평균 ``o_{c*}`` (V0 mechanism의 보정량).
    - ``key_pool`` (+ scaler) : R0 routing용 KEY 풀과 StandardScaler 파라미터
                            (cold 측에서 동일 정규화를 재현하기 위해 mean/scale을 같이 기록).

이 코드북은 fit 이후 **동결**되며 STE(straight-through estimator)는 사용하지 않는다.
멀티 시드 sweep ({42, 123, 7})은 스크립트 안에 두지 않고 ``--seed S``로 외부 launcher가
시드마다 한 번씩 호출한다 (memory: feedback_argparse_per_seed).

For one seed:
    1. Load the frozen T2 backbone produced by ``02_train_arms.py``.
    2. Forward all train apts' train-segment windows (stride=24, matching v01)
       through the frozen backbone; collect (h_g, y_hat_z, y_true_z, key).
    3. Fit KMeans++ with M=32 on h_g — the codebook is **post-hoc 1-shot**
       (CLAUDE.md: iterative federated KMeans is out of scope through v03).
       1-shot인 이유: arxiv:2511.07073의 TAR attack(반복적 centroid 공개로
       43~77% 입력 재구성)을 회피하기 위함 (plan §"Why h_g aggregation is acceptable").
    4. Compute per-cluster residual offsets in z-norm space:
           offset_c = mean over {windows i: c*(i) = c} of (y_true_z[i] - y_hat_z[i]).
       이는 W5 family의 V0 보정항 ``o_{c*}``로, 한 cluster에 속한 train 윈도우들의
       평균 잔차이다 (cold 측에서 ``ŷ_corr = ŷ_base + α_v0·o_{c*} + α_w1·g(t;ĥ,â,σ)``).
    5. Build the KEY pool for R0 routing: 5-d KEY for every train window plus
       the StandardScaler params; cold side will reproduce the scaler exactly.
       KEY 디스크립터는 입력 윈도우만으로 계산되는 5-d 요약 (input-only,
       no future leakage) — cold gucha가 backbone을 부르지 않고도 동일하게 계산 가능.

The codebook bundle is saved to
``outputs/v02_fl_8020_ratio/seed{S}/codebook.npz`` and a separate
``codebook_diagnostics.json`` records utilisation / perplexity / k_min so the
v02 G1 health-metric check (k_min ≥ 113 at M=32) is reproducible.

Per-seed invocation:
    uv run python experiments/v02_fl_8020_ratio/03_fit_codebook.py --seed 42
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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import OUTPUT_DIR, RANDOM_SEED, TRAIN_RATIO
from dataloader.splits import load_v02_split
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from models.nbeatsx_aux import NBEATSxAux
from models.vq_kmeans import VectorQuantizerKMeans
from probes.peak_descriptor import extract_key

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V02_OUT_ROOT = OUTPUT_DIR / "v02_fl_8020_ratio"


def gather_train_segment(
    apts: list[str],
    model: NBEATSxAux,
    batch: int = 256,
    stride: int = 24,
) -> dict[str, np.ndarray]:
    """Collect (h_g, y_hat_z, y_true_z, key) on the train segment of each apt.

    Stride matches v01's gather_features (= horizon, non-overlapping) so
    codebook fit statistics stay comparable across versions.

    (한글) 각 train apt의 학습 구간(전체 시계열의 ``TRAIN_RATIO=0.7``까지)에서
    stride=24 (= horizon, non-overlapping)로 슬라이딩 윈도우를 만들어 frozen T2
    backbone에 통과시키고 다음 4종을 모은다:

        - ``h_g``     : ``h_generic`` ∈ ℝ^{N×64}, KMeans 학습 입력.
        - ``y_hat_z`` : 모델의 z-norm space 예측값. residual = y_true_z - y_hat_z.
        - ``y_true_z``: ground truth (z-norm space).
        - ``key``     : 5-d KEY 디스크립터 [max, argmax/96, mean, std, last24_max] —
                        R0 routing의 KEY pool.

    per-apt z-norm 통계(mean, std)는 학습 구간(``seg = series[:train_end]``)에서만
    계산하며 ``std<1e-8``이면 1.0으로 떨어뜨린다 (CLAUDE.md 컨벤션).
    """
    h_chunks, yhat_chunks, ytrue_chunks, key_chunks = [], [], [], []
    n_windows_per_apt = []
    for apt in apts:
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            print(f"  [skip] {apt}: missing")
            continue
        # 시계열 전체 길이의 70%까지만 train 구간으로 사용 (CLAUDE.md TRAIN_RATIO=0.7).
        n = len(series)
        train_end = int(n * TRAIN_RATIO)
        seg = series[:train_end]
        # per-apt z-norm: 학습 구간 통계만 사용 (standard 컨벤션, std<1e-8이면 1.0).
        m_ = float(seg.mean())
        s_ = float(seg.std()) if seg.std() > 1e-8 else 1.0
        # stride=24 → 윈도우들이 겹치지 않게 (v01과 동일, codebook 통계 호환).
        ds = HouseholdDataset(seg, m_, s_, stride=stride)
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=batch, shuffle=False)
        per_apt = 0
        for x, y in loader:
            x_dev = x.to(DEVICE)
            with torch.no_grad():
                # frozen T2 forward: y_hat (예측), hiddens (h_trend/h_seasonal/h_generic),
                # 세 번째 반환값(_)는 NBEATSxAux의 aux head 출력 — 여기선 미사용.
                y_hat, hiddens, _ = model(x_dev)
            # h_generic만 codebook fit에 사용 (h_concat 192-d는 v01 T3 arm 한정, v02는 T2).
            h_g = hiddens["h_generic"].cpu().numpy()
            h_chunks.append(h_g)
            yhat_chunks.append(y_hat.cpu().numpy())
            ytrue_chunks.append(y.numpy())
            # KEY는 입력 x로부터 직접 계산 — backbone 의존성 없음 (cold side에서 동일 재현).
            key_chunks.append(extract_key(x.numpy()))
            per_apt += len(x)
        n_windows_per_apt.append(per_apt)
    return {
        "h_g": np.concatenate(h_chunks, axis=0),
        "y_hat_z": np.concatenate(yhat_chunks, axis=0),
        "y_true_z": np.concatenate(ytrue_chunks, axis=0),
        "key": np.concatenate(key_chunks, axis=0),
        "n_windows_per_apt": np.asarray(n_windows_per_apt, dtype=np.int64),
    }


def fit_codebook(seed: int, M: int, arm: str, batch: int, stride: int) -> dict:
    """한 seed에 대해 codebook + offsets + KEY pool 번들을 만들고 npz로 저장한다.

    절차:
        1. v02 split 로드(80 train apts) → 이 seed의 T2 best.pt 체크포인트 로드.
           backbone은 ``02_train_arms.py``에서 학습되며 여기선 frozen forward만.
        2. ``gather_train_segment``로 train 구간 윈도우 latent / 예측 / 잔차 / KEY 수집.
        3. ``VectorQuantizerKMeans(M=32, D=64).fit()`` — 1-shot KMeans++ (post-hoc).
           후속 단계에서 centroid는 갱신되지 않으며 STE도 사용하지 않는다 (CLAUDE.md §
           "things to know"). 1-shot 선택은 iterative KMeans의 TAR attack
           (arxiv:2511.07073)을 회피하기 위함.
        4. 각 train 윈도우의 cluster index를 ``vq.forward``로 얻어 per-cluster
           residual 평균 ``offset_c = mean_{i ∈ c}(y_true_z[i] - y_hat_z[i])`` 를
           계산. 이것이 W5 V0 보정항 ``o_{c*}`` (z-norm space).
        5. KEY 풀 + StandardScaler를 fit → cold side에서 동일 mean/scale로 정규화한
           뒤 1-NN으로 train cluster를 찾는 R0 routing의 룩업 자료.
        6. 저장:
            - ``codebook.npz`` (centroid는 R1 routing용 룩업, 04_coldstart_eval.py 사용)
            - ``codebook_diagnostics.json`` (utilization / perplexity / k_min / inertia
              + v01 health threshold k_min ≥ 113 통과 여부 G1 점검)
    """
    torch.manual_seed(seed); np.random.seed(seed)

    # v02 80:20 split의 train 80 apts. cold 20 apts는 04에서 따로 사용.
    apts = load_v02_split(seed)["train"]
    seed_root = V02_OUT_ROOT / f"seed{seed}"
    arm_dir = seed_root / arm
    ckpt = arm_dir / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"backbone checkpoint missing: {ckpt}. "
            f"Run 02_train_arms.py --seed {seed} --arms {arm} first."
        )

    # T2 backbone (NBEATSxAux + peak_aux head) 로드 후 frozen — eval 모드 고정.
    # latent_source='h_generic'이면 codebook D=64 (v02 표준). h_concat(192)은 v01 T3 한정.
    model = NBEATSxAux(latent_source="h_generic").to(DEVICE).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    print(f"[fit] seed={seed} arm={arm} backbone loaded ({sum(p.numel() for p in model.parameters())} params)")

    feats = gather_train_segment(apts, model, batch=batch, stride=stride)
    h_g = feats["h_g"]
    print(f"[fit] {len(apts)} apts, {h_g.shape[0]} train windows; stride={stride}")

    # 1-shot KMeans++ fit. n_init=10은 vq_kmeans.py에 하드코딩 (sklearn KMeans 기본값).
    # post-hoc & non-iterative — 학습 루프에서 centroid 업데이트 없음, gradient 흐르지 않음.
    vq = VectorQuantizerKMeans(num_embeddings=M, embedding_dim=h_g.shape[1], random_state=seed)
    diag = vq.fit(torch.from_numpy(h_g).float())
    print(
        f"[fit] M={M}  util={diag['utilization']:.3f}  ppl={diag['perplexity']:.2f}  "
        f"k_min={diag['k_min']}  k_max={diag['k_max']}  inertia={diag['kmeans_inertia']:.1f}"
    )

    # ---- per-cluster residual offset 계산 (W5 V0 보정량 o_{c*}) ----
    # centroids: (M, D) — R1 routing의 룩업 대상 (cold h_g와 직접 거리 비교).
    # counts: (M,) — fit 시점 각 cluster에 할당된 train 윈도우 수.
    centroids = vq.codebook.cpu().numpy()
    counts = vq.counts.cpu().numpy()
    h_t = torch.from_numpy(h_g).float()
    with torch.no_grad():
        # 각 train 윈도우의 cluster index. 비어있는 cluster는 offsets에서 0으로 남음.
        _, cluster_idx_t = vq(h_t)
    cluster_idx = cluster_idx_t.cpu().numpy().astype(np.int64)
    # residual = y_true_z - y_hat_z (z-norm space). cold 측에서도 동일한 z-norm space에서
    # ŷ_corr_z = ŷ_base_z + α_v0·o_{c*} + α_w1·g 로 적용.
    residuals = feats["y_true_z"] - feats["y_hat_z"]
    horizon = residuals.shape[1]
    offsets = np.zeros((M, horizon), dtype=np.float32)
    for c in range(M):
        mask = cluster_idx == c
        if mask.any():
            # cluster 내 평균 잔차 — 한 cluster의 "전형적인 미보정 패턴".
            offsets[c] = residuals[mask].mean(axis=0)

    # ---- R0 routing용 KEY pool + scaler 저장 ----
    # cold gucha는 backbone forward 없이 입력만으로 KEY를 계산할 수 있으므로,
    # 여기서 fit한 scaler의 mean/scale을 그대로 cold 측에서 재현해야 1-NN 정규화 일관.
    key_pool = feats["key"].astype(np.float32)
    key_scaler = StandardScaler().fit(key_pool)
    key_pool_scaled = key_scaler.transform(key_pool).astype(np.float32)

    # 한 번의 npz에 04_coldstart_eval.py가 R0/R1 routing × W5 보정에 필요한 모든 자료를 모음.
    # (v01은 04_quantize_h1b.py에서 codebook+counts만 저장하고, offsets/key_pool은 cold-start
    #  단계에서 매번 재계산했음 — v02는 reproducibility를 위해 fit 시점 스냅샷을 통째로 보관.)
    out_path = seed_root / "codebook.npz"
    np.savez(
        out_path,
        codebook=centroids.astype(np.float32),       # (M, D)  R1 routing 룩업 대상
        counts=counts.astype(np.int64),              # (M,)     cluster별 train 할당 수
        offsets=offsets,                             # (M, H)  V0 보정항 o_{c*} (z-norm)
        cluster_idx=cluster_idx.astype(np.int32),    # (N,)     train 윈도우의 cluster 매핑
        key_pool=key_pool,                           # (N, 5)  R0 KEY pool (raw)
        key_pool_scaled=key_pool_scaled,             # (N, 5)  R0 KEY pool (StandardScaler 적용)
        key_scaler_mean=key_scaler.mean_.astype(np.float32),     # cold 재현용
        key_scaler_scale=key_scaler.scale_.astype(np.float32),   # cold 재현용
        n_windows_per_apt=feats["n_windows_per_apt"],
    )

    # v02 G1 health check: codebook 진단치를 JSON으로 별도 보관.
    # k_min ≥ 113 임계는 v01에서 50:50 split·M=32일 때 가장 sparse한 cluster의 train 윈도우 수
    #   기준선이며, 80:20에서도 codebook이 "건강하게" fit되었는지를 호환 가능하게 점검하는 게이트.
    diagnostics = {
        "seed": int(seed),
        "arm": arm,
        "split_version": "v02",
        "M": int(M),
        "embedding_dim": int(h_g.shape[1]),
        "n_train_apts": int(len(apts)),
        "n_train_windows": int(h_g.shape[0]),
        "stride": int(stride),
        "horizon": int(horizon),
        "vq_utilization": float(diag["utilization"]),
        "vq_perplexity": float(diag["perplexity"]),
        "vq_k_min": int(diag["k_min"]),
        "vq_k_max": int(diag["k_max"]),
        "vq_kmeans_inertia": float(diag["kmeans_inertia"]),
        "n_empty_clusters": int((counts == 0).sum()),
        "k_min_health_threshold_v01": 113,
        "k_min_health_pass": bool(int(diag["k_min"]) >= 113),
        "key_dim": int(key_pool.shape[1]),
    }
    with open(seed_root / "codebook_diagnostics.json", "w") as fh:
        json.dump(diagnostics, fh, indent=2)

    print(f"[fit] saved {out_path.name} + codebook_diagnostics.json")
    return diagnostics


def main() -> None:
    # CLI 인자: per-seed 호출 컨벤션 (memory: feedback_argparse_per_seed). 멀티시드 sweep은
    # 외부 launcher가 ``--seed 42 / 123 / 7`` 세 번 돌려서 처리하며, 스크립트 내부 루프 X.
    # M=32, T2-only는 plan §"Method"와 README "Backbone arms" 표에 고정된 값.
    ap = argparse.ArgumentParser(description="Fit M=32 KMeans codebook on v02 T2 latents (per-seed).")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--M", type=int, default=32, help="Codebook size.")
    ap.add_argument("--arm", type=str, default="T2", choices=["T2"], help="Backbone arm; v02 uses T2 only.")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--stride", type=int, default=24, help="Window stride; v01 uses 24 (= horizon).")
    args = ap.parse_args()

    fit_codebook(seed=args.seed, M=args.M, arm=args.arm, batch=args.batch, stride=args.stride)


if __name__ == "__main__":
    main()
