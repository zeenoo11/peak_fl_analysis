"""v09 TSFM zero-shot baseline — per-client protocol matching FedVQ (02).

(한글 요약)
v04/v05 의 FM zero-shot 베이스라인은 v02 cold pool(~20 가구)의 **train 구간**
(앞 70%) 윈도우를 window-pooled 로 평가한다 — v09 round-level FedVQ 와는
(1) 가구 집합, (2) 평가 구간, (3) 메트릭 집계 세 축이 모두 달라 직접 비교가
불가능하다. 본 스크립트는 그 세 축을 v09 에 정렬한 zero-shot TSFM 하한선이다:

  1. 가구 집합 = ``build_per_client_splits(seed)`` 의 114 가구 전체
     (02_fl_vq_dynamics.py 와 동일 split 캐시 재사용).
  2. 평가 구간 = 각 가구의 **test split (뒤 20%)** (``test_x`` / ``test_y``).
  3. 집계 = 가구별 PAPE/HR/MAE/MSE 를 구한 뒤 **가구 평균** (02 의
     ``_eval_per_client`` 와 동일: mean across clients, std ddof=1).

v09 split 의 ``test_x``/``test_y`` 는 per-apt train z-norm 이 적용된 z-space 다.
FM forecaster 는 raw kW 를 받아 내부 스케일링을 하므로, FM 입력 전 각 가구의
``mean``/``std`` 로 역정규화(``x*s+m``)하여 raw kW 를 만든 뒤 forecast 하고,
메트릭은 kW 공간에서 계산한다 (02 의 ``_eval_per_client`` 가 z→kW 로 되돌려
PAPE 를 재는 것과 동일한 kW 공간).

zero-shot 이라 학습이 없다 — ``--seed`` 는 split 캐시 경로만 바꾼다(현재 split 은
seed 에 무관하게 결정론적이므로 세 seed 의 숫자는 동일하지만, multi-seed
집계 컨벤션과 결과 디렉토리 구조를 02 와 맞추기 위해 per-seed 로 둔다).

Output (``outputs/v09_round_vq_codebook/seed{S}/fm_{model}/result.json``):
  - ``test_terminal`` — 02 의 ``test_terminal`` 과 동일 key 구조의 가구-평균 메트릭.

CLI:

    uv run python experiments/v09_round_vq_codebook/04_fm_zero_shot_per_client.py \\
        --seed 42 --model timesfm
    uv run python experiments/v09_round_vq_codebook/04_fm_zero_shot_per_client.py \\
        --seed 42 --model chronos_bolt_small
    uv run python experiments/v09_round_vq_codebook/04_fm_zero_shot_per_client.py \\
        --seed 42 --model chronos_t5_tiny

Per-seed argparse — multi-seed sweep is the executor's job.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch

from config import OUTPUT_DIR, RANDOM_SEED
from dataloader.per_client_split import build_per_client_splits
from utils.metrics import compute_hr, compute_mae, compute_mse, compute_pape

# Same model keys / factory as v04 03_fm_zero_shot.py and v05 05_recompute_mse_fm.py,
# plus the larger "upper" variants (same wrappers, just bigger checkpoints):
#   chronos_bolt_base  — Bolt family, larger than bolt_small (current best).
#   chronos_t5_small   — T5 family, one size up from t5_tiny.
#   timesfm_2          — TimesFM 2.0 (500M) vs the 1.0 (200M) default.
FM_FACTORY = {
    "chronos_bolt_small": ("ChronosForecaster", {"model_id": "amazon/chronos-bolt-small"}),
    "chronos_bolt_base":  ("ChronosForecaster", {"model_id": "amazon/chronos-bolt-base"}),
    "chronos_t5_tiny":    ("ChronosForecaster", {"model_id": "amazon/chronos-t5-tiny"}),
    "chronos_t5_small":   ("ChronosForecaster", {"model_id": "amazon/chronos-t5-small"}),
    "timesfm":            ("TimesFMForecaster", {}),
    "timesfm_2":          ("TimesFMForecaster", {"checkpoint_repo": "google/timesfm-2.0-500m-pytorch"}),
}


def _gpu_snapshot() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        used, free, total, util = (int(s.strip()) for s in out.strip().split(","))
        return {"used_MiB": used, "free_MiB": free, "total_MiB": total, "util_pct": util}
    except Exception:
        return {"cpu_only": not torch.cuda.is_available()}


def _build_forecaster(model_key: str):
    cls_name, kwargs = FM_FACTORY[model_key]
    if cls_name == "ChronosForecaster":
        from fm import ChronosForecaster
        return ChronosForecaster(**kwargs)
    if cls_name == "TimesFMForecaster":
        from fm import TimesFMForecaster
        return TimesFMForecaster(**kwargs)
    raise ValueError(f"unknown forecaster class: {cls_name}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=("v09 TSFM zero-shot baseline on the FedVQ per-client protocol "
                     "(114 apts, test split, per-client-mean aggregation).")
    )
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--model", required=True, choices=list(FM_FACTORY.keys()))
    ap.add_argument("--batch_size", type=int, default=64,
                    help="Per-call batch for the FM .forecast() - 64 is safe under 16 GB.")
    ap.add_argument("--output_namespace", type=str, default="v09_round_vq_codebook")
    args = ap.parse_args()

    # 114-apt per-client split (same cache as 02_fl_vq_dynamics.py).
    splits = build_per_client_splits(seed=args.seed)
    n_clients = len(splits)

    cell = f"fm_{args.model}"
    out_dir = OUTPUT_DIR / args.output_namespace / f"seed{args.seed}" / cell
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v09 FM] seed={args.seed}  model={args.model}  clients={n_clients}")
    gpu_start = _gpu_snapshot()
    print(f"[v09 FM] GPU @start: {gpu_start}")

    forecaster = _build_forecaster(args.model)
    print("[v09 FM] forecaster ready")
    gpu_after_load = _gpu_snapshot()
    print(f"[v09 FM] GPU after load: {gpu_after_load}")

    t0 = time.time()
    papes, maes, mses, hr1s, hr2s, hr3s = [], [], [], [], [], []
    n_test_windows = 0
    for apt, sp in splits.items():
        x_z = sp["test_x"]          # (Nte, 96) z-space
        y_z = sp["test_y"]          # (Nte, 24) z-space
        if x_z.shape[0] == 0:
            continue
        m_, s_ = float(sp["mean"]), float(sp["std"])
        # De-normalise z-space inputs back to raw kW for the FM (FM does its
        # own internal scaling; 02 measures PAPE in kW space, so we match).
        x_kw = (x_z * s_ + m_).astype(np.float32)
        y_true_kw = (y_z * s_ + m_).astype(np.float32)

        y_pred = []
        for i in range(0, x_kw.shape[0], args.batch_size):
            y_pred.append(forecaster.forecast(x_kw[i:i + args.batch_size]))
        y_pred_kw = np.concatenate(y_pred, axis=0).astype(np.float32)

        papes.append(float(compute_pape(y_true_kw, y_pred_kw)))
        maes.append(float(compute_mae(y_true_kw, y_pred_kw)))
        mses.append(float(compute_mse(y_true_kw, y_pred_kw)))
        hr1s.append(float(compute_hr(y_true_kw, y_pred_kw, tol=1)))
        hr2s.append(float(compute_hr(y_true_kw, y_pred_kw, tol=2)))
        hr3s.append(float(compute_hr(y_true_kw, y_pred_kw, tol=3)))
        n_test_windows += int(y_true_kw.shape[0])

    elapsed = time.time() - t0
    gpu_end = _gpu_snapshot()
    print(f"[v09 FM] GPU @end: {gpu_end}")

    # Per-client-mean aggregation — identical keys to 02's _eval_per_client.
    test_terminal = {
        "pape_mean":               float(np.mean(papes)) if papes else float("nan"),
        "pape_std_across_clients": float(np.std(papes, ddof=1)) if len(papes) > 1 else 0.0,
        "mae_mean":                float(np.mean(maes)) if maes else float("nan"),
        "mse_kw2_mean":            float(np.mean(mses)) if mses else float("nan"),
        "hr@1_mean":               float(np.mean(hr1s)) if hr1s else float("nan"),
        "hr@2_mean":               float(np.mean(hr2s)) if hr2s else float("nan"),
        "hr@3_mean":               float(np.mean(hr3s)) if hr3s else float("nan"),
        "n_clients":               int(len(papes)),
        "n_test_windows":          int(n_test_windows),
    }

    print(f"[v09 FM] test (per-client mean): PAPE={test_terminal['pape_mean']:.2f}  "
          f"HR@1={test_terminal['hr@1_mean']:.1f}  HR@2={test_terminal['hr@2_mean']:.1f}  "
          f"MAE={test_terminal['mae_mean']:.4f}  "
          f"({test_terminal['n_clients']} clients, {n_test_windows} windows)")
    print(f"[v09 FM] elapsed: {elapsed:.0f}s")

    result = {
        "cell": cell,
        "model": args.model,
        "model_factory": FM_FACTORY[args.model],
        "seed": int(args.seed),
        "n_clients": n_clients,
        "protocol": (
            "v09 per-client: 114-apt build_per_client_splits, evaluated on each "
            "apt's test split (last 20%), per-client-mean aggregation. FM fed raw "
            "kW (z-space test_x de-normalised via per-apt train mean/std)."
        ),
        "test_terminal": test_terminal,
        "elapsed_seconds": float(elapsed),
        "gpu_at_start": gpu_start,
        "gpu_after_load": gpu_after_load,
        "gpu_at_end": gpu_end,
    }
    with (out_dir / "result.json").open("w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[v09 FM] saved -> {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
