"""v09 NF baseline — centralized-pooled training on the FedVQ per-client protocol.

(한글 요약)
v04 의 NF 베이스라인(DLinear / NHITS / Crossformer)은 80 train apts 를 pooled 로
centralized 학습하고 20 cold apts 에 inference 한다 — v09 round-level FedVQ 와
가구 집합·평가 구간·집계가 모두 다르다. 본 스크립트는 그 세 축을 v09 에 맞춘
**centralized 상한선**이다 (FL 없음; 모든 데이터를 한 곳에 모아 학습):

  1. 데이터  = ``build_per_client_splits(seed)`` 114 가구의 ``train_x``/``train_y``
     를 한 풀로 합쳐 centralized 학습. 이는 FedVQ(02) 가 federated 로 보는 것과
     **정확히 동일한 train 윈도우**(z-space, stride=24)다 — federation 만 제거한
     공정한 centralized 대조군.
  2. 평가     = 각 가구의 **test split(뒤 20%)**, kW 공간.
  3. 집계     = 가구별 PAPE/HR/MAE/MSE → **가구 평균** (02 의 ``_eval_per_client``
     와 동일 key: pape_mean, pape_std_across_clients, mae_mean, mse_kw2_mean,
     hr@{1,2,3}_mean, n_clients).

학습 프로토콜은 v04 02_nf_train.py 와 동일 (MAE loss, Adam, bf16 autocast,
early stop on val MAE). NF 모델 생성자는 전부 INPUT_SIZE=96 / HORIZON=24 기본값.

Output (``outputs/v09_round_vq_codebook/seed{S}/nf_{model}/``):
  - ``result.json`` — history + best_val_mae + ``test_terminal`` (02 호환 key)
  - ``best.pt``     — best-val state dict

Per-seed argparse — multi-seed sweep is the executor's job.

CLI:

    uv run python experiments/v09_round_vq_codebook/05_nf_centralized_per_client.py \\
        --seed 42 --model dlinear
    uv run python experiments/v09_round_vq_codebook/05_nf_centralized_per_client.py \\
        --seed 42 --model nhits
    uv run python experiments/v09_round_vq_codebook/05_nf_centralized_per_client.py \\
        --seed 42 --model crossformer
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import OUTPUT_DIR, RANDOM_SEED
from dataloader.per_client_split import build_per_client_splits
from models.crossformer import Crossformer
from models.dlinear import DLinear
from models.nhits import NHITS
from utils.metrics import compute_hr, compute_mae, compute_mse, compute_pape

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V09_OUT_ROOT = OUTPUT_DIR / "v09_round_vq_codebook"

NF_MODELS = {"dlinear": DLinear, "nhits": NHITS, "crossformer": Crossformer}


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def _amp_ctx(use_amp: bool):
    if use_amp and DEVICE.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return _NullCtx()


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


@torch.no_grad()
def _eval_per_client(
    model,
    splits: dict[str, dict],
    split_key: str,
    *,
    batch_size: int,
    use_amp: bool,
) -> dict[str, float]:
    """Across-client mean of per-apt PAPE/HR/MAE/MSE on `split_key` ∈ {val,test}.

    Mirrors 02_fl_vq_dynamics.py's _eval_per_client exactly (same denorm,
    same per-client-mean aggregation, same keys) but calls the NF model's
    plain-tensor forward instead of the dict-returning NBEATSxAuxVQ.
    """
    model.eval()
    papes, maes, mses, hr1s, hr2s, hr3s = [], [], [], [], [], []
    for _apt, sp in splits.items():
        x = sp[f"{split_key}_x"]
        y = sp[f"{split_key}_y"]
        if x.shape[0] == 0:
            continue
        m_, s_ = float(sp["mean"]), float(sp["std"])
        yhat_chunks = []
        for i in range(0, int(x.shape[0]), batch_size):
            xb = torch.from_numpy(x[i : i + batch_size]).to(DEVICE, non_blocking=True)
            with _amp_ctx(use_amp):
                y_hat = model(xb)
            yhat_chunks.append(y_hat.float().cpu().numpy())
        y_hat_z = np.concatenate(yhat_chunks, axis=0).astype(np.float32)
        y_true_kw = (y * s_ + m_).astype(np.float32)
        y_hat_kw = (y_hat_z * s_ + m_).astype(np.float32)
        papes.append(float(compute_pape(y_true_kw, y_hat_kw)))
        maes.append(float(compute_mae(y_true_kw, y_hat_kw)))
        mses.append(float(compute_mse(y_true_kw, y_hat_kw)))
        hr1s.append(float(compute_hr(y_true_kw, y_hat_kw, tol=1)))
        hr2s.append(float(compute_hr(y_true_kw, y_hat_kw, tol=2)))
        hr3s.append(float(compute_hr(y_true_kw, y_hat_kw, tol=3)))
    return {
        "pape_mean":               float(np.mean(papes)) if papes else float("nan"),
        "pape_std_across_clients": float(np.std(papes, ddof=1)) if len(papes) > 1 else 0.0,
        "mae_mean":                float(np.mean(maes)) if maes else float("nan"),
        "mse_kw2_mean":            float(np.mean(mses)) if mses else float("nan"),
        "hr@1_mean":               float(np.mean(hr1s)) if hr1s else float("nan"),
        "hr@2_mean":               float(np.mean(hr2s)) if hr2s else float("nan"),
        "hr@3_mean":               float(np.mean(hr3s)) if hr3s else float("nan"),
        "n_clients":               int(len(papes)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=("v09 NF baseline: centralized-pooled training on the 114-apt "
                     "per-client splits, per-client-mean test evaluation.")
    )
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--model", required=True, choices=list(NF_MODELS.keys()))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--output_namespace", type=str, default="v09_round_vq_codebook")
    args = ap.parse_args()

    use_amp = not args.no_amp
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cell = f"nf_{args.model}"
    out_dir = OUTPUT_DIR / args.output_namespace / f"seed{args.seed}" / cell
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v09 NF] seed={args.seed}  model={args.model}  epochs={args.epochs}  "
          f"batch={args.batch_size}  amp={use_amp}")
    gpu_start = _gpu_snapshot()
    print(f"[v09 NF] GPU @start: {gpu_start}")

    # 114-apt per-client split (same cache as 02_fl_vq_dynamics.py).
    splits = build_per_client_splits(seed=args.seed)
    n_clients = len(splits)

    # Pool every apt's train windows — the same z-space windows FedVQ trains on,
    # just centralized instead of federated.
    train_x = np.concatenate([sp["train_x"] for sp in splits.values()], axis=0)
    train_y = np.concatenate([sp["train_y"] for sp in splits.values()], axis=0)
    n_train_windows = int(train_x.shape[0])
    print(f"[v09 NF] {n_clients} apts, {n_train_windows} pooled train windows")
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.batch_size, shuffle=True, drop_last=False,
    )

    model = NF_MODELS[args.model]().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[v09 NF] params: {n_params}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_mae, best_state, bad, history = float("inf"), None, 0, []
    t_total = time.time()
    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        model.train()
        loss_sum, n_batches = 0.0, 0
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _amp_ctx(use_amp):
                y_hat = model(x)
                loss = F.l1_loss(y_hat, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            n_batches += 1
        val = _eval_per_client(model, splits, "val", batch_size=args.batch_size, use_amp=use_amp)
        rec = {"epoch": epoch, "train_loss": loss_sum / max(n_batches, 1),
               "val_mae": val["mae_mean"], "val_pape": val["pape_mean"],
               "val_hr@1": val["hr@1_mean"], "wall_s": round(time.time() - t_ep, 1)}
        history.append(rec)
        improved = val["mae_mean"] < best_val_mae - 1e-6
        flag = " *" if improved else ""
        if improved:
            best_val_mae = val["mae_mean"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        print(f"  ep{epoch:02d} loss={rec['train_loss']:.4f} val_mae={rec['val_mae']:.4f} "
              f"val_pape={rec['val_pape']:.2f} hr1={rec['val_hr@1']:.1f} ({rec['wall_s']}s){flag}")
        if bad >= args.patience:
            print(f"  early stop @ ep {epoch}")
            break

    train_elapsed = time.time() - t_total

    # Reload best for terminal test eval.
    if best_state is not None:
        model.load_state_dict(best_state)
    test_terminal = _eval_per_client(
        model, splits, "test", batch_size=args.batch_size, use_amp=use_amp,
    )
    gpu_end = _gpu_snapshot()
    print(f"[v09 NF] GPU @end: {gpu_end}")
    print(f"[v09 NF] test (per-client mean): PAPE={test_terminal['pape_mean']:.2f}  "
          f"HR@1={test_terminal['hr@1_mean']:.1f}  HR@2={test_terminal['hr@2_mean']:.1f}  "
          f"MAE={test_terminal['mae_mean']:.4f}")
    print(f"[v09 NF] train elapsed: {train_elapsed:.0f}s ({train_elapsed/60:.1f} min)")

    torch.save(best_state, out_dir / "best.pt")
    result = {
        "cell": cell,
        "model": args.model,
        "seed": int(args.seed),
        "n_clients": n_clients,
        "n_params": int(n_params),
        "n_train_windows": n_train_windows,
        "protocol": (
            "v09 centralized-pooled: 114-apt build_per_client_splits train windows "
            "pooled (z-space, stride=24, no FL), MAE training with early stop on "
            "per-client-mean val MAE, evaluated on each apt's test split (last 20%), "
            "per-client-mean aggregation. Centralized upper bound vs FedVQ (02)."
        ),
        "config": {"epochs": args.epochs, "patience": args.patience, "lr": args.lr,
                   "batch_size": args.batch_size, "weight_decay": args.weight_decay,
                   "use_amp": use_amp},
        "history": history,
        "best_val_mae": float(best_val_mae),
        "test_terminal": test_terminal,
        "elapsed_seconds": float(train_elapsed),
        "gpu_at_start": gpu_start,
        "gpu_at_end": gpu_end,
    }
    with (out_dir / "result.json").open("w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[v09 NF] saved -> {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
