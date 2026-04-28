"""v04 NF baseline training (centralised pooled, no FL).

One invocation = one seed × one model. NF baselines are trained
centrally on the 80 train apts pooled (per-apt z-norm, stride=1) with
plain MAE loss — same protocol as v02 02_train_arms.py for the T0 arm,
just substituting DLinear / NHITS / Crossformer for MinimalNBEATSx.

Cold inference: warm-start z-norm + stride=24 + frozen forward,
identical to v02 04_coldstart_eval.py (no Peak-VQ correction, raw
forecast).

Per-seed CLI:

    uv run python experiments/v04_full_baseline_comparison/02_nf_train.py \\
        --seed 42 --model dlinear
    uv run python experiments/v04_full_baseline_comparison/02_nf_train.py \\
        --seed 42 --model nhits
    uv run python experiments/v04_full_baseline_comparison/02_nf_train.py \\
        --seed 42 --model crossformer

Output: outputs/v04_full_baseline_comparison/seed{S}/nf_{model}/{result.json, best.pt}.
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
from torch.utils.data import ConcatDataset, DataLoader

from config import HORIZON, OUTPUT_DIR, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO
from dataloader.splits import load_v02_split
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from models.crossformer import Crossformer
from models.dlinear import DLinear
from models.nhits import NHITS
from utils.metrics import compute_hr, compute_mae, compute_pape

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V04_OUT_ROOT = OUTPUT_DIR / "v04_full_baseline_comparison"

NF_MODELS = {"dlinear": DLinear, "nhits": NHITS, "crossformer": Crossformer}


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


def build_pooled_loaders(train_apts: list[str], batch: int):
    """Pooled training across 80 train apts; per-apt z-norm + stride=1 (v02 02 protocol)."""
    train_sets, val_sets, norm, present = [], [], {}, []
    for apt in train_apts:
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            continue
        n = len(series)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
        m = float(series[:train_end].mean())
        s = float(series[:train_end].std()) if series[:train_end].std() > 1e-8 else 1.0
        train_sets.append(HouseholdDataset(series[:train_end], m, s, stride=1))
        val_sets.append(HouseholdDataset(series[train_end:val_end], m, s, stride=1))
        norm[apt] = {"mean": m, "std": s}
        present.append(apt)
    train_loader = DataLoader(
        ConcatDataset(train_sets), batch_size=batch, shuffle=True, drop_last=False
    )
    return train_sets, val_sets, norm, train_loader, present


def eval_per_apt_kw(model, val_sets, present, norm, batch, use_amp) -> dict:
    """Per-apt val eval, denormalised to kW. Same shape as v02 02 eval_per_apt."""
    amp_ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
               if use_amp and DEVICE.type == "cuda" else _NullCtx())
    model.eval()
    apt_idx, true_z, pred_z = [], [], []
    with torch.no_grad():
        for ai, ds in enumerate(val_sets):
            for x, y in DataLoader(ds, batch_size=batch, shuffle=False):
                with amp_ctx:
                    y_hat = model(x.to(DEVICE, non_blocking=True))
                true_z.append(y.numpy())
                pred_z.append(y_hat.float().cpu().numpy())
                apt_idx.append(np.full(len(y), ai, dtype=np.int32))
    t_z = np.concatenate(true_z, 0); p_z = np.concatenate(pred_z, 0)
    a = np.concatenate(apt_idx, 0)
    means = np.array([norm[ap]["mean"] for ap in present])
    stds = np.array([norm[ap]["std"] for ap in present])
    t_kw = t_z * stds[a, None] + means[a, None]
    p_kw = p_z * stds[a, None] + means[a, None]
    return {
        "pape": float(compute_pape(t_kw, p_kw)),
        "hr@1": float(compute_hr(t_kw, p_kw, tol=1)),
        "hr@2": float(compute_hr(t_kw, p_kw, tol=2)),
        "mae": float(compute_mae(t_kw, p_kw)),
    }


def evaluate_cold_kw(model, cold_apts, batch, use_amp) -> dict:
    """Cold inference (warm-start z-norm, stride=24, denormalise to kW)."""
    amp_ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
               if use_amp and DEVICE.type == "cuda" else _NullCtx())
    model.eval()
    true_chunks, pred_chunks, mean_chunks, std_chunks = [], [], [], []
    n_apts_seen = 0
    for apt in cold_apts:
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            continue
        n = len(series)
        train_end = int(n * TRAIN_RATIO)
        seg = series[:train_end]
        m_ = float(seg.mean())
        s_ = float(seg.std()) if seg.std() > 1e-8 else 1.0
        ds = HouseholdDataset(seg, m_, s_, stride=HORIZON)
        if len(ds) == 0:
            continue
        n_apts_seen += 1
        for x, y in DataLoader(ds, batch_size=batch, shuffle=False):
            with torch.no_grad(), amp_ctx:
                y_hat = model(x.to(DEVICE, non_blocking=True))
            true_chunks.append(y.numpy())
            pred_chunks.append(y_hat.float().cpu().numpy())
            mean_chunks.append(np.full(len(y), m_, dtype=np.float32))
            std_chunks.append(np.full(len(y), s_, dtype=np.float32))
    if not true_chunks:
        return {"pape": float("nan"), "hr@1": float("nan"), "hr@2": float("nan"),
                "mae": float("nan"), "n_cold_windows": 0, "n_cold_apts": 0}
    t_z = np.concatenate(true_chunks, 0); p_z = np.concatenate(pred_chunks, 0)
    m_arr = np.concatenate(mean_chunks, 0); s_arr = np.concatenate(std_chunks, 0)
    t_kw = t_z * s_arr[:, None] + m_arr[:, None]
    p_kw = p_z * s_arr[:, None] + m_arr[:, None]
    return {
        "pape": float(compute_pape(t_kw, p_kw)),
        "hr@1": float(compute_hr(t_kw, p_kw, tol=1)),
        "hr@2": float(compute_hr(t_kw, p_kw, tol=2)),
        "mae": float(compute_mae(t_kw, p_kw)),
        "n_cold_windows": int(t_z.shape[0]),
        "n_cold_apts": n_apts_seen,
    }


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main() -> None:
    ap = argparse.ArgumentParser(description="v04 NF baseline (centralised pooled training).")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--model", required=True, choices=list(NF_MODELS.keys()))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    use_amp = not args.no_amp
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    sp = load_v02_split(args.seed)
    train_apts, cold_apts = sp["train"], sp["cold"]
    out_dir = V04_OUT_ROOT / f"seed{args.seed}" / f"nf_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v04 NF] seed={args.seed}  model={args.model}  epochs={args.epochs}  batch={args.batch_size}  amp={use_amp}")
    gpu_start = _gpu_snapshot()
    print(f"[v04 NF] GPU @start: {gpu_start}")

    train_sets, val_sets, norm, train_loader, present = build_pooled_loaders(train_apts, args.batch_size)
    n_train_windows = sum(len(ds) for ds in train_sets)
    print(f"[v04 NF] {len(present)} apts, {n_train_windows} train windows")

    model = NF_MODELS[args.model]().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[v04 NF] params: {n_params}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
               if use_amp and DEVICE.type == "cuda" else _NullCtx())

    best_val_mae, best_state, bad, history = float("inf"), None, 0, []
    t_total = time.time()
    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        model.train()
        loss_sum, n_batches = 0.0, 0
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                y_hat = model(x)
                loss = F.l1_loss(y_hat, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()); n_batches += 1
        m = eval_per_apt_kw(model, val_sets, present, norm, args.batch_size, use_amp)
        rec = {"epoch": epoch, "train_loss": loss_sum / n_batches,
               "val_mae": m["mae"], "val_pape": m["pape"], "val_hr@1": m["hr@1"],
               "wall_s": round(time.time() - t_ep, 1)}
        history.append(rec)
        improved = m["mae"] < best_val_mae - 1e-6
        flag = " *" if improved else ""
        if improved:
            best_val_mae = m["mae"]
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

    # Reload best for cold inference.
    if best_state is not None:
        model.load_state_dict(best_state)
    cold_metrics = evaluate_cold_kw(model, cold_apts, args.batch_size, use_amp)
    gpu_end = _gpu_snapshot()
    print(f"[v04 NF] GPU @end: {gpu_end}")
    print(f"[v04 NF] cold: PAPE={cold_metrics['pape']:.2f}  HR@1={cold_metrics['hr@1']:.1f}  HR@2={cold_metrics['hr@2']:.1f}")
    print(f"[v04 NF] train elapsed: {train_elapsed:.0f}s ({train_elapsed/60:.1f} min)")

    torch.save(best_state, out_dir / "best.pt")
    with open(out_dir / "result.json", "w") as fh:
        json.dump({
            "algorithm": f"nf_{args.model}",
            "model": args.model,
            "n_params": n_params,
            "seed": int(args.seed),
            "config": {"epochs": args.epochs, "patience": args.patience, "lr": args.lr,
                       "batch_size": args.batch_size, "weight_decay": args.weight_decay,
                       "use_amp": use_amp},
            "history": history,
            "best_val_mae": best_val_mae,
            "cold_metrics": cold_metrics,
            "n_train_windows": n_train_windows,
            "n_train_apts": len(present),
            "elapsed_seconds": train_elapsed,
            "gpu_at_start": gpu_start,
            "gpu_at_end": gpu_end,
        }, fh, indent=2)
    print(f"[v04 NF] saved -> {out_dir}")


if __name__ == "__main__":
    main()
