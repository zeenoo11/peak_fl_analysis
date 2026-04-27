"""Train arms T0/T2 on the v02 80-train-apt split (per-seed).

Adapter over experiments/v01_peak_from_latent/01_train_arms.py:
    - reads the v02 stratified split via load_v02_split(seed) instead of v10's;
    - writes to outputs/v02_fl_8020_ratio/seed{seed}/{T0,T2}/best.pt;
    - drops T3 (h_concat) — v02 scope is T0 and T2 only (plans/v02-01_fl_8020_ratio.md).

T0: MinimalNBEATSx with pure MAE.        — peak_aux OFF (E1 baseline)
T2: NBEATSxAux(latent_source='h_generic') with MAE + lambda * peak_aux.

Per-seed invocation:
    uv run python experiments/v02_fl_8020_ratio/02_train_arms.py --seed 42
    uv run python experiments/v02_fl_8020_ratio/02_train_arms.py --seed 123
    uv run python experiments/v02_fl_8020_ratio/02_train_arms.py --seed 7
"""

from __future__ import annotations

import argparse
import json
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

from config import OUTPUT_DIR, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO
from dataloader.splits import load_v02_split
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from models.nbeatsx import MinimalNBEATSx
from models.nbeatsx_aux import NBEATSxAux
from models.peak_aux_head import peak_aux_loss
from utils.metrics import seven_axis_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V02_OUT_ROOT = OUTPUT_DIR / "v02_fl_8020_ratio"


def build_loaders(apts: list[str], batch: int):
    train_sets, val_sets, norm, present_apts = [], [], {}, []
    for apt in apts:
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            print(f"  [skip] {apt}: missing")
            continue
        n = len(series)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
        mean = float(series[:train_end].mean())
        std = float(series[:train_end].std()) if series[:train_end].std() > 1e-8 else 1.0
        train_sets.append(HouseholdDataset(series[:train_end], mean, std, stride=1))
        val_sets.append(HouseholdDataset(series[train_end:val_end], mean, std, stride=1))
        norm[apt] = {"mean": mean, "std": std}
        present_apts.append(apt)
    train_loader = DataLoader(
        ConcatDataset(train_sets), batch_size=batch, shuffle=True, drop_last=False
    )
    return train_sets, val_sets, norm, train_loader, present_apts


def eval_per_apt(model, val_sets, present_apts, norm, batch, use_aux):
    """Per-apt val eval. Returns 7-axis metrics in kW units."""
    model.eval()
    apt_idx_arr, true_chunks, pred_chunks = [], [], []
    with torch.no_grad():
        for ai, ds in enumerate(val_sets):
            for x, y in DataLoader(ds, batch_size=batch, shuffle=False):
                if use_aux:
                    y_hat, _, _ = model(x.to(DEVICE))
                else:
                    y_hat, _ = model(x.to(DEVICE))
                true_chunks.append(y.numpy())
                pred_chunks.append(y_hat.cpu().numpy())
                apt_idx_arr.append(np.full(len(y), ai, dtype=np.int32))
    t_z = np.concatenate(true_chunks, axis=0)
    p_z = np.concatenate(pred_chunks, axis=0)
    a_idx = np.concatenate(apt_idx_arr, axis=0)
    means = np.array([norm[a]["mean"] for a in present_apts])
    stds = np.array([norm[a]["std"] for a in present_apts])
    t_kw = t_z * stds[a_idx, None] + means[a_idx, None]
    p_kw = p_z * stds[a_idx, None] + means[a_idx, None]
    return seven_axis_metrics(t_kw, p_kw)


def train_arm(
    arm: str,
    apts: list[str],
    epochs: int,
    lr: float,
    batch: int,
    patience: int,
    lam: float,
    seed: int,
    out_root: Path,
) -> None:
    torch.manual_seed(seed); np.random.seed(seed)
    out_dir = out_root / arm
    out_dir.mkdir(parents=True, exist_ok=True)

    train_sets, val_sets, norm, train_loader, present = build_loaders(apts, batch)
    n_train = sum(len(d) for d in train_sets)
    print(f"[{arm}] {len(present)} apts, {n_train} train windows")

    use_aux = arm == "T2"
    if arm == "T2":
        model = NBEATSxAux(latent_source="h_generic").to(DEVICE)
    elif arm == "T0":
        model = MinimalNBEATSx().to(DEVICE)
    else:
        raise ValueError(f"unsupported arm for v02: {arm} (expected T0 or T2)")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_mae, best_val_pape, best_state, bad, history = float("inf"), float("inf"), None, 0, []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, aux_sum, n = 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if use_aux:
                y_hat, _, (amp_p, hr_p) = model(x)
                main = F.l1_loss(y_hat, y)
                aux = peak_aux_loss(amp_p, hr_p, y)
                loss = main + lam * aux
                aux_sum += float(aux.item())
            else:
                y_hat, _ = model(x)
                loss = F.l1_loss(y_hat, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss.item()); n += 1

        m = eval_per_apt(model, val_sets, present, norm, batch, use_aux)
        rec = {
            "epoch": epoch,
            "train_loss": loss_sum / n,
            "val_mae": m["mae"],
            "val_pape": m["pape"],
            "val_hr@1": m["hr@1"],
            "wall_s": round(time.time() - t0, 1),
        }
        if use_aux:
            rec["train_aux"] = aux_sum / n
        history.append(rec)

        improved = m["mae"] < best_val_mae - 1e-6
        if improved:
            best_val_mae = m["mae"]
            best_val_pape = m["pape"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        flag = " *" if improved else ""
        msg = (
            f"  ep{epoch:02d} loss={rec['train_loss']:.4f} "
            f"val_mae={rec['val_mae']:.4f} val_pape={rec['val_pape']:.2f} "
            f"hr1={rec['val_hr@1']:.1f} ({rec['wall_s']}s){flag}"
        )
        if use_aux:
            msg += f"  aux={rec['train_aux']:.4f}"
        print(msg)
        if bad >= patience:
            print(f"  early stop @ ep {epoch}")
            break

    torch.save(best_state, out_dir / "best.pt")
    with open(out_dir / "training_log.json", "w") as fh:
        json.dump(
            {
                "arm": arm,
                "seed": seed,
                "split_version": "v02",
                "lam": lam,
                "norm": norm,
                "history": history,
                "n_train_windows": n_train,
                "n_apts": len(present),
                "best_val_mae": best_val_mae,
                "best_val_pape": best_val_pape,
            },
            fh,
            indent=2,
        )
    print(f"[{arm}] saved best.pt; best_val_mae={best_val_mae:.4f} best_val_pape={best_val_pape:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train T0/T2 on v02 80-train-apt split for one seed.")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED, help="Seed for split lookup AND model init.")
    ap.add_argument("--arms", nargs="+", default=["T0", "T2"], choices=["T0", "T2"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.3)
    args = ap.parse_args()

    apts = load_v02_split(args.seed)["train"]
    out_root = V02_OUT_ROOT / f"seed{args.seed}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[setup] seed={args.seed}; {len(apts)} train apts; arms: {args.arms}")
    print(f"[setup] device={DEVICE}; out_root={out_root}")
    for arm in args.arms:
        print(f"\n========== {arm} (seed {args.seed}) ==========")
        train_arm(
            arm,
            apts,
            args.epochs,
            args.lr,
            args.batch,
            args.patience,
            args.lam,
            args.seed,
            out_root,
        )


if __name__ == "__main__":
    main()
