"""Train CLEAN NBEATSx — pure MAE, no bc-reg, no peak weight.

Why this exists:
    The v10 B2 checkpoint was trained with three contaminations:
      - peak_weighted_smooth_l1 (FedPM heritage, not NBEATSx canon)
      - peak-amplitude weight multiplier (alpha=2 on argmax position)
      - backcast → 0 regularization (warps stack residual flow)
    Probing that checkpoint conflates "does h_generic encode peak in NBEATSx"
    with "does it encode peak under those distortions". This script produces a
    clean reference checkpoint so we can isolate the architectural answer.

Setup:
    - Pooled centralized training over N train households (z-normalized per
      household so each household's local stats are preserved).
    - Loss: F.l1_loss(y_hat, y_z) only. Nothing else.
    - Model: MinimalNBEATSx (identical to v10 b2 architecture & layer names,
      so the probe script can swap checkpoints freely).
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

from config import (
    HORIZON,
    INPUT_SIZE,
    OUTPUT_DIR,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
)
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from models.nbeatsx import MinimalNBEATSx
from utils.metrics import seven_axis_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_APTS = [
    "Apt3", "Apt4", "Apt5", "Apt6", "Apt8",
    "Apt9", "Apt10", "Apt11", "Apt14", "Apt15",
]


def build_household_datasets(
    apts: list[str], stride_train: int = 1, stride_test: int = HORIZON
) -> tuple[list[HouseholdDataset], list[HouseholdDataset], list[HouseholdDataset], dict]:
    train_sets, val_sets, test_sets = [], [], []
    norm = {}
    for apt in apts:
        series = load_apartment_hourly(apt).values.astype(np.float32)
        n = len(series)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
        mean = float(series[:train_end].mean())
        std = float(series[:train_end].std()) if series[:train_end].std() > 1e-8 else 1.0

        train_sets.append(HouseholdDataset(series[:train_end], mean, std, stride=stride_train))
        val_sets.append(HouseholdDataset(series[train_end:val_end], mean, std, stride=stride_train))
        test_sets.append(
            HouseholdDataset(series[max(0, val_end - INPUT_SIZE):], mean, std, stride=stride_test)
        )
        norm[apt] = {"mean": mean, "std": std,
                     "n_train": len(train_sets[-1]),
                     "n_val": len(val_sets[-1]),
                     "n_test": len(test_sets[-1])}
    return train_sets, val_sets, test_sets, norm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apts", nargs="+", default=DEFAULT_APTS)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--tag", default="clean_mae")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = OUTPUT_DIR / "v11_clean_pretrain" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data] {len(args.apts)} households: {args.apts}")
    train_sets, val_sets, test_sets, norm = build_household_datasets(args.apts)

    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=args.batch,
                              shuffle=True, drop_last=False)
    val_loader = DataLoader(ConcatDataset(val_sets), batch_size=args.batch,
                            shuffle=False, drop_last=False)
    test_loaders = [
        DataLoader(ts, batch_size=args.batch, shuffle=False) for ts in test_sets
    ]
    n_train = sum(len(ds) for ds in train_sets)
    n_val = sum(len(ds) for ds in val_sets)
    print(f"[data] windows: train={n_train}  val={n_val}")

    model = MinimalNBEATSx().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"[train] CLEAN MAE, no bc-reg, no peak weight, {args.epochs} epochs max")
    best_val_pape = float("inf")
    best_state = None
    bad = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_hat, _ = model(x)
            loss = F.l1_loss(y_hat, y)        # PURE MAE
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_sum += float(loss.item())
            n_batches += 1

        model.eval()
        v_true_z, v_pred_z = [], []
        with torch.no_grad():
            for x, y in val_loader:
                y_hat, _ = model(x.to(DEVICE))
                v_true_z.append(y.numpy())
                v_pred_z.append(y_hat.cpu().numpy())
        # val metrics in z-space (no need to denorm — pooled, mixed apts)
        vt = np.concatenate(v_true_z, axis=0)
        vp = np.concatenate(v_pred_z, axis=0)
        val_metrics = seven_axis_metrics(vt, vp)

        rec = {
            "epoch": epoch,
            "train_loss": train_loss_sum / n_batches,
            "val_pape": val_metrics["pape"],
            "val_mae": val_metrics["mae"],
            "val_hr@1": val_metrics["hr@1"],
            "wallclock_s": round(time.time() - t0, 1),
        }
        history.append(rec)

        improved = val_metrics["pape"] < best_val_pape - 1e-6
        if improved:
            best_val_pape = val_metrics["pape"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        flag = " *" if improved else ""
        print(
            f"  ep{epoch:02d}  loss={rec['train_loss']:.4f}  "
            f"val_mae={rec['val_mae']:.4f}  val_pape={rec['val_pape']:.2f}  "
            f"val_hr1={rec['val_hr@1']:.1f}  ({rec['wallclock_s']}s){flag}"
        )
        if bad >= args.patience:
            print(f"  early stop @ epoch {epoch}")
            break

    # save best
    ckpt_path = out_dir / "best.pt"
    torch.save(best_state, ckpt_path)
    print(f"[save] {ckpt_path}")

    # per-apt test (denormalized so PAPE is in kW units)
    model.load_state_dict(best_state)
    model.eval()
    per_apt = {}
    for apt, tl in zip(args.apts, test_loaders):
        yt, yp = [], []
        with torch.no_grad():
            for x, y in tl:
                y_hat, _ = model(x.to(DEVICE))
                yt.append(y.numpy()); yp.append(y_hat.cpu().numpy())
        m, s = norm[apt]["mean"], norm[apt]["std"]
        yt_kw = np.concatenate(yt, axis=0) * s + m
        yp_kw = np.concatenate(yp, axis=0) * s + m
        per_apt[apt] = seven_axis_metrics(yt_kw, yp_kw)

    papes = np.array([per_apt[a]["pape"] for a in args.apts if a in per_apt])
    hr1s = np.array([per_apt[a]["hr@1"] for a in args.apts if a in per_apt])
    print(f"\n[test] per-apt PAPE mean ± std: {papes.mean():.2f} ± {papes.std():.2f}")
    print(f"[test] per-apt HR@1 mean ± std:  {hr1s.mean():.1f} ± {hr1s.std():.1f}")

    with open(out_dir / "training_log.json", "w") as fh:
        json.dump(
            {
                "config": vars(args),
                "norm": norm,
                "history": history,
                "per_apt_test": per_apt,
                "summary": {
                    "pape_mean": float(papes.mean()),
                    "pape_std": float(papes.std()),
                    "hr@1_mean": float(hr1s.mean()),
                    "hr@1_std": float(hr1s.std()),
                },
            },
            fh,
            indent=2,
        )
    print(f"[done] log: {out_dir / 'training_log.json'}")


if __name__ == "__main__":
    main()
