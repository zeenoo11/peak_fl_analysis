"""v03 MVP — F2a head-only K-shot dry run on a single cold apt.

Quick sanity check to decide whether to commit to a full v03 sweep.
Loads v02 frozen artifacts (NBEATSxAux T2 backbone + Peak-VQ codebook),
takes one cold apt, K-shot adapts the aux head only (F2a), and compares
to the no-adapt baseline on the SAME held-out segment.

For F2a specifically: the backbone is frozen, so h_g and y_hat_base are
unchanged → c* (routing) is unchanged → V0 cluster offset term is
unchanged. The only thing that moves is the W1a Gaussian template
(because pred_amp / pred_hr change). So the ΔPAPE this script reports
is the contribution of cold-side aux head fine-tuning isolated to W1a.

CLI (defaults give the smallest possible cell):

    uv run python experiments/v03_kshot_pfl/00_mvp_dryrun.py

    uv run python experiments/v03_kshot_pfl/00_mvp_dryrun.py \\
        --seed 42 --cold_apt Apt1 --epochs 30 --lr 1e-3

Output:
    outputs/v03_kshot_pfl/mvp/seed{S}/{apt}/F2a_result.json
"""

from __future__ import annotations

import argparse
import copy
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
from torch.utils.data import DataLoader

from config import HORIZON, INPUT_SIZE, OUTPUT_DIR, TRAIN_RATIO
from dataloader.splits import load_v02_split
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from eval.cold_helpers import (
    OPERATING_POINTS,
    gauss_template,
    metrics_z_to_kw,
    route_R0,
)
from models.nbeatsx_aux import NBEATSxAux
from models.peak_aux_head import peak_aux_loss
from probes.peak_descriptor import extract_key

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V02_OUT_ROOT = OUTPUT_DIR / "v02_fl_8020_ratio"
V03_MVP_ROOT = OUTPUT_DIR / "v03_kshot_pfl" / "mvp"

K_SHOT_DAYS = 30
BUFFER_DAYS = 7


def _build_window_arrays(
    apt: str, stride: int = 24
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (x_z [N, L], y_z [N, H], mean, std) for the cold apt.

    Mirrors gather_cold's z-norm convention: per-apt mean / std on the
    full first 70% segment, sliding stride=24 windows on that segment.
    """
    series = load_apartment_hourly(apt).values.astype(np.float32)
    n = len(series)
    train_end = int(n * TRAIN_RATIO)
    seg = series[:train_end]
    m_ = float(seg.mean())
    s_ = float(seg.std()) if seg.std() > 1e-8 else 1.0
    ds = HouseholdDataset(seg, m_, s_, stride=stride)
    xs, ys = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        xs.append(x.numpy())
        ys.append(y.numpy())
    return np.stack(xs), np.stack(ys), m_, s_


def _forward_all(
    model: NBEATSxAux, x_z: np.ndarray, batch: int = 256
) -> dict[str, np.ndarray]:
    """Frozen no_grad forward over a stack of z-normed input windows."""
    model.eval()
    h_chunks, yhat_chunks, amp_chunks, hr_chunks = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(x_z), batch):
            x = torch.from_numpy(x_z[i : i + batch]).to(DEVICE)
            y_hat, hiddens, (amp_p, hr_p) = model(x)
            h_chunks.append(hiddens["h_generic"].cpu().numpy())
            yhat_chunks.append(y_hat.cpu().numpy())
            amp_chunks.append(amp_p.cpu().numpy().reshape(-1))
            hr_chunks.append(hr_p.argmax(dim=1).cpu().numpy())
    return {
        "h_g": np.concatenate(h_chunks).astype(np.float32),
        "y_hat_z": np.concatenate(yhat_chunks).astype(np.float32),
        "pred_amp": np.concatenate(amp_chunks).astype(np.float32),
        "pred_hr": np.concatenate(hr_chunks).astype(np.int64),
    }


def _evaluate_with_W5(
    fwd: dict[str, np.ndarray],
    y_true_z: np.ndarray,
    mean_arr: np.ndarray,
    std_arr: np.ndarray,
    cold_cluster: np.ndarray,
    offsets: np.ndarray,
) -> dict:
    """Baseline + (HR-pres, PAPE-aggr) op-points from a forward dict."""
    base = metrics_z_to_kw(y_true_z, fwd["y_hat_z"], mean_arr, std_arr)
    cluster_offset = offsets[cold_cluster]  # [N, 24]
    op_results = {}
    for op_name, op in OPERATING_POINTS.items():
        g = gauss_template(fwd["pred_hr"], fwd["pred_amp"], sigma=op["sigma"])
        corrected_z = (
            fwd["y_hat_z"]
            + op["alpha_v0"] * cluster_offset
            + op["alpha_w1"] * g
        ).astype(np.float32)
        op_results[op_name] = {
            "alpha_v0": op["alpha_v0"],
            "alpha_w1": op["alpha_w1"],
            "metrics": metrics_z_to_kw(y_true_z, corrected_z, mean_arr, std_arr),
        }
    return {"baseline": base, "operating_points": op_results}


def _aux_acc(fwd: dict[str, np.ndarray], y_true_z: np.ndarray) -> dict:
    """Aux head diagnostic — peak-hour top-1 / ±1 / ±2 vs ground truth."""
    cold_true_hr = y_true_z.argmax(axis=1)
    return {
        "top1": float((fwd["pred_hr"] == cold_true_hr).mean()),
        "within_1h": float((np.abs(fwd["pred_hr"] - cold_true_hr) <= 1).mean()),
        "within_2h": float((np.abs(fwd["pred_hr"] - cold_true_hr) <= 2).mean()),
        "amp_mae_z": float(
            np.abs(fwd["pred_amp"] - y_true_z.max(axis=1)).mean()
        ),
    }


def _f2a_adapt(
    base_model: NBEATSxAux,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    lr: float,
    patience: int,
    hr_weight: float,
    seed: int,
) -> tuple[NBEATSxAux, list[dict]]:
    """F2a head-only K-shot fine-tune. Returns (adapted_model, history)."""
    torch.manual_seed(seed)
    model = copy.deepcopy(base_model).to(DEVICE)
    # F2a: freeze backbone, only aux_head trainable.
    for p in model.backbone.parameters():
        p.requires_grad = False
    aux_params = [p for p in model.aux_head.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in aux_params)
    optim = torch.optim.Adam(aux_params, lr=lr)

    x_t = torch.from_numpy(x_train).to(DEVICE)
    y_t = torch.from_numpy(y_train).to(DEVICE)
    x_v = torch.from_numpy(x_val).to(DEVICE)
    y_v = torch.from_numpy(y_val).to(DEVICE)

    history: list[dict] = []
    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        # backbone.eval() to keep BN/Dropout (none here) deterministic;
        # backbone params are frozen anyway via requires_grad=False.
        model.backbone.eval()
        y_hat, hiddens, (amp_p, hr_p) = model(x_t)
        train_loss = peak_aux_loss(amp_p, hr_p, y_t, hr_weight=hr_weight)
        optim.zero_grad()
        train_loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            y_hat_v, _, (amp_v, hr_v) = model(x_v)
            val_loss = peak_aux_loss(amp_v, hr_v, y_v, hr_weight=hr_weight)
        history.append(
            {"epoch": ep, "train_loss": float(train_loss.item()), "val_loss": float(val_loss.item())}
        )
        if val_loss.item() < best_val - 1e-5:
            best_val = float(val_loss)
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.aux_head.eval()
    print(
        f"  [F2a adapt] trainable params={n_trainable}; "
        f"best val_loss={best_val:.4f} after {len(history)} epoch(s)"
    )
    return model, history


def main() -> None:
    ap = argparse.ArgumentParser(description="v03 MVP F2a dry run on one cold apt.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cold_apt", type=str, default="Apt1")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--hr_weight", type=float, default=0.1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    seed_root = V02_OUT_ROOT / f"seed{args.seed}"
    ckpt = seed_root / "T2" / "best.pt"
    cb_path = seed_root / "codebook.npz"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing {ckpt}; run v02 02_train_arms.py first.")
    if not cb_path.exists():
        raise FileNotFoundError(f"missing {cb_path}; run v02 03_fit_codebook.py first.")

    cold_apts = load_v02_split(args.seed)["cold"]
    if args.cold_apt not in cold_apts:
        raise ValueError(
            f"{args.cold_apt} not in seed={args.seed} cold pool {cold_apts}"
        )

    print(f"[setup] seed={args.seed} cold_apt={args.cold_apt} device={DEVICE}")
    print(f"[setup] K-shot={K_SHOT_DAYS}d  buffer={BUFFER_DAYS}d  variant=F2a")

    # ---- v02 frozen backbone + codebook ----
    base_model = NBEATSxAux(latent_source="h_generic").to(DEVICE).eval()
    base_model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))

    cb_npz = np.load(cb_path)
    cb = {k: cb_npz[k] for k in cb_npz.files}

    # ---- windows for the cold apt (z-norm on first 70%, stride=24) ----
    x_z, y_z, mean_, std_ = _build_window_arrays(args.cold_apt, stride=HORIZON)
    n_total = len(x_z)
    # K-shot: first ~K_SHOT_DAYS days of windows. Each stride=24 step = 1 day.
    n_kshot = K_SHOT_DAYS  # 30 windows ≈ 30 days
    n_buffer = BUFFER_DAYS
    n_eval = n_total - n_kshot - n_buffer
    if n_eval <= 0:
        raise RuntimeError(
            f"{args.cold_apt}: only {n_total} windows; not enough for K+buffer+eval."
        )
    # split K-shot itself into train/val for early stop (last 5 windows are val).
    n_kshot_val = min(5, max(1, n_kshot // 6))
    n_kshot_train = n_kshot - n_kshot_val
    x_kshot_tr, y_kshot_tr = x_z[:n_kshot_train], y_z[:n_kshot_train]
    x_kshot_va, y_kshot_va = (
        x_z[n_kshot_train:n_kshot],
        y_z[n_kshot_train:n_kshot],
    )
    eval_start = n_kshot + n_buffer
    x_eval, y_eval = x_z[eval_start:], y_z[eval_start:]
    n_eval_actual = len(x_eval)
    print(
        f"[data] {n_total} windows total; K-shot tr/va={n_kshot_train}/{n_kshot_val}, "
        f"buffer={n_buffer}, eval={n_eval_actual}"
    )

    # ---- routing pieces (R0 KEY-NN; matches v02 default) ----
    eval_keys = extract_key(x_eval).astype(np.float32)
    cold_cluster = route_R0(
        eval_keys,
        cb["key_scaler_mean"],
        cb["key_scaler_scale"],
        cb["key_pool_scaled"],
        cb["cluster_idx"].astype(np.int64),
    )
    mean_arr = np.full(n_eval_actual, mean_, dtype=np.float32)
    std_arr = np.full(n_eval_actual, std_, dtype=np.float32)

    # ---- baseline (no adapt) on eval segment ----
    t0 = time.perf_counter()
    fwd_base = _forward_all(base_model, x_eval)
    base_eval = _evaluate_with_W5(
        fwd_base, y_eval, mean_arr, std_arr, cold_cluster, cb["offsets"]
    )
    base_aux = _aux_acc(fwd_base, y_eval)
    t_base = time.perf_counter() - t0

    # ---- F2a adapt + eval ----
    t1 = time.perf_counter()
    adapted_model, train_history = _f2a_adapt(
        base_model,
        x_train=x_kshot_tr, y_train=y_kshot_tr,
        x_val=x_kshot_va, y_val=y_kshot_va,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
        hr_weight=args.hr_weight, seed=args.seed,
    )
    fwd_adapt = _forward_all(adapted_model, x_eval)
    adapt_eval = _evaluate_with_W5(
        fwd_adapt, y_eval, mean_arr, std_arr, cold_cluster, cb["offsets"]
    )
    adapt_aux = _aux_acc(fwd_adapt, y_eval)
    t_adapt = time.perf_counter() - t1

    # ---- F2a sanity: backbone really frozen → h_g and y_hat unchanged ----
    h_diff = float(np.abs(fwd_adapt["h_g"] - fwd_base["h_g"]).max())
    y_diff = float(np.abs(fwd_adapt["y_hat_z"] - fwd_base["y_hat_z"]).max())
    print(
        f"[sanity] max |dh_g|={h_diff:.2e}  max |dy_hat_z|={y_diff:.2e}  "
        f"(both should be 0 - F2a freezes backbone)"
    )

    # ---- summary print ----
    print(f"\n[time] baseline forward={t_base:.1f}s  F2a adapt+forward={t_adapt:.1f}s")
    for op in ("HR-preserving", "PAPE-aggressive"):
        b = base_eval["operating_points"][op]["metrics"]
        a = adapt_eval["operating_points"][op]["metrics"]
        d_pape = a["pape"] - b["pape"]
        d_hr1 = a["hr@1"] - b["hr@1"]
        print(
            f"  {op:<16}  baseline PAPE={b['pape']:.2f} HR@1={b['hr@1']:.1f}  ->  "
            f"F2a PAPE={a['pape']:.2f} HR@1={a['hr@1']:.1f}  "
            f"(dPAPE={d_pape:+.2f}  dHR@1={d_hr1:+.1f})"
        )
    print(
        f"  aux@eval  top1: base={base_aux['top1']:.3f} -> F2a={adapt_aux['top1']:.3f}; "
        f"amp_mae_z: base={base_aux['amp_mae_z']:.3f} -> F2a={adapt_aux['amp_mae_z']:.3f}"
    )

    # ---- save ----
    out_dir = V03_MVP_ROOT / f"seed{args.seed}" / args.cold_apt
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "seed": int(args.seed),
        "cold_apt": args.cold_apt,
        "variant": "F2a",
        "config": {
            "K_shot_days": K_SHOT_DAYS,
            "buffer_days": BUFFER_DAYS,
            "epochs": args.epochs,
            "lr": args.lr,
            "patience": args.patience,
            "hr_weight": args.hr_weight,
            "n_kshot_train": n_kshot_train,
            "n_kshot_val": n_kshot_val,
            "n_eval_windows": n_eval_actual,
        },
        "routing": "R0",
        "cluster_assignment_unchanged_F2a": True,
        "baseline": {"eval": base_eval, "aux": base_aux},
        "adapted_F2a": {"eval": adapt_eval, "aux": adapt_aux},
        "sanity": {"max_abs_dh_g": h_diff, "max_abs_dy_hat_z": y_diff},
        "train_history": train_history,
        "elapsed_seconds": {"baseline": t_base, "adapt": t_adapt},
    }
    out_path = out_dir / "F2a_result.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n[v03 MVP] saved -> {out_path}")


if __name__ == "__main__":
    main()
