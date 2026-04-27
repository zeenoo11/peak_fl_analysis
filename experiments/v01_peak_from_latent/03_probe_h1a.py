"""H1a gate: probe whether each arm's latent encodes peak.

7 arms tested:
    T0 = h_generic (MAE only)
    T1 = h_concat  (MAE only — alias of T0 ckpt with different readout)
    T2 = h_generic + peak_aux
    T3 = h_concat  + peak_aux
    T4 = W·h_concat (Ridge projection, T0 backbone)
    T5 = forecast (T0 backbone, ŷ)
    T6 = h_generic ‖ stats2 (hybrid, T0 backbone)

Probes:
    Ridge / MLP regression on peak_amp_fc -> R² + PAPE
    Logistic on peak_hr_fc -> top-1 / top-3

Across-household split: 40 train / 10 cold-probe (last 10 of train list).

PASS criterion (H1a): Ridge R²(peak_amp_fc) >= 0.70.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, top_k_accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import OUTPUT_DIR, RANDOM_SEED, TRAIN_RATIO
from dataloader.splits import load_v10_split
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from models.nbeatsx import MinimalNBEATSx
from models.nbeatsx_aux import NBEATSxAux
from utils.metrics import compute_pape

OUT = OUTPUT_DIR / "v01_peak_from_latent"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARMS = ["T0", "T1", "T2", "T3", "T4", "T5", "T6"]
H1A_PASS_R2 = 0.70


def load_arm_extractor(arm: str):
    """Returns extract_fn(x: torch.Tensor [B,96]) -> latent np.ndarray [B, D]."""
    if arm in ("T0", "T1", "T4", "T5", "T6"):
        m = MinimalNBEATSx().to(DEVICE).eval()
        m.load_state_dict(
            torch.load(OUT / "T0" / "best.pt", map_location="cpu", weights_only=False)
        )
    elif arm == "T2":
        m = NBEATSxAux(latent_source="h_generic").to(DEVICE).eval()
        m.load_state_dict(
            torch.load(OUT / arm / "best.pt", map_location="cpu", weights_only=False)
        )
    elif arm == "T3":
        m = NBEATSxAux(latent_source="h_concat").to(DEVICE).eval()
        m.load_state_dict(
            torch.load(OUT / arm / "best.pt", map_location="cpu", weights_only=False)
        )
    else:
        raise ValueError(arm)

    if arm == "T0":
        def fn(x):
            with torch.no_grad():
                _, h = m(x.to(DEVICE))
            return h["h_generic"].cpu().numpy()
    elif arm == "T1":
        def fn(x):
            with torch.no_grad():
                _, h = m(x.to(DEVICE))
            return torch.cat([h["h_trend"], h["h_seasonal"], h["h_generic"]], dim=1).cpu().numpy()
    elif arm == "T2":
        def fn(x):
            with torch.no_grad():
                _, h, _ = m(x.to(DEVICE))
            return h["h_generic"].cpu().numpy()
    elif arm == "T3":
        def fn(x):
            with torch.no_grad():
                _, h, _ = m(x.to(DEVICE))
            return torch.cat([h["h_trend"], h["h_seasonal"], h["h_generic"]], dim=1).cpu().numpy()
    elif arm == "T4":
        Wd = np.load(OUT / "T4" / "W.npz")
        W, mu, sc = Wd["W"], Wd["scaler_mean"], Wd["scaler_scale"]
        def fn(x):
            with torch.no_grad():
                _, h = m(x.to(DEVICE))
            hc = torch.cat([h["h_trend"], h["h_seasonal"], h["h_generic"]], dim=1).cpu().numpy()
            return ((hc - mu) / sc) @ W.T
    elif arm == "T5":
        def fn(x):
            with torch.no_grad():
                y_hat, _ = m(x.to(DEVICE))
            return y_hat.cpu().numpy()
    elif arm == "T6":
        def fn(x):
            with torch.no_grad():
                _, h = m(x.to(DEVICE))
            x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
            stats2 = np.stack([x_np.mean(axis=1), x_np.std(axis=1)], axis=1)
            return np.concatenate([h["h_generic"].cpu().numpy(), stats2], axis=1)
    return fn


def gather_features(extract_fn, apts: list[str]):
    feats, amp, hr, apt_idx = [], [], [], []
    for ai, apt in enumerate(apts):
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            continue
        n = len(series)
        train_end = int(n * TRAIN_RATIO)
        seg = series[:train_end]
        mean = float(seg.mean()); std = float(seg.std()) if seg.std() > 1e-8 else 1.0
        ds = HouseholdDataset(seg, mean, std, stride=24)
        for x, y in DataLoader(ds, batch_size=256, shuffle=False):
            feats.append(extract_fn(x))
            amp.append(y.numpy().max(axis=1))
            hr.append(y.numpy().argmax(axis=1))
            apt_idx.append(np.full(len(y), ai, dtype=np.int32))
    return (
        np.concatenate(feats, axis=0),
        np.concatenate(amp, axis=0),
        np.concatenate(hr, axis=0),
        np.concatenate(apt_idx, axis=0),
    )


def peak_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Per-window MAPE on scalar peak amplitudes (1-D arrays).

    Replaces the degenerate `compute_pape(y[None,:], p[None,:])` which collapses
    to a single (max(p)-max(y))/|max(y)| comparison instead of per-sample error.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    valid = np.abs(y_true) > 1e-5
    if not valid.any():
        return 0.0
    return float(np.mean(np.abs(y_pred[valid] - y_true[valid]) / np.abs(y_true[valid])) * 100.0)


def run_probe(X_tr, y_tr, X_te, y_te, kind: str):
    sc = StandardScaler().fit(X_tr)
    Xs_tr, Xs_te = sc.transform(X_tr), sc.transform(X_te)
    if kind == "regression":
        ridge = Ridge(alpha=1.0).fit(Xs_tr, y_tr)
        pred_r = ridge.predict(Xs_te)
        mlp = MLPRegressor(
            hidden_layer_sizes=(64,), max_iter=200,
            random_state=RANDOM_SEED, early_stopping=True
        ).fit(Xs_tr, y_tr)
        pred_m = mlp.predict(Xs_te)
        return {
            "ridge_R2": float(r2_score(y_te, pred_r)),
            "ridge_PAPE": peak_mape(y_te, pred_r),
            "mlp_R2": float(r2_score(y_te, pred_m)),
            "mlp_PAPE": peak_mape(y_te, pred_m),
        }
    clf = LogisticRegression(max_iter=500, random_state=RANDOM_SEED).fit(Xs_tr, y_tr)
    pred = clf.predict(Xs_te); proba = clf.predict_proba(Xs_te)
    return {
        "top1": float(accuracy_score(y_te, pred)),
        "top3": float(top_k_accuracy_score(y_te, proba, k=3, labels=clf.classes_)),
    }


def main():
    np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
    apts = load_v10_split()["train"]
    train_apts = apts[:40]; cold_probe_apts = apts[40:]
    print(f"[probe] {len(train_apts)} train_probe + {len(cold_probe_apts)} cold_probe")

    results = {}
    for arm in ARMS:
        print(f"\n========== {arm} ==========")
        extract_fn = load_arm_extractor(arm)
        X_tr, amp_tr, hr_tr, _ = gather_features(extract_fn, train_apts)
        X_te, amp_te, hr_te, _ = gather_features(extract_fn, cold_probe_apts)
        print(f"  X_tr={X_tr.shape}  X_te={X_te.shape}")

        amp_res = run_probe(X_tr, amp_tr, X_te, amp_te, "regression")
        hr_res = run_probe(X_tr, hr_tr, X_te, hr_te, "classification")
        results[arm] = {
            "peak_amp_fc": amp_res, "peak_hr_fc": hr_res,
            "n_tr": int(X_tr.shape[0]), "n_te": int(X_te.shape[0]),
            "feat_dim": int(X_tr.shape[1]),
        }
        gate = "PASS" if amp_res["ridge_R2"] >= H1A_PASS_R2 else "FAIL"
        print(f"  amp_fc: Ridge R²={amp_res['ridge_R2']:.3f}  PAPE={amp_res['ridge_PAPE']:.1f}%  "
              f"MLP R²={amp_res['mlp_R2']:.3f}  [{gate} H1a]")
        print(f"  hr_fc:  top1={hr_res['top1']:.3f}  top3={hr_res['top3']:.3f}")

    pass_arms = [a for a, r in results.items()
                 if r["peak_amp_fc"]["ridge_R2"] >= H1A_PASS_R2]
    print(f"\n[H1a] PASS arms: {pass_arms}")

    out_path = OUT / "probe_h1a.json"
    with open(out_path, "w") as fh:
        json.dump({"pass_threshold": H1A_PASS_R2, "results": results,
                   "pass_arms": pass_arms}, fh, indent=2)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
