"""Probe: does NBEATSx h_generic encode peak information?

Setup:
    Model       : MinimalNBEATSx (3-stack, no VQ).
    Checkpoint  : selected via --ckpt. Conclusions about the architecture
                  should come from a clean MAE-trained reference, NOT from
                  v10 b2 (which used peak_weighted_smooth_l1 + bc-reg and
                  thus distorts both peak signal and stack decomposition).
    Households  : 10 from the v10 train split.
    Windows     : sliding stride=24 (non-overlap) on the 70% train segment.
    Targets     : (peak_amp, peak_hour) for INPUT and FORECAST horizons.

Probes (all on z-normalized targets where applicable):
    1. Ridge regression       : feature -> peak_amp        -> R²
    2. MLP probe (1 hidden)   : feature -> peak_amp        -> R² (nonlinear UB)
    3. Logistic regression    : feature -> peak_hour bin   -> top-1 / top-3 acc
    4. Same probes on h_trend, h_seasonal      (control: should be worse)
    5. Baselines:
       - Last 24h of input window    (24-d input features)
       - Daily-mean / std            (2-d input statistics)
       - Random Gaussian features    (sanity floor)

Splits:
    A. Within-household : all windows mixed, 80/20 random  (intra transfer)
    B. Across-household : 7 train households / 3 test     (inter transfer)
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, top_k_accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import OUTPUT_DIR, RANDOM_SEED
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from models.nbeatsx import MinimalNBEATSx

DEFAULT_APTS = [
    "Apt3", "Apt4", "Apt5", "Apt6", "Apt8",
    "Apt9", "Apt10", "Apt11", "Apt14", "Apt15",
]
TRAIN_RATIO = 0.7
STRIDE = 24
BATCH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLEAN_CKPT = OUTPUT_DIR / "v11_clean_pretrain" / "clean_mae" / "best.pt"


def extract_features_one_household(
    model: MinimalNBEATSx, apt: str
) -> dict[str, np.ndarray]:
    series = load_apartment_hourly(apt).values.astype(np.float32)
    n = len(series)
    train_end = int(n * TRAIN_RATIO)
    seg = series[:train_end]
    mean = float(seg.mean())
    std = float(seg.std()) if seg.std() > 1e-8 else 1.0

    ds = HouseholdDataset(seg, mean, std, stride=STRIDE)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False)

    h_t_list, h_s_list, h_g_list = [], [], []
    x_list, y_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            _, hidd = model(x)
            h_t_list.append(hidd["h_trend"].cpu().numpy())
            h_s_list.append(hidd["h_seasonal"].cpu().numpy())
            h_g_list.append(hidd["h_generic"].cpu().numpy())
            x_list.append(x.cpu().numpy())
            y_list.append(y.numpy())

    x_arr = np.concatenate(x_list, axis=0)
    y_arr = np.concatenate(y_list, axis=0)
    return {
        "h_trend": np.concatenate(h_t_list, axis=0),
        "h_seasonal": np.concatenate(h_s_list, axis=0),
        "h_generic": np.concatenate(h_g_list, axis=0),
        "input_z": x_arr,
        "fcst_z": y_arr,
        "peak_amp_in": x_arr.max(axis=1),
        "peak_hr_in": x_arr.argmax(axis=1) % 24,
        "peak_amp_fc": y_arr.max(axis=1),
        "peak_hr_fc": y_arr.argmax(axis=1),
        "apt": np.array([apt] * x_arr.shape[0]),
    }


def linear_probe_r2(X_tr, y_tr, X_te, y_te) -> float:
    sc = StandardScaler().fit(X_tr)
    return float(r2_score(y_te, Ridge(alpha=1.0).fit(sc.transform(X_tr), y_tr).predict(sc.transform(X_te))))


def mlp_probe_r2(X_tr, y_tr, X_te, y_te) -> float:
    sc = StandardScaler().fit(X_tr)
    model = MLPRegressor(
        hidden_layer_sizes=(64,),
        max_iter=200,
        random_state=RANDOM_SEED,
        early_stopping=True,
    ).fit(sc.transform(X_tr), y_tr)
    return float(r2_score(y_te, model.predict(sc.transform(X_te))))


def logistic_topk_acc(X_tr, y_tr, X_te, y_te) -> tuple[float, float]:
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=500, random_state=RANDOM_SEED).fit(sc.transform(X_tr), y_tr)
    pred = clf.predict(sc.transform(X_te))
    proba = clf.predict_proba(sc.transform(X_te))
    top1 = float(accuracy_score(y_te, pred))
    top3 = float(top_k_accuracy_score(y_te, proba, k=3, labels=clf.classes_))
    return top1, top3


def run_probes(feat, target, target_kind, tr_idx, te_idx) -> dict[str, float]:
    sources = {
        "h_generic": feat["h_generic"],
        "h_seasonal": feat["h_seasonal"],
        "h_trend": feat["h_trend"],
        "h_concat": np.concatenate(
            [feat["h_trend"], feat["h_seasonal"], feat["h_generic"]], axis=1
        ),
        "last24": feat["input_z"][:, -24:],
        "stats2": np.stack(
            [feat["input_z"].mean(axis=1), feat["input_z"].std(axis=1)], axis=1
        ),
        "random64": np.random.RandomState(RANDOM_SEED).randn(feat["h_generic"].shape[0], 64),
    }
    out: dict[str, float] = {}
    for name, X in sources.items():
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = target[tr_idx], target[te_idx]
        if target_kind == "regression":
            out[f"{name}/linear_R2"] = linear_probe_r2(Xtr, ytr, Xte, yte)
            out[f"{name}/mlp_R2"] = mlp_probe_r2(Xtr, ytr, Xte, yte)
        else:
            top1, top3 = logistic_topk_acc(Xtr, ytr.astype(int), Xte, yte.astype(int))
            out[f"{name}/top1"] = top1
            out[f"{name}/top3"] = top3
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=CLEAN_CKPT,
                    help="MinimalNBEATSx state_dict (default: clean_mae)")
    ap.add_argument("--tag", default="clean", help="output subdir label")
    ap.add_argument("--apts", nargs="+", default=DEFAULT_APTS)
    args = ap.parse_args()

    out_dir = OUTPUT_DIR / "v11_probe" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print(f"[load] checkpoint: {args.ckpt}")
    if not args.ckpt.exists():
        raise SystemExit(f"checkpoint missing: {args.ckpt}")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "stack_trend.fc1.weight" not in state:
        state = state.get("model_state", state)
    model = MinimalNBEATSx().to(DEVICE).eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    print(f"[extract] {len(args.apts)} households, stride={STRIDE}")
    per_apt = []
    for apt in args.apts:
        try:
            f = extract_features_one_household(model, apt)
            print(f"  {apt}: {f['h_generic'].shape[0]} windows")
            per_apt.append(f)
        except FileNotFoundError as e:
            print(f"  {apt}: SKIP ({e})")
    keys = ["h_trend", "h_seasonal", "h_generic", "input_z", "fcst_z",
            "peak_amp_in", "peak_hr_in", "peak_amp_fc", "peak_hr_fc"]
    feat = {k: np.concatenate([d[k] for d in per_apt], axis=0) for k in keys}
    apt_arr = np.concatenate([d["apt"] for d in per_apt], axis=0)
    N = feat["h_generic"].shape[0]
    print(f"[extract] total windows: {N}")

    rng = np.random.RandomState(RANDOM_SEED)
    perm = rng.permutation(N)
    tr_a = perm[: int(N * 0.8)]
    te_a = perm[int(N * 0.8) :]

    test_apts = set(args.apts[-3:])
    is_test = np.array([a in test_apts for a in apt_arr])
    tr_b = np.where(~is_test)[0]
    te_b = np.where(is_test)[0]

    targets = {
        "peak_amp_in": ("regression", feat["peak_amp_in"]),
        "peak_amp_fc": ("regression", feat["peak_amp_fc"]),
        "peak_hr_in": ("classification", feat["peak_hr_in"]),
        "peak_hr_fc": ("classification", feat["peak_hr_fc"]),
    }

    results: dict[str, dict[str, dict[str, float]]] = {}
    for split_name, (tr, te) in [("A_within", (tr_a, te_a)), ("B_across", (tr_b, te_b))]:
        print(f"\n[probe] split={split_name}  n_tr={len(tr)}  n_te={len(te)}")
        results[split_name] = {}
        for tgt_name, (kind, y) in targets.items():
            res = run_probes(feat, y, kind, tr, te)
            results[split_name][tgt_name] = res
            top = sorted(
                ((k, v) for k, v in res.items() if k.endswith(("_R2", "/top1"))),
                key=lambda kv: kv[1],
                reverse=True,
            )
            print(f"  {tgt_name} ({kind}):")
            for k, v in top:
                print(f"    {k:30s} {v:+.4f}")

    out_path = out_dir / "probe_results.json"
    with open(out_path, "w") as fh:
        json.dump(
            {
                "n_total_windows": int(N),
                "n_apartments": len(per_apt),
                "checkpoint": str(args.ckpt),
                "tag": args.tag,
                "apts": list(args.apts),
                "results": results,
            },
            fh,
            indent=2,
        )
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
