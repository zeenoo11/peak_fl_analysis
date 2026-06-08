"""V9 CMO correction-strength (alpha) sensitivity — post-hoc, no retraining.

(한글 요약)
RoundCB 의 추론 시점 보정은 ``ŷ_corr = ŷ_base + α · offset[c*]`` 으로, α 는 학습과
무관한 post-hoc 스칼라다. 따라서 이미 학습된 backbone(``final_state_dict.pt``)과
최종 codebook/offset(``codebook_history.pt`` 의 마지막 라운드)을 로드해
apt 별 base 예측·routing·offset 을 **1회만** 구한 뒤, α grid 에 대해
보정 metric 을 재계산하면 된다. backbone 은 절대 codebook 을 통과하지 않는다.

조건: **aux=0** (= ``aux_lambda=0.0``, MAE-only backbone). 기본 namespace 는
``v09_round_vq_codebook_R20_MAEonly`` 이며, 여기에 aux=0 RoundCB 셀들이 이미 있다.

α=0 은 보정 없음(= base), α=1.0 은 기존 보고값과 일치해야 한다(sanity check).

Per-seed argparse — multi-seed sweep 은 호출 측의 몫.

Output (per seed):
    outputs/{namespace}/seed{S}/V9-RoundCB-{Algo}/alpha_sweep.json

Figure (mean ± std over whatever seeds are present):
    papers/conference_draft/figures/fig_alpha_sweep.png
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

from config import HORIZON, OUTPUT_DIR, RANDOM_SEED
from dataloader.per_client_split import build_per_client_splits
from fl.base import DEVICE
from fl.codebook_fl import _NullCtx, _route_h_g_to_codebook
from fl.fedavg_aux import init_backbone_aux
from utils.metrics import compute_hr, compute_mae, compute_mse, compute_pape

_ALGO_PRETTY = {
    "fedavg":   "FedAvg",
    "fedprox":  "FedProx",
    "fedrep":   "FedRep",
    "ditto":    "Ditto",
    "fedproto": "FedProto",
}


# --- helpers copied verbatim from 03_fl_per_round_codebook.py -----------------


def _forward_test_h_g(model, test_x, *, batch_size, use_amp):
    n = int(test_x.shape[0])
    if n == 0:
        return (np.zeros((0, 64), dtype=np.float32),
                np.zeros((0, HORIZON), dtype=np.float32))
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (use_amp and DEVICE.type == "cuda")
        else _NullCtx()
    )
    model.eval()
    h_chunks, yhat_chunks = [], []
    for i in range(0, n, batch_size):
        xb = torch.from_numpy(test_x[i : i + batch_size]).to(DEVICE, non_blocking=True)
        with torch.no_grad(), amp_ctx:
            ret = model(xb)
        h_chunks.append(ret[1]["h_generic"].float().cpu().numpy())
        yhat_chunks.append(ret[0].float().cpu().numpy())
    return (np.concatenate(h_chunks, axis=0).astype(np.float32),
            np.concatenate(yhat_chunks, axis=0).astype(np.float32))


def _wmape_kw(y_true_kw, y_pred_kw):
    """Weighted abs % error (WMAPE/WAPE) = 100 * sum|e| / sum|y| over the apt.

    Robust to near-zero off-peak loads (no per-point division), unlike raw MAPE.
    """
    denom = float(np.abs(y_true_kw).sum())
    if denom <= 1e-9:
        return float("nan")
    return 100.0 * float(np.abs(y_pred_kw - y_true_kw).sum()) / denom


def _per_apt_metrics_kw(y_true_kw, y_pred_kw):
    return {
        "pape":    float(compute_pape(y_true_kw, y_pred_kw)),
        "hr@1":    float(compute_hr(y_true_kw, y_pred_kw, tol=1)),
        "hr@2":    float(compute_hr(y_true_kw, y_pred_kw, tol=2)),
        "mae":     float(compute_mae(y_true_kw, y_pred_kw)),
        "mse_kw2": float(compute_mse(y_true_kw, y_pred_kw)),
        "wmape":   _wmape_kw(y_true_kw, y_pred_kw),
    }


def _mean_across(per_apt, key):
    return float(np.mean([m[key] for m in per_apt])) if per_apt else float("nan")


# --- core: cache base preds + offsets once, then sweep alpha ------------------


def run_alpha_sweep(seed, algorithm, namespace, alphas, batch_size, use_amp):
    cell_name = f"V9-RoundCB-{_ALGO_PRETTY[algorithm]}"
    cell_dir = OUTPUT_DIR / namespace / f"seed{seed}" / cell_name
    sd_path = cell_dir / "final_state_dict.pt"
    hist_path = cell_dir / "codebook_history.pt"
    if not sd_path.exists() or not hist_path.exists():
        raise FileNotFoundError(
            f"missing artifacts in {cell_dir} — train the aux=0 backbone first."
        )

    # 1) load aux=0 backbone.
    model = init_backbone_aux(seed)
    model.load_state_dict(torch.load(sd_path, map_location=DEVICE))
    model.eval()

    # 2) final-round codebook + offsets (last entry in history).
    hist = torch.load(hist_path, map_location="cpu")
    codebook = hist["codebook"][-1].numpy().astype(np.float32)  # (M, D)
    offsets = hist["offsets"][-1].numpy().astype(np.float32)    # (M, H)
    final_round = int(hist["rounds"][-1])

    # 3) per-apt: forward test ONCE, route, cache base preds + per-window offset.
    splits = build_per_client_splits(seed=seed)
    cache = []  # list of (y_true_kw, y_hat_base_z, offset_win, std, mean)
    for apt, sp in splits.items():
        x, y = sp["test_x"], sp["test_y"]
        if x.shape[0] == 0:
            continue
        m_, s_ = float(sp["mean"]), float(sp["std"])
        h_g, y_hat_base_z = _forward_test_h_g(
            model, x, batch_size=batch_size, use_amp=use_amp
        )
        c_idx = _route_h_g_to_codebook(h_g, codebook)
        cache.append({
            "y_true_kw": (y * s_ + m_).astype(np.float32),
            "y_hat_base_z": y_hat_base_z,
            "offset_win": offsets[c_idx].astype(np.float32),  # (n, H)
            "std": s_, "mean": m_,
        })

    # 4) sweep alpha — base preds reused, only the offset scaling changes.
    rows = []
    for a in alphas:
        per_apt = []
        for c in cache:
            y_corr_z = (c["y_hat_base_z"] + float(a) * c["offset_win"]).astype(np.float32)
            y_corr_kw = (y_corr_z * c["std"] + c["mean"]).astype(np.float32)
            per_apt.append(_per_apt_metrics_kw(c["y_true_kw"], y_corr_kw))
        rows.append({
            "alpha":  float(a),
            "pape":   _mean_across(per_apt, "pape"),
            "hr@1":   _mean_across(per_apt, "hr@1"),
            "hr@2":   _mean_across(per_apt, "hr@2"),
            "mae":    _mean_across(per_apt, "mae"),
            "mse_kw2": _mean_across(per_apt, "mse_kw2"),
            "wmape":  _mean_across(per_apt, "wmape"),
        })

    out = {
        "cell": cell_name,
        "seed": int(seed),
        "namespace": namespace,
        "aux_lambda": 0.0,
        "final_round": final_round,
        "M": int(codebook.shape[0]),
        "n_clients": len(cache),
        "alphas": [float(a) for a in alphas],
        "sweep": rows,
        "comment": (
            "Post-hoc CMO correction-strength (alpha) sweep on a frozen aux=0 "
            "(MAE-only) backbone + final-round federated codebook. No retraining: "
            "base preds and per-window offsets cached once, only alpha scales the "
            "offset. alpha=0 == base (no correction); alpha=1.0 reproduces the "
            "reported RoundCB lift."
        ),
    }
    out_path = cell_dir / "alpha_sweep.json"
    out_path.write_text(json.dumps(out, indent=2))
    base = rows[0]
    print(f"[{cell_name}] seed{seed}  R{final_round}  base PAPE={base['pape']:.2f}")
    for r in rows:
        print(f"    alpha={r['alpha']:.2f}  PAPE={r['pape']:.2f}  "
              f"lift={r['pape'] - base['pape']:+.2f}  MAE={r['mae']:.4f}")
    print(f"[{cell_name}] wrote {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Post-hoc CMO alpha sensitivity on a frozen aux=0 backbone."
    )
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--algorithm", default="fedavg", choices=list(_ALGO_PRETTY.keys()))
    ap.add_argument("--namespace", default="v09_round_vq_codebook_R20_MAEonly")
    ap.add_argument("--alpha_min", type=float, default=0.0)
    ap.add_argument("--alpha_max", type=float, default=2.0)
    ap.add_argument("--alpha_step", type=float, default=0.25)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    n = int(round((args.alpha_max - args.alpha_min) / args.alpha_step)) + 1
    alphas = [round(args.alpha_min + i * args.alpha_step, 4) for i in range(n)]

    run_alpha_sweep(
        seed=args.seed, algorithm=args.algorithm, namespace=args.namespace,
        alphas=alphas, batch_size=args.batch_size, use_amp=not args.no_amp,
    )


if __name__ == "__main__":
    main()
