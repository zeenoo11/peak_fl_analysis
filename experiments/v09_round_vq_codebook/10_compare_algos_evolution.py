"""V9-RoundCB — compare federated codebook evolution across FL algorithms.

(한글 요약)
``09_visualize_codebook_evolution.py`` 가 한 알고리즘의 라운드별 진화를 4-패널로
보여준다면, 본 스크립트는 5개 FL 알고리즘 (FedAvg / FedProx / FedRep / Ditto /
FedProto) 을 한 그림에 *겹쳐* 나란히 비교한다. 진화 지표는 모두 라운드별 스칼라
(permutation-invariant) 라서 알고리즘마다 한 줄씩 overlay 하면 된다 — PCA 좌표계
차이 문제 없음.

  Panel A  Chamfer set-drift (R-1→R)  : 라운드 간 codebook 집합 이동량 = 수렴 속도.
  Panel B  codebook spread             : 엔트리들이 중심에서 얼마나 퍼져 있나 (다양성).
  Panel C  perplexity                  : test routing 의 cluster 사용 균등도.
  Panel D  ΔPAPE lift (after − before) : codebook 보정 효과 (음수 = 도움).

기본 namespace 는 MAE-only (``v09_round_vq_codebook_R20_MAEonly``). ``--namespace``
로 aux=0.3 (``..._R20``) 등으로 교체 가능. 학습/재-fit 없는 순수 후처리.

Output:
  outputs/{namespace}/figures/v09_algos_evolution_compare_seed{S}.png

Per-seed argparse (memory: feedback_argparse_per_seed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import OUTPUT_DIR, RANDOM_SEED

try:
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
except Exception:
    pass
matplotlib.rcParams["axes.unicode_minus"] = False


# Wong colour-blind-safe palette (matches 07_make_figures.py _G_COLORS).
_ALGO_COLORS = {
    "FedAvg":   "#0072B2",
    "FedProx":  "#D55E00",
    "FedRep":   "#009E73",
    "Ditto":    "#CC79A7",
    "FedProto": "#56B4E9",
}


def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer distance between point sets a(M,D), b(N,D)."""
    d2 = (a ** 2).sum(1, keepdims=True) - 2.0 * a @ b.T + (b ** 2).sum(1)
    d = np.sqrt(np.clip(d2, 0.0, None))
    return 0.5 * (float(d.min(axis=1).mean()) + float(d.min(axis=0).mean()))


def _load_algo(run_dir: Path) -> dict | None:
    """Read one algorithm's per-round evolution metrics, or None if missing."""
    hist_path = run_dir / "codebook_history.pt"
    log_path = run_dir / "codebook_log.jsonl"
    if not hist_path.exists():
        return None
    hist = torch.load(hist_path, map_location="cpu")
    rounds = list(hist["rounds"])
    cb = hist["codebook"].float().numpy()      # (R, M, D)
    R = cb.shape[0]
    centroid = cb.mean(axis=1)                 # (R, D)
    spread = np.linalg.norm(cb - centroid[:, None, :], axis=2).mean(axis=1)
    chamfer = [np.nan] + [_chamfer(cb[i - 1], cb[i]) for i in range(1, R)]
    ppl = lift = None
    if log_path.exists():
        rows = {r["round"]: r for r in (json.loads(l) for l in log_path.open())}
        ppl = [rows[rd]["codebook_diag"]["perplexity"] for rd in rounds if rd in rows]
        lift = [rows[rd]["lift"]["pape_delta"] for rd in rounds if rd in rows]
    return {"rounds": rounds, "spread": spread, "chamfer": chamfer,
            "ppl": ppl, "lift": lift}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compare v09 RoundCB federated codebook evolution across 5 FL "
            "algorithms (overlaid lines). Default namespace = MAE-only. "
            "No training, no re-fit. Single seed per invocation."
        )
    )
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--namespace", default="v09_round_vq_codebook_R20_MAEonly",
                    help="Output namespace (default MAE-only; use ..._R20 for aux=0.3).")
    args = ap.parse_args()

    root = OUTPUT_DIR / args.namespace / f"seed{args.seed}"
    fig_dir = OUTPUT_DIR / args.namespace / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_png = fig_dir / f"v09_algos_evolution_compare_seed{args.seed}.png"

    aux_label = "MAE-only (λ_aux=0)" if "MAEonly" in args.namespace else "aux=0.3"
    print(f"[cmp] namespace={args.namespace}  seed={args.seed}  ({aux_label})")

    data: dict[str, dict] = {}
    for algo in _ALGO_COLORS:
        d = _load_algo(root / f"V9-RoundCB-{algo}")
        if d is None:
            print(f"[cmp] WARNING: missing {algo}, skipping.")
            continue
        data[algo] = d
    if not data:
        raise FileNotFoundError(f"No V9-RoundCB-* runs under {root}.")
    print(f"[cmp] algorithms: {list(data)}")

    any_rounds = next(iter(data.values()))["rounds"]
    R = len(any_rounds)
    xticks = any_rounds[::max(1, R // 10)]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    def _line(ax, key):
        for algo, d in data.items():
            y = d[key]
            if y is None:
                continue
            ax.plot(d["rounds"], y, "-o", color=_ALGO_COLORS[algo], lw=2.0,
                    markersize=4, label=algo)
        ax.set_xlabel("Round"); ax.set_xticks(xticks); ax.grid(alpha=0.3)

    # Panel A — Chamfer set-drift (convergence speed).
    axA = axes[0, 0]
    _line(axA, "chamfer")
    axA.set_ylabel("Chamfer set-drift  (R-1 → R)")
    axA.set_title("Panel A — 라운드 간 codebook 이동량 (수렴 속도)\n"
                  "set-to-set, permutation-invariant",
                  fontsize=11, fontweight="bold")
    axA.legend(fontsize=9, loc="upper right")

    # Panel B — codebook spread (diversity).
    axB = axes[0, 1]
    _line(axB, "spread")
    axB.set_ylabel("codebook spread (mean dist→centroid)")
    axB.set_title("Panel B — codebook spread (엔트리 다양성)",
                  fontsize=11, fontweight="bold")
    axB.legend(fontsize=9, loc="best")

    # Panel C — perplexity.
    axC = axes[1, 0]
    _line(axC, "ppl")
    axC.set_ylabel("perplexity")
    axC.set_title("Panel C — test routing perplexity (cluster 사용 균등도)",
                  fontsize=11, fontweight="bold")
    axC.legend(fontsize=9, loc="best")

    # Panel D — ΔPAPE lift.
    axD = axes[1, 1]
    axD.axhline(0, ls="--", color="gray", alpha=0.6, lw=1.0)
    _line(axD, "lift")
    axD.set_ylabel("ΔPAPE  (after − before; 음수=codebook 도움)")
    axD.set_title("Panel D — codebook 보정 효과 ΔPAPE",
                  fontsize=11, fontweight="bold")
    axD.legend(fontsize=9, loc="best")

    fig.suptitle(
        f"v09 RoundCB — codebook evolution across FL algorithms  "
        f"[{aux_label}]  seed={args.seed}, R={R}, E=5",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[cmp] saved: {out_png}")


if __name__ == "__main__":
    main()
