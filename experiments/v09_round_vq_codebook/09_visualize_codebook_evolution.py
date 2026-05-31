"""V9-RoundCB — visualise how the federated codebook evolves across rounds.

(한글 요약)
``03_fl_per_round_codebook.py`` 는 매 라운드 종료 시 federated KMeans 로 server
codebook 을 다시 fit 하고 그 스냅샷을 ``codebook_history.pt`` (rounds, codebook
(R,M,D), offsets (R,M,H)) 에 쌓는다. 본 스크립트는 그 스냅샷을 읽어 라운드별
진화를 시각화한다. 학습/codebook fit 을 다시 하지 않는 순수 후처리.

**중요 — index 궤적 금지**: ``merge_local_codebooks`` 는 매 라운드 KMeans 를 새로
돌리므로 codebook[r][i] 와 codebook[r+1][i] 가 같은 클러스터라는 보장이 없다
(label permutation). 따라서 *엔트리 index 를 이어 그리는 궤적은 틀린 그림*이다.
대신 permutation-invariant 한 양만 쓴다:
  - codebook 엔트리 전체를 공유 PCA 에 round-gradient 색으로 (분포 이동/확장).
  - codebook **평균(centroid) 궤적** — 엔트리 평균은 순서 무관.
  - 연속 라운드 간 **Chamfer set-distance** — 집합 대 집합, 순서 무관.
  - spread / perplexity / utilization / ΔPAPE lift (codebook_log.jsonl).

PCA 는 모든 라운드 codebook 스냅샷을 stack 해 한 번 fit 한다. 단, backbone 도
라운드마다 바뀌므로 (h_generic 의 basis 자체가 이동) 절대 좌표는 codebook 과
backbone 의 *결합* drift 를 보여주는 것임에 유의 (제목에 명시).

Output:
  outputs/{namespace}/figures/v09_codebook_evolution_{cell}_seed{S}.png

Per-seed argparse (memory: feedback_argparse_per_seed). 학습 없음.
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
from sklearn.decomposition import PCA

from config import OUTPUT_DIR, RANDOM_SEED

try:
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
except Exception:
    pass
matplotlib.rcParams["axes.unicode_minus"] = False


_ALGO_PRETTY = {
    "fedavg": "FedAvg", "fedprox": "FedProx", "fedrep": "FedRep",
    "ditto": "Ditto", "fedproto": "FedProto",
}


def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer distance between two point sets a(M,D), b(N,D).

    Permutation-invariant: 0.5·(mean_i min_j ||a_i-b_j|| + mean_j min_i ||a_i-b_j||).
    """
    d2 = (a ** 2).sum(1, keepdims=True) - 2.0 * a @ b.T + (b ** 2).sum(1)
    d = np.sqrt(np.clip(d2, 0.0, None))
    return 0.5 * (float(d.min(axis=1).mean()) + float(d.min(axis=0).mean()))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Visualise the round-by-round evolution of the v09 RoundCB federated "
            "codebook from codebook_history.pt + codebook_log.jsonl (no training, "
            "no re-fit). Single seed × single algorithm per invocation."
        )
    )
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--algorithm", default="fedavg", choices=list(_ALGO_PRETTY))
    ap.add_argument("--namespace", default="v09_round_vq_codebook_R20",
                    help="Output namespace holding the trained run.")
    args = ap.parse_args()

    cell = f"V9-RoundCB-{_ALGO_PRETTY[args.algorithm]}"
    run_dir = OUTPUT_DIR / args.namespace / f"seed{args.seed}" / cell
    hist_path = run_dir / "codebook_history.pt"
    log_path = run_dir / "codebook_log.jsonl"
    if not hist_path.exists():
        raise FileNotFoundError(
            f"No codebook_history.pt at {hist_path}. Run 03_fl_per_round_codebook.py "
            f"for --algorithm {args.algorithm} --seed {args.seed} first."
        )

    fig_dir = OUTPUT_DIR / args.namespace / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_png = fig_dir / f"v09_codebook_evolution_{cell}_seed{args.seed}.png"

    print(f"[evo] cell={cell}  seed={args.seed}  namespace={args.namespace}")

    # 1) Codebook snapshots.
    hist = torch.load(hist_path, map_location="cpu")
    rounds = list(hist["rounds"])
    cb = hist["codebook"].float().numpy()          # (R, M, D)
    offsets = hist["offsets"].float().numpy()      # (R, M, H)
    R, M, D = cb.shape
    print(f"[evo] codebook (R,M,D)=({R},{M},{D})  rounds={rounds[0]}..{rounds[-1]}")

    # 2) Per-round diagnostics from the log (if present).
    util = ppl = lift = None
    if log_path.exists():
        rows = [json.loads(l) for l in log_path.open()]
        lr = {r["round"]: r for r in rows}
        util = [lr[rd]["codebook_diag"]["utilization"] for rd in rounds if rd in lr]
        ppl = [lr[rd]["codebook_diag"]["perplexity"] for rd in rounds if rd in lr]
        lift = [lr[rd]["lift"]["pape_delta"] for rd in rounds if rd in lr]

    # 3) Shared PCA over all snapshots.
    proj = PCA(n_components=2, random_state=int(args.seed))
    cb_flat = cb.reshape(R * M, D)
    proj.fit(cb_flat)
    cb2 = proj.transform(cb_flat).reshape(R, M, 2)
    ev = proj.explained_variance_ratio_
    axis_lbl = (f"PC1 ({ev[0] * 100:.1f}%)", f"PC2 ({ev[1] * 100:.1f}%)")

    # Permutation-invariant per-round quantities.
    centroid = cb.mean(axis=1)                                              # (R, D)
    centroid2 = proj.transform(centroid)                                   # (R, 2)
    spread = np.linalg.norm(cb - centroid[:, None, :], axis=2).mean(axis=1)  # (R,)
    chamfer = [np.nan] + [_chamfer(cb[i - 1], cb[i]) for i in range(1, R)]   # (R,)
    cstep = [np.nan] + [float(np.linalg.norm(centroid[i] - centroid[i - 1]))
                        for i in range(1, R)]                               # (R,)
    off_norm = np.linalg.norm(offsets, axis=2).mean(axis=1)                 # (R,)

    # 4) Render 2×2.
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=rounds[0], vmax=rounds[-1])
    xticks = rounds[::max(1, R // 10)]

    # --- Panel A: all codebook entries colored by round + centroid trajectory ---
    axA = axes[0, 0]
    for i, rd in enumerate(rounds):
        axA.scatter(cb2[i, :, 0], cb2[i, :, 1], s=22, color=cmap(norm(rd)),
                    alpha=0.55, linewidths=0)
    axA.plot(centroid2[:, 0], centroid2[:, 1], "-", color="black", lw=1.4, alpha=0.8, zorder=4)
    axA.scatter(centroid2[:, 0], centroid2[:, 1], s=70, c=rounds, cmap=cmap,
                edgecolor="black", linewidths=0.8, zorder=5, marker="o")
    for i in (0, R - 1):
        axA.annotate(f"R{rounds[i]} mean", (centroid2[i, 0], centroid2[i, 1]),
                     fontsize=9, fontweight="bold", zorder=6,
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="gray", alpha=0.85))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=axA, label="round", fraction=0.046, pad=0.04)
    axA.set_title("Panel A — codebook 엔트리 진화 (색=round) + 평균 궤적\n"
                  "(backbone 도 라운드마다 변함: codebook+backbone 결합 drift)",
                  fontsize=11, fontweight="bold")
    axA.set_xlabel(axis_lbl[0]); axA.set_ylabel(axis_lbl[1]); axA.grid(alpha=0.25)

    # --- Panel B: convergence — Chamfer set-drift + centroid step ---
    axB = axes[0, 1]
    axB.plot(rounds, chamfer, "-o", color="#d62728", lw=2.0, markersize=5,
             label="Chamfer set-drift  (R-1 → R)")
    axB.plot(rounds, cstep, "-s", color="#1f77b4", lw=1.8, markersize=4,
             label="centroid step  ||mean_R − mean_{R-1}||")
    axB.set_title("Panel B — 라운드 간 codebook 이동량 (수렴 정량화)\n"
                  "set-to-set (permutation-invariant)",
                  fontsize=11, fontweight="bold")
    axB.set_xlabel("Round"); axB.set_ylabel("latent-space distance")
    axB.set_xticks(xticks); axB.grid(alpha=0.3)
    axB.legend(fontsize=9, loc="upper right")

    # --- Panel C: spread + perplexity ---
    axC = axes[1, 0]
    axC.plot(rounds, spread, "-o", color="#2ca02c", lw=2.0, markersize=5,
             label="codebook spread (mean dist→centroid)")
    axC.set_xlabel("Round"); axC.set_ylabel("spread", color="#2ca02c")
    axC.tick_params(axis="y", labelcolor="#2ca02c")
    axC.set_xticks(xticks); axC.grid(alpha=0.3)
    if ppl is not None:
        axC2 = axC.twinx()
        axC2.plot(rounds, ppl, "-D", color="#9467bd", lw=1.8, markersize=4, label="perplexity")
        axC2.set_ylabel("perplexity", color="#9467bd")
        axC2.tick_params(axis="y", labelcolor="#9467bd")
        l1, lab1 = axC.get_legend_handles_labels()
        l2, lab2 = axC2.get_legend_handles_labels()
        axC.legend(l1 + l2, lab1 + lab2, fontsize=9, loc="lower right")
    else:
        axC.legend(fontsize=9, loc="lower right")
    axC.set_title("Panel C — codebook 다양성: spread & perplexity",
                  fontsize=11, fontweight="bold")

    # --- Panel D: ΔPAPE lift + offset strength ---
    axD = axes[1, 1]
    if lift is not None:
        axD.axhline(0, ls="--", color="gray", alpha=0.6, lw=1.0)
        axD.plot(rounds, lift, "-o", color="#ff7f0e", lw=2.0, markersize=5,
                 label="ΔPAPE  (after − before)")
        axD.set_ylabel("ΔPAPE  (음수=codebook 도움)", color="#ff7f0e")
        axD.tick_params(axis="y", labelcolor="#ff7f0e")
    axDr = axD.twinx()
    axDr.plot(rounds, off_norm, "-^", color="#8c564b", lw=1.6, markersize=4,
              label="mean CMO offset L2")
    axDr.set_ylabel("offset L2 norm", color="#8c564b")
    axDr.tick_params(axis="y", labelcolor="#8c564b")
    axD.set_xlabel("Round"); axD.set_xticks(xticks); axD.grid(alpha=0.3)
    l1, lab1 = axD.get_legend_handles_labels()
    l2, lab2 = axDr.get_legend_handles_labels()
    axD.legend(l1 + l2, lab1 + lab2, fontsize=9, loc="upper right")
    axD.set_title("Panel D — codebook 효과(ΔPAPE) & CMO offset 세기",
                  fontsize=11, fontweight="bold")

    fig.suptitle(
        f"{cell}  seed={args.seed}  ({args.namespace})  —  "
        f"federated codebook round-by-round evolution  (R={R}, M={M})",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[evo] PC1+PC2 explained var = {ev[0]*100:.1f}% + {ev[1]*100:.1f}%")
    print(f"[evo] saved: {out_png}")


if __name__ == "__main__":
    main()
