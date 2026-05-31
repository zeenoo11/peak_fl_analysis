"""V9-RoundCB — visualise the latent (h_generic) space at codebook aggregation.

(한글 요약)
``03_fl_per_round_codebook.py`` 의 post-hoc federated codebook 파이프라인이 실제로
잠재공간(h_generic, 64-d)에서 무엇을 하는지 눈으로 확인하기 위한 시각화/점검 스크립트.
학습은 다시 하지 않고 기존 run 의 ``final_state_dict.pt`` 를 로드해 backbone 을 복원한
뒤, ``src/fl/codebook_fl.py`` 의 *동일한* 헬퍼로 다음 3단계를 재현·시각화한다:

  Panel A  Local Stage-1   : 가구별 train h_g 구름 + 가구별 local KMeans centroid.
  Panel B  Server Stage-2  : 전체 local centroid(파랑) 위에 merge 된 server codebook(빨강).
  Panel C  1-NN routing    : test h_g 를 server codebook 으로 1-NN 라우팅한 결과(클러스터별 색).

64-d → 2-d 투영은 기본 PCA (local/server/test 가 같은 좌표계를 공유하도록 한 번 fit 후
transform). ``--method tsne`` 는 plot 대상 점들을 한꺼번에 stack 해서 fit (t-SNE 는
out-of-sample transform 이 없으므로).

federation contract 는 분석 목적상 무시한다 (raw h_g 를 한 프로세스에서 모아 그림).

Output:
  outputs/{namespace}/figures/v09_latent_viz_{cell}_seed{S}_{method}.png

Per-seed argparse (memory: feedback_argparse_per_seed). 학습 없음 — 순수 후처리 시각화.
"""

from __future__ import annotations

import argparse
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

from config import HORIZON, OUTPUT_DIR, RANDOM_SEED
from dataloader.per_client_split import build_per_client_splits
from fl.base import DEVICE, _NullCtx
from fl.codebook_fl import (
    _route_h_g_to_codebook,
    local_codebook_step_from_splits,
    merge_local_codebooks,
)
from fl.fedavg_aux import init_backbone_aux

try:
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
except Exception:
    pass
matplotlib.rcParams["axes.unicode_minus"] = False


_ALGO_PRETTY = {
    "fedavg": "FedAvg", "fedprox": "FedProx", "fedrep": "FedRep",
    "ditto": "Ditto", "fedproto": "FedProto",
}


def _forward_test_h_g(model, test_x, *, batch_size, use_amp):
    """Forward one apt's test windows → (h_g_cold (N,64), y_hat_z (N,H)).

    Mirrors ``03_fl_per_round_codebook.py:_forward_test_h_g`` so the routed
    latent matches the experiment exactly.
    """
    n = int(test_x.shape[0])
    if n == 0:
        return np.zeros((0, 64), np.float32), np.zeros((0, HORIZON), np.float32)
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (use_amp and DEVICE.type == "cuda") else _NullCtx()
    )
    model.eval()
    h_chunks, yhat_chunks = [], []
    for i in range(0, n, batch_size):
        xb = torch.from_numpy(test_x[i:i + batch_size]).to(DEVICE, non_blocking=True)
        with torch.no_grad(), amp_ctx:
            ret = model(xb)
        h_chunks.append(ret[1]["h_generic"].float().cpu().numpy())
        yhat_chunks.append(ret[0].float().cpu().numpy())
    return (
        np.concatenate(h_chunks, 0).astype(np.float32),
        np.concatenate(yhat_chunks, 0).astype(np.float32),
    )


def _subsample(arr, n_max, rng):
    if arr.shape[0] <= n_max:
        return arr, np.arange(arr.shape[0])
    idx = rng.choice(arr.shape[0], size=n_max, replace=False)
    return arr[idx], idx


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Visualise h_generic latent space of the v09 RoundCB post-hoc "
            "federated codebook: local Stage-1 centroids, merged server codebook, "
            "and 1-NN routing of test latents. Loads an existing trained backbone "
            "(no training). Single seed × single algorithm per invocation."
        )
    )
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--algorithm", default="fedavg", choices=list(_ALGO_PRETTY))
    ap.add_argument("--namespace", default="v09_round_vq_codebook",
                    help="Output namespace holding the trained run "
                         "(e.g. v09_round_vq_codebook_R20).")
    ap.add_argument("--M", type=int, default=32, help="Global codebook size (match the run).")
    ap.add_argument("--K_local", type=int, default=2, help="Per-client Stage-1 cluster count.")
    ap.add_argument("--method", default="pca", choices=["pca", "tsne"],
                    help="2-d projection. pca = shared fit+transform (default); "
                         "tsne = joint fit on the plotted points.")
    ap.add_argument("--n_highlight", type=int, default=6,
                    help="How many clients to colour individually in Panel A.")
    ap.add_argument("--max_per_client", type=int, default=300,
                    help="Max train h_g points plotted per highlighted client (Panel A).")
    ap.add_argument("--max_test_points", type=int, default=6000,
                    help="Max test h_g points plotted in Panel C (1-NN routing).")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    use_amp = not args.no_amp
    rng = np.random.default_rng(args.seed)
    cell = f"V9-RoundCB-{_ALGO_PRETTY[args.algorithm]}"
    run_dir = OUTPUT_DIR / args.namespace / f"seed{args.seed}" / cell
    sd_path = run_dir / "final_state_dict.pt"
    if not sd_path.exists():
        raise FileNotFoundError(
            f"No trained backbone at {sd_path}. Run 03_fl_per_round_codebook.py "
            f"for --algorithm {args.algorithm} --seed {args.seed} first, or point "
            f"--namespace at a directory that has it."
        )

    fig_dir = OUTPUT_DIR / args.namespace / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_png = fig_dir / f"v09_latent_viz_{cell}_seed{args.seed}_{args.method}.png"

    print(f"[viz] cell={cell}  seed={args.seed}  namespace={args.namespace}")
    print(f"[viz] backbone <- {sd_path}")

    # 1) Restore the trained backbone.
    model = init_backbone_aux(args.seed)
    state = torch.load(sd_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # 2) Per-client splits (cached on disk).
    splits = build_per_client_splits(seed=args.seed)
    apt_order = list(splits.keys())
    print(f"[viz] {len(apt_order)} apartments.")

    # 3) Stage-1 local KMeans per apt (same helper as the experiment).
    packets = []
    for apt in apt_order:
        sp = splits[apt]
        packets.append(local_codebook_step_from_splits(
            model, sp["train_x"], sp["train_y"],
            K_local=int(args.K_local), seed=int(args.seed),
            batch_size=int(args.batch_size), use_amp=use_amp,
        ))

    # 4) Stage-2 federated merge → server codebook (M, D).
    merge = merge_local_codebooks(packets, M_global=int(args.M), seed=int(args.seed))
    codebook = merge["codebook"].astype(np.float32)  # (M, D)
    print(f"[viz] codebook (M,D)={codebook.shape}  util={merge['utilization']:.3f}  "
          f"ppl={merge['perplexity']:.2f}")

    # All local centroids stacked (Stage-2 input cloud).
    local_centroids = np.vstack(
        [p["centroids"] for p in packets if int(p["K_local_i"]) > 0]
    ).astype(np.float32)

    # 5) Test forward + 1-NN routing, collected across apts.
    test_h_list, test_idx_list = [], []
    for apt in apt_order:
        x = splits[apt]["test_x"]
        if x.shape[0] == 0:
            continue
        h_test, _ = _forward_test_h_g(model, x, batch_size=int(args.batch_size), use_amp=use_amp)
        c_idx = _route_h_g_to_codebook(h_test, codebook)
        test_h_list.append(h_test)
        test_idx_list.append(c_idx)
    test_h = np.concatenate(test_h_list, 0).astype(np.float32)
    test_idx = np.concatenate(test_idx_list, 0).astype(np.int64)
    test_h_s, sub = _subsample(test_h, int(args.max_test_points), rng)
    test_idx_s = test_idx[sub]
    print(f"[viz] test latents routed: N={test_h.shape[0]} "
          f"(plotting {test_h_s.shape[0]}), clusters used={len(np.unique(test_idx))}/{args.M}")

    # 6) Highlighted clients for Panel A (spread across the apt list).
    hi_pos = np.unique(np.linspace(0, len(apt_order) - 1, num=int(args.n_highlight)).astype(int))
    hi_apts = [apt_order[i] for i in hi_pos]
    hi_h, hi_cent, hi_names = [], [], []
    for apt in hi_apts:
        p = packets[apt_order.index(apt)]
        if int(p["K_local_i"]) == 0:
            continue
        hg, _ = _subsample(p["h_g"], int(args.max_per_client), rng)
        hi_h.append(hg)
        hi_cent.append(p["centroids"])
        hi_names.append(apt)

    # 7) Shared 2-d projection.
    if args.method == "pca":
        # Fit once on (all local centroids + codebook + pooled train sample); transform all.
        pooled_train = np.vstack([
            _subsample(p["h_g"], 200, rng)[0]
            for p in packets if int(p["K_local_i"]) > 0
        ]).astype(np.float32)
        proj = PCA(n_components=2, random_state=int(args.seed))
        proj.fit(np.vstack([local_centroids, codebook, pooled_train]))
        tf = proj.transform
        ev = proj.explained_variance_ratio_
        axis_lbl = (f"PC1 ({ev[0] * 100:.1f}%)", f"PC2 ({ev[1] * 100:.1f}%)")
        hi_h2 = [tf(h) for h in hi_h]
        hi_cent2 = [tf(c) for c in hi_cent]
        local_centroids2 = tf(local_centroids)
        codebook2 = tf(codebook)
        test_h2 = tf(test_h_s)
    else:
        from sklearn.manifold import TSNE
        # Joint fit: stack everything we plot, remember the slice boundaries.
        blocks, spans, cur = [], [], 0
        for h in hi_h:
            blocks.append(h); spans.append((cur, cur + len(h))); cur += len(h)
        s_lc = (cur, cur + len(local_centroids)); blocks.append(local_centroids); cur += len(local_centroids)
        s_cb = (cur, cur + len(codebook)); blocks.append(codebook); cur += len(codebook)
        s_te = (cur, cur + len(test_h_s)); blocks.append(test_h_s); cur += len(test_h_s)
        stacked = np.vstack(blocks).astype(np.float32)
        emb = TSNE(n_components=2, random_state=int(args.seed),
                   perplexity=min(30, max(5, stacked.shape[0] // 100))).fit_transform(stacked)
        hi_h2 = [emb[a:b] for (a, b) in spans]
        # per-client centroids fall out of a single TSNE block; recover by apt order
        # by re-deriving them as the nearest emb of their own block mean is messy —
        # instead embed centroids via their own slice:
        lc_all = emb[s_lc[0]:s_lc[1]]
        # split lc_all back per highlighted client using known centroid counts
        hi_cent2, off = [], 0
        # local_centroids stacks ALL clients; highlighted centroids are a subset.
        # For tsne we simply re-embed highlighted centroids by index lookup:
        # (they are contained in local_centroids in apt order)
        # Build index map.
        full_names = [apt for apt in apt_order if int(packets[apt_order.index(apt)]["K_local_i"]) > 0]
        counts = [packets[apt_order.index(n)]["centroids"].shape[0] for n in full_names]
        starts = np.cumsum([0] + counts)
        for n in hi_names:
            j = full_names.index(n)
            hi_cent2.append(lc_all[starts[j]:starts[j + 1]])
        local_centroids2 = lc_all
        codebook2 = emb[s_cb[0]:s_cb[1]]
        test_h2 = emb[s_te[0]:s_te[1]]
        axis_lbl = ("t-SNE 1", "t-SNE 2")

    # 8) Render 3 panels.
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.4))
    cli_cmap = plt.get_cmap("tab10")

    # --- Panel A: local clouds + per-client centroids ---
    axA = axes[0]
    for k, (h2, c2, name) in enumerate(zip(hi_h2, hi_cent2, hi_names)):
        col = cli_cmap(k % 10)
        axA.scatter(h2[:, 0], h2[:, 1], s=6, color=col, alpha=0.25, linewidths=0)
        axA.scatter(c2[:, 0], c2[:, 1], s=240, marker="*", color=col,
                    edgecolor="black", linewidths=0.9, zorder=5,
                    label=f"{name} (K={c2.shape[0]})")
    axA.set_title("Panel A — Local Stage-1\n가구별 train h_g 구름 + local KMeans centroid (★)",
                  fontsize=11, fontweight="bold")
    axA.set_xlabel(axis_lbl[0]); axA.set_ylabel(axis_lbl[1])
    axA.legend(fontsize=7, loc="best", framealpha=0.9)
    axA.grid(alpha=0.25)

    # --- Panel B: server aggregation over all local centroids ---
    axB = axes[1]
    axB.scatter(local_centroids2[:, 0], local_centroids2[:, 1], s=18, color="#1f77b4",
                alpha=0.45, linewidths=0, label=f"local centroids (n={local_centroids.shape[0]})")
    axB.scatter(codebook2[:, 0], codebook2[:, 1], s=130, marker="X", color="#d62728",
                edgecolor="black", linewidths=0.8, zorder=5,
                label=f"server codebook (M={codebook.shape[0]})")
    axB.set_title("Panel B — Server Stage-2 aggregation\n전체 local centroid 위에 merge 된 server codebook (X)",
                  fontsize=11, fontweight="bold")
    axB.set_xlabel(axis_lbl[0]); axB.set_ylabel(axis_lbl[1])
    axB.legend(fontsize=8, loc="best", framealpha=0.9)
    axB.grid(alpha=0.25)

    # --- Panel C: 1-NN routing of test latents ---
    axC = axes[2]
    rt_cmap = plt.get_cmap("tab20", int(args.M))
    axC.scatter(test_h2[:, 0], test_h2[:, 1], s=6, c=test_idx_s, cmap=rt_cmap,
                vmin=0, vmax=int(args.M) - 1, alpha=0.5, linewidths=0)
    axC.scatter(codebook2[:, 0], codebook2[:, 1], s=120, marker="X", color="black",
                edgecolor="white", linewidths=0.8, zorder=5, label="codebook entry")
    # Annotate the most-used codebook entries.
    used, cnts = np.unique(test_idx, return_counts=True)
    for c in used[np.argsort(cnts)[::-1][:8]]:
        axC.annotate(str(int(c)), (codebook2[c, 0], codebook2[c, 1]),
                     fontsize=8, fontweight="bold", color="black",
                     ha="center", va="center", zorder=6,
                     bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
    axC.set_title("Panel C — 1-NN routing (test)\ntest h_g 를 codebook 으로 1-NN 라우팅 (색 = 배정 cluster)",
                  fontsize=11, fontweight="bold")
    axC.set_xlabel(axis_lbl[0]); axC.set_ylabel(axis_lbl[1])
    axC.legend(fontsize=8, loc="best", framealpha=0.9)
    axC.grid(alpha=0.25)

    fig.suptitle(
        f"{cell}  seed={args.seed}  ({args.namespace})  —  h_generic latent space "
        f"[{args.method.upper()}]   util={merge['utilization']:.2f}  ppl={merge['perplexity']:.1f}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved: {out_png}")


if __name__ == "__main__":
    main()
