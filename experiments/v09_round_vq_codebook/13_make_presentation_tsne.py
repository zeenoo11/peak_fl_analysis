"""Presentation t-SNE of the federated residual codebook in latent space.

Mirrors 08_visualize_latents.py (same src helpers, same pipeline) but renders
the 3 panels with audience-facing English titles and writes into
papers/conference_draft/figures/ — no internal jargon (no namespace string,
no "h_generic", no Korean panel labels).

Fixed to the FedAvg / seed 42 run of the MAE-only namespace (t-SNE is
run-specific; averaging across algorithms is not meaningful for a projection).

Writes: papers/conference_draft/figures/fig7_latent_codebook_tsne.png
"""

from __future__ import annotations

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
from sklearn.manifold import TSNE

from config import HORIZON, OUTPUT_DIR, RANDOM_SEED
from dataloader.per_client_split import build_per_client_splits
from fl.base import DEVICE, _NullCtx
from fl.codebook_fl import (
    _route_h_g_to_codebook,
    local_codebook_step_from_splits,
    merge_local_codebooks,
)
from fl.fedavg_aux import init_backbone_aux

matplotlib.rcParams["axes.unicode_minus"] = False

SEED = RANDOM_SEED            # 42
NAMESPACE = "v09_round_vq_codebook_R20_MAEonly"
CELL = "V9-RoundCB-FedAvg"
M, K_LOCAL = 32, 2
N_HIGHLIGHT, MAX_PER_CLIENT, MAX_TEST = 6, 300, 5000
BATCH, USE_AMP = 512, True

OUT = Path(__file__).resolve().parents[2] / "papers/conference_draft/figures"
OUT.mkdir(parents=True, exist_ok=True)


def _forward_test_h_g(model, test_x):
    n = int(test_x.shape[0])
    if n == 0:
        return np.zeros((0, 64), np.float32), np.zeros((0, HORIZON), np.float32)
    amp_ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
               if (USE_AMP and DEVICE.type == "cuda") else _NullCtx())
    model.eval()
    h_chunks, yhat_chunks = [], []
    for i in range(0, n, BATCH):
        xb = torch.from_numpy(test_x[i:i + BATCH]).to(DEVICE, non_blocking=True)
        with torch.no_grad(), amp_ctx:
            ret = model(xb)
        h_chunks.append(ret[1]["h_generic"].float().cpu().numpy())
        yhat_chunks.append(ret[0].float().cpu().numpy())
    return np.concatenate(h_chunks, 0).astype(np.float32), np.concatenate(yhat_chunks, 0)


def _subsample(arr, n_max, rng):
    if arr.shape[0] <= n_max:
        return arr, np.arange(arr.shape[0])
    idx = rng.choice(arr.shape[0], size=n_max, replace=False)
    return arr[idx], idx


def main() -> None:
    rng = np.random.default_rng(SEED)
    run_dir = OUTPUT_DIR / NAMESPACE / f"seed{SEED}" / CELL
    sd_path = run_dir / "final_state_dict.pt"
    if not sd_path.exists():
        raise FileNotFoundError(f"No trained backbone at {sd_path}.")

    model = init_backbone_aux(SEED)
    model.load_state_dict(torch.load(sd_path, map_location=DEVICE))
    model.eval()

    splits = build_per_client_splits(seed=SEED)
    apt_order = list(splits.keys())
    print(f"[tsne] {len(apt_order)} apartments; backbone <- {sd_path}")

    # Stage-1 local KMeans per apt; Stage-2 federated merge.
    packets = [local_codebook_step_from_splits(
        model, splits[apt]["train_x"], splits[apt]["train_y"],
        K_local=K_LOCAL, seed=SEED, batch_size=BATCH, use_amp=USE_AMP)
        for apt in apt_order]
    merge = merge_local_codebooks(packets, M_global=M, seed=SEED)
    codebook = merge["codebook"].astype(np.float32)
    local_centroids = np.vstack(
        [p["centroids"] for p in packets if int(p["K_local_i"]) > 0]).astype(np.float32)
    print(f"[tsne] codebook M={codebook.shape[0]}  util={merge['utilization']:.2f}  "
          f"ppl={merge['perplexity']:.1f}")

    # Test forward + 1-NN routing across apts.
    test_h_list, test_idx_list = [], []
    for apt in apt_order:
        x = splits[apt]["test_x"]
        if x.shape[0] == 0:
            continue
        h_test, _ = _forward_test_h_g(model, x)
        test_h_list.append(h_test)
        test_idx_list.append(_route_h_g_to_codebook(h_test, codebook))
    test_h = np.concatenate(test_h_list, 0).astype(np.float32)
    test_idx = np.concatenate(test_idx_list, 0).astype(np.int64)
    test_h_s, sub = _subsample(test_h, MAX_TEST, rng)
    test_idx_s = test_idx[sub]

    # Highlighted clients for Panel A.
    hi_pos = np.unique(np.linspace(0, len(apt_order) - 1, num=N_HIGHLIGHT).astype(int))
    hi_apts = [apt_order[i] for i in hi_pos]
    hi_h, hi_cent, hi_names = [], [], []
    for apt in hi_apts:
        p = packets[apt_order.index(apt)]
        if int(p["K_local_i"]) == 0:
            continue
        hg, _ = _subsample(p["h_g"], MAX_PER_CLIENT, rng)
        hi_h.append(hg)
        hi_cent.append(p["centroids"])
        hi_names.append(apt)

    # Joint t-SNE fit over everything we plot.
    blocks, spans, cur = [], [], 0
    for h in hi_h:
        blocks.append(h); spans.append((cur, cur + len(h))); cur += len(h)
    s_lc = (cur, cur + len(local_centroids)); blocks.append(local_centroids); cur += len(local_centroids)
    s_cb = (cur, cur + len(codebook)); blocks.append(codebook); cur += len(codebook)
    s_te = (cur, cur + len(test_h_s)); blocks.append(test_h_s); cur += len(test_h_s)
    stacked = np.vstack(blocks).astype(np.float32)
    print(f"[tsne] fitting t-SNE on {stacked.shape[0]} points ...")
    emb = TSNE(n_components=2, random_state=SEED,
               perplexity=min(30, max(5, stacked.shape[0] // 100))).fit_transform(stacked)

    hi_h2 = [emb[a:b] for (a, b) in spans]
    lc_all = emb[s_lc[0]:s_lc[1]]
    full_names = [apt for apt in apt_order if int(packets[apt_order.index(apt)]["K_local_i"]) > 0]
    counts = [packets[apt_order.index(n)]["centroids"].shape[0] for n in full_names]
    starts = np.cumsum([0] + counts)
    hi_cent2 = [lc_all[starts[full_names.index(n)]:starts[full_names.index(n) + 1]] for n in hi_names]
    local_centroids2 = lc_all
    codebook2 = emb[s_cb[0]:s_cb[1]]
    test_h2 = emb[s_te[0]:s_te[1]]
    axis_lbl = ("t-SNE 1", "t-SNE 2")

    # Render 3 panels with English titles.
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.4))
    cli_cmap = plt.get_cmap("tab10")

    axA = axes[0]
    for k, (h2, c2, name) in enumerate(zip(hi_h2, hi_cent2, hi_names)):
        col = cli_cmap(k % 10)
        axA.scatter(h2[:, 0], h2[:, 1], s=6, color=col, alpha=0.25, linewidths=0)
        axA.scatter(c2[:, 0], c2[:, 1], s=240, marker="*", color=col,
                    edgecolor="black", linewidths=0.9, zorder=5,
                    label=f"Household {name}")
    axA.set_title("Local prototypes (Stage 1)\n"
                  "Per-household clouds + local centroids (★)",
                  fontsize=12, fontweight="bold")
    axA.set_xlabel(axis_lbl[0]); axA.set_ylabel(axis_lbl[1])
    axA.legend(fontsize=8, loc="best", framealpha=0.9)
    axA.grid(alpha=0.25)

    axB = axes[1]
    axB.scatter(local_centroids2[:, 0], local_centroids2[:, 1], s=18, color="#1f77b4",
                alpha=0.45, linewidths=0, label=f"Local prototypes (n={local_centroids.shape[0]})")
    axB.scatter(codebook2[:, 0], codebook2[:, 1], s=130, marker="X", color="#d62728",
                edgecolor="black", linewidths=0.8, zorder=5,
                label=f"Global codebook (M={codebook.shape[0]})")
    axB.set_title("Server codebook (Stage 2)\n"
                  "Merged over all local prototypes (✕)",
                  fontsize=12, fontweight="bold")
    axB.set_xlabel(axis_lbl[0]); axB.set_ylabel(axis_lbl[1])
    axB.legend(fontsize=9, loc="best", framealpha=0.9)
    axB.grid(alpha=0.25)

    axC = axes[2]
    rt_cmap = plt.get_cmap("tab20", M)
    axC.scatter(test_h2[:, 0], test_h2[:, 1], s=6, c=test_idx_s, cmap=rt_cmap,
                vmin=0, vmax=M - 1, alpha=0.5, linewidths=0)
    axC.scatter(codebook2[:, 0], codebook2[:, 1], s=120, marker="X", color="black",
                edgecolor="white", linewidths=0.8, zorder=5, label="Codebook entry")
    used, cnts = np.unique(test_idx, return_counts=True)
    for c in used[np.argsort(cnts)[::-1][:8]]:
        axC.annotate(str(int(c)), (codebook2[c, 0], codebook2[c, 1]),
                     fontsize=8, fontweight="bold", color="black",
                     ha="center", va="center", zorder=6,
                     bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
    axC.set_title("Test-time routing\n"
                  "Each test latent → nearest entry (color = cluster)",
                  fontsize=12, fontweight="bold")
    axC.set_xlabel(axis_lbl[0]); axC.set_ylabel(axis_lbl[1])
    axC.legend(fontsize=9, loc="best", framealpha=0.9)
    axC.grid(alpha=0.25)

    fig.suptitle("Residual Codebook in Latent Space (t-SNE)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = OUT / "fig7_latent_codebook_tsne.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[tsne] saved: {out_png}")


if __name__ == "__main__":
    main()
