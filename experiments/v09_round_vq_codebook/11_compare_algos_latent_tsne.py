"""V9-RoundCB — compare FL algorithms in latent (h_generic) space via t-SNE.

(한글 요약)
5개 FL 알고리즘 (FedAvg / FedProx / FedRep / Ditto / FedProto) 의 잠재공간 1-NN
routing 구조를 나란히 비교한다.

**왜 알고리즘별 독립 t-SNE 인가**: 알고리즘마다 backbone 이 달라 64-d 잠재공간이
다르다. t-SNE 는 비모수적이라 좌표축에 의미가 없고 배치가 임의이므로, 여러 공간의
점을 한 t-SNE 에 섞으면 (algorithm 으로 색칠) 서로 다른 공간의 거리를 섞는 거짓
그림이 된다. 따라서 알고리즘마다 *독립* t-SNE 를 grid 로 그린다. 패널 내부 구조
(클러스터 분리, codebook tiling) 만 정직하게 비교 가능.

**패널 간 정량 비교**: t-SNE 배치는 패널 간 비교 불가하므로, 실제 64-d 공간에서
1-NN 라우팅 라벨에 대한 **silhouette score** (intra/inter 분리도, [-1,1], 배치
무관) 를 각 패널에 표기하고 마지막 패널에 bar 로 모아 비교한다. perplexity / active
cluster 수도 함께.

공정성: 모든 알고리즘이 *동일한* test 윈도우 (같은 splits/seed) 를 쓰므로 점 집합은
동일하고 임베딩만 다르다.

기본 namespace 는 MAE-only. 학습 없음 — 기존 run 의 final_state_dict.pt 로드.

Output:
  outputs/{namespace}/figures/v09_algos_latent_tsne_seed{S}.png

Per-seed argparse (memory: feedback_argparse_per_seed).
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
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

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
_ALGO_BAR = {
    "FedAvg": "#0072B2", "FedProx": "#D55E00", "FedRep": "#009E73",
    "Ditto": "#CC79A7", "FedProto": "#56B4E9",
}


def _forward_test_h_g(model, test_x, *, batch_size, use_amp):
    """Forward one apt's test windows → h_g_cold (N,64). Mirrors 03's helper."""
    n = int(test_x.shape[0])
    if n == 0:
        return np.zeros((0, 64), np.float32)
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (use_amp and DEVICE.type == "cuda") else _NullCtx()
    )
    model.eval()
    chunks = []
    for i in range(0, n, batch_size):
        xb = torch.from_numpy(test_x[i:i + batch_size]).to(DEVICE, non_blocking=True)
        with torch.no_grad(), amp_ctx:
            ret = model(xb)
        chunks.append(ret[1]["h_generic"].float().cpu().numpy())
    return np.concatenate(chunks, 0).astype(np.float32)


def _routing_for_algo(algo_key, *, seed, namespace, splits, M, K_local,
                      batch_size, use_amp):
    """Return (h_test (N,64), cluster_idx (N,), codebook (M,64), diag) for one algo.

    None if the trained backbone is missing.
    """
    cell = f"V9-RoundCB-{_ALGO_PRETTY[algo_key]}"
    sd_path = OUTPUT_DIR / namespace / f"seed{seed}" / cell / "final_state_dict.pt"
    if not sd_path.exists():
        return None
    model = init_backbone_aux(seed)
    model.load_state_dict(torch.load(sd_path, map_location=DEVICE))
    model.eval()

    # Stage-1 local + Stage-2 merge (same helpers as the experiment).
    packets = [
        local_codebook_step_from_splits(
            model, sp["train_x"], sp["train_y"],
            K_local=int(K_local), seed=int(seed),
            batch_size=int(batch_size), use_amp=use_amp,
        )
        for sp in splits.values()
    ]
    merge = merge_local_codebooks(packets, M_global=int(M), seed=int(seed))
    codebook = merge["codebook"].astype(np.float32)

    # Test forward + 1-NN routing across apts.
    h_list, idx_list = [], []
    for sp in splits.values():
        x = sp["test_x"]
        if x.shape[0] == 0:
            continue
        h = _forward_test_h_g(model, x, batch_size=batch_size, use_amp=use_amp)
        h_list.append(h)
        idx_list.append(_route_h_g_to_codebook(h, codebook))
    h_test = np.concatenate(h_list, 0).astype(np.float32)
    idx = np.concatenate(idx_list, 0).astype(np.int64)
    diag = {"perplexity": float(merge["perplexity"]),
            "utilization": float(merge["utilization"]),
            "n_active": int(len(np.unique(idx)))}
    return h_test, idx, codebook, diag


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compare 5 FL algorithms in h_generic latent space via per-algorithm "
            "t-SNE (1-NN routing colour) + cross-algorithm silhouette bar. Default "
            "namespace = MAE-only. No training. Single seed per invocation."
        )
    )
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--namespace", default="v09_round_vq_codebook_R20_MAEonly",
                    help="Default MAE-only; use ..._R20 for aux=0.3.")
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--K_local", type=int, default=2)
    ap.add_argument("--max_test_points", type=int, default=3000,
                    help="Points per panel for t-SNE + silhouette (kept modest: "
                         "t-SNE & silhouette are O(n^2)).")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    use_amp = not args.no_amp
    rng = np.random.default_rng(args.seed)
    aux_label = "MAE-only (λ_aux=0)" if "MAEonly" in args.namespace else "aux=0.3"
    print(f"[tsne-cmp] namespace={args.namespace}  seed={args.seed}  ({aux_label})")

    splits = build_per_client_splits(seed=args.seed)
    print(f"[tsne-cmp] {len(splits)} apartments.")

    # Same test-window subsample indices reused across algorithms (point identity).
    sub = None
    results: dict[str, dict] = {}
    for key in _ALGO_PRETTY:
        r = _routing_for_algo(
            key, seed=args.seed, namespace=args.namespace, splits=splits,
            M=args.M, K_local=args.K_local, batch_size=args.batch_size, use_amp=use_amp,
        )
        if r is None:
            print(f"[tsne-cmp] WARNING: missing {key}, skipping.")
            continue
        h_test, idx, codebook, diag = r
        if sub is None:
            n = h_test.shape[0]
            sub = (rng.choice(n, size=min(int(args.max_test_points), n), replace=False)
                   if n > int(args.max_test_points) else np.arange(n))
        h_s, idx_s = h_test[sub], idx[sub]
        # silhouette in real 64-d space on the 1-NN labels (comparable across algos).
        try:
            sil = float(silhouette_score(h_s, idx_s)) if len(np.unique(idx_s)) > 1 else float("nan")
        except Exception:
            sil = float("nan")
        results[_ALGO_PRETTY[key]] = {
            "h": h_s, "idx": idx_s, "codebook": codebook, "diag": diag, "sil": sil,
        }
        print(f"[tsne-cmp] {_ALGO_PRETTY[key]:9s} silhouette={sil:+.3f}  "
              f"ppl={diag['perplexity']:.2f}  active={diag['n_active']}/{args.M}")

    if not results:
        raise FileNotFoundError(f"No V9-RoundCB-* runs under {args.namespace}/seed{args.seed}.")

    algos = list(results)
    fig, axes = plt.subplots(2, 3, figsize=(19, 12))
    axes = axes.ravel()
    rt_cmap = plt.get_cmap("tab20", int(args.M))

    for ax, algo in zip(axes, algos):
        res = results[algo]
        # Joint t-SNE on [test points ; codebook] so codebook embeds with its points.
        stacked = np.vstack([res["h"], res["codebook"]]).astype(np.float32)
        perp = min(30, max(5, stacked.shape[0] // 100))
        emb = TSNE(n_components=2, random_state=int(args.seed),
                   perplexity=perp).fit_transform(stacked)
        n_pts = res["h"].shape[0]
        pts, cbk = emb[:n_pts], emb[n_pts:]
        ax.scatter(pts[:, 0], pts[:, 1], s=6, c=res["idx"], cmap=rt_cmap,
                   vmin=0, vmax=int(args.M) - 1, alpha=0.5, linewidths=0)
        ax.scatter(cbk[:, 0], cbk[:, 1], s=70, marker="X", color="black",
                   edgecolor="white", linewidths=0.7, zorder=5)
        d = res["diag"]
        ax.set_title(
            f"{algo}\nsilhouette={res['sil']:+.3f}  ppl={d['perplexity']:.1f}  "
            f"active={d['n_active']}/{args.M}",
            fontsize=11, fontweight="bold")
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.2)

    # Last panel: silhouette comparison bar (cross-algorithm, layout-independent).
    axbar = axes[len(algos)] if len(algos) < len(axes) else None
    for j in range(len(algos) + (1 if axbar is not None else 0), len(axes)):
        axes[j].axis("off")
    if axbar is not None:
        sils = [results[a]["sil"] for a in algos]
        cols = [_ALGO_BAR.get(a, "#888888") for a in algos]
        bars = axbar.bar(range(len(algos)), sils, color=cols, edgecolor="black", linewidth=0.6)
        for b, s in zip(bars, sils):
            axbar.text(b.get_x() + b.get_width() / 2, b.get_height(),
                       f"{s:+.3f}", ha="center",
                       va="bottom" if s >= 0 else "top", fontsize=10, fontweight="bold")
        axbar.axhline(0, color="gray", lw=0.8)
        axbar.set_xticks(range(len(algos)))
        axbar.set_xticklabels(algos, rotation=15, ha="right", fontsize=10)
        axbar.set_ylabel("silhouette (64-d, 1-NN labels)")
        axbar.set_title("패널 간 비교: routing 분리도 (silhouette)\n"
                        "↑ 높을수록 cluster 분리 명확 (배치 무관)",
                        fontsize=11, fontweight="bold")
        axbar.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"v09 RoundCB — FL algorithms in h_generic latent space [t-SNE, {aux_label}]\n"
        f"점 색 = 1-NN 배정 cluster · X = codebook · seed={args.seed} "
        f"(독립 t-SNE: 패널 간 좌표 비교 불가 → silhouette 로 정량 비교)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png = OUTPUT_DIR / args.namespace / "figures" / f"v09_algos_latent_tsne_seed{args.seed}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[tsne-cmp] saved: {out_png}")


if __name__ == "__main__":
    main()
