"""V9-FedVQ — federated NBEATSxAux with a federated VQ codebook on h_generic.

(한글 요약)
plan ``plans/v09-01_round_wise_codebook.md`` §3 의 *minimal-viable* 본 구현.
smoke driver (``01_fl_vq_naive_smoke.py``) 에서 naive FedAvg 가 VQ 버퍼를 통째로
weighted-average 하여 round 2 만에 perplexity=1.00 으로 codebook collapse 가
발생함을 확인했다. 본 driver 는 다음 세 가지 mechanism 으로 그 collapse 를
해소한다:

  1. **Round-start EMA reset on each client.** ``apply_state_dict`` 직후
     ``stack_generic.vq.ema_count`` / ``ema_weight`` 를 ``.zero_()`` 한다.
     ``codebook`` 은 routing 용으로 그대로 유지. 이로써 client local EMA 는
     그 round 의 batch 만 누적한다.
  2. **Mass-weighted codebook aggregation + EMA blending.** 각 codebook entry
     c 에 대해
         total_mass    = Σ_i ema_count_local_i[c]
         new_centroid  = (Σ_i ema_weight_local_i[c]) / total_mass   (mass > 0 일 때)
         new_codebook  = γ · prev_codebook[c] + (1 − γ) · new_centroid
     dead (total_mass == 0) 인 entry 는 이전 codebook 을 그대로 유지.
     서버측 ``ema_count`` / ``ema_weight`` 는 client 합으로 갱신.
  3. **Dead-code respawn.** ``respawn_period`` 라운드마다 aggregation 직후
     ``ema_count[c] < respawn_n_min`` 인 entry 를 최다 사용 entry 방향 + 작은
     Gaussian noise 로 교체. ``ema_count[c]=1.0``, ``ema_weight[c]=codebook[c]``.

Backbone (non-VQ) 파라미터는 표준 FedAvg weighted-by-``n_train_windows`` (즉
``fl.base.weighted_average``) 로 집계한다.

Cell name:
  - ``V9-FedVQ-aux``    when ``--use_aux_head`` (default)
  - ``V9-FedVQ-noaux``  when ``--no-use_aux_head``

Output (``outputs/v09_round_vq_codebook/seed{S}/{cell}/``):
  - ``round_log.jsonl``      — per-round train/val + server VQ diagnostics
  - ``codebook_history.pt``  — ``{rounds, codebook, ema_count, ema_weight}``
  - ``final_state_dict.pt``
  - ``result.json``          — terminal summary including ``test_terminal``

Per-seed argparse — multi-seed sweep is the executor's job.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.cluster import KMeans

from config import D_MODEL, OUTPUT_DIR, RANDOM_SEED
from dataloader.per_client_split import build_per_client_splits
from fl.base import DEVICE, apply_state_dict, clone_state_dict, weighted_average
from models.nbeatsx_aux_vq import NBEATSxAuxVQ
from models.peak_aux_head import peak_aux_loss
from utils.metrics import compute_hr, compute_mae, compute_mse, compute_pape


# ---------------------------------------------------------------------------
# AMP helper (mirrors 01_fl_vq_naive_smoke.py).
# ---------------------------------------------------------------------------


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def _wrap_amp(use_amp: bool):
    if use_amp and DEVICE.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return _NullCtx()


# ---------------------------------------------------------------------------
# Server-side init.
# ---------------------------------------------------------------------------


def _init_model(
    seed: int, num_embeddings: int, commitment_beta: float, use_aux_head: bool
) -> NBEATSxAuxVQ:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = NBEATSxAuxVQ(
        num_embeddings=num_embeddings,
        commitment_beta=commitment_beta,
        use_aux_head=use_aux_head,
    ).to(DEVICE)
    return model


# ---------------------------------------------------------------------------
# Local training (one client × n_epochs).
# ---------------------------------------------------------------------------


def _local_train_one_client(
    model: NBEATSxAuxVQ,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    n_epochs: int,
    use_amp: bool,
    use_aux_head: bool,
    aux_lambda: float,
    hr_weight: float,
) -> dict:
    """One client × n_epochs of (MAE + λ·peak_aux + commit) SGD.

    The ``VectorQuantizerEMA`` inside ``stack_generic.vq`` updates its EMA
    buffers in-place on every forward() while ``model.training==True``.
    """
    model.train()
    n_batches = 0
    sum_main, sum_aux, sum_commit = 0.0, 0.0, 0.0
    sum_util, sum_ppl = 0.0, 0.0
    for _ in range(n_epochs):
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _wrap_amp(use_amp):
                out = model(x)
                y_hat = out["y_hat"]
                commit = out["vq_state"]["commit_loss"]
                main = F.l1_loss(y_hat, y)
                loss = main + commit
                if use_aux_head and out["aux"] is not None:
                    amp_pred, hr_pred = out["aux"]
                    aux = peak_aux_loss(amp_pred, hr_pred, y, hr_weight=hr_weight)
                    loss = loss + aux_lambda * aux
                    sum_aux += float(aux.item())
            loss.backward()
            optimizer.step()
            sum_main += float(main.item())
            sum_commit += float(commit.item())
            sum_util += float(out["vq_state"]["utilization"])
            sum_ppl += float(out["vq_state"]["perplexity"])
            n_batches += 1
    return {
        "n_batches": n_batches,
        "main_loss_mean":   sum_main   / max(n_batches, 1),
        "aux_loss_mean":    sum_aux    / max(n_batches, 1) if use_aux_head else 0.0,
        "commit_loss_mean": sum_commit / max(n_batches, 1),
        "vq_util_mean":     sum_util   / max(n_batches, 1),
        "vq_ppl_mean":      sum_ppl    / max(n_batches, 1),
    }


# ---------------------------------------------------------------------------
# Per-client evaluation (mirrors 01_fl_vq_naive_smoke.py / RoundLogger).
# ---------------------------------------------------------------------------


@torch.no_grad()
def _eval_per_client(
    model: NBEATSxAuxVQ,
    splits: dict[str, dict],
    split_key: str,
    *,
    batch_size: int,
    use_amp: bool,
) -> dict[str, float]:
    """Across-client mean of per-apt PAPE/HR/MAE/MSE on `split_key` ∈ {val,test}."""
    model.eval()
    papes, maes, mses, hr1s, hr2s, hr3s = [], [], [], [], [], []
    util_sum, ppl_sum, n_chunks = 0.0, 0.0, 0
    for _apt, sp in splits.items():
        x = sp[f"{split_key}_x"]
        y = sp[f"{split_key}_y"]
        if x.shape[0] == 0:
            continue
        m_, s_ = float(sp["mean"]), float(sp["std"])
        yhat_chunks = []
        for i in range(0, int(x.shape[0]), batch_size):
            xb = torch.from_numpy(x[i : i + batch_size]).to(DEVICE, non_blocking=True)
            with _wrap_amp(use_amp):
                out = model(xb)
            yhat_chunks.append(out["y_hat"].float().cpu().numpy())
            util_sum += float(out["vq_state"]["utilization"])
            ppl_sum  += float(out["vq_state"]["perplexity"])
            n_chunks += 1
        y_hat_z   = np.concatenate(yhat_chunks, axis=0).astype(np.float32)
        y_true_kw = (y * s_ + m_).astype(np.float32)
        y_hat_kw  = (y_hat_z * s_ + m_).astype(np.float32)
        papes.append(float(compute_pape(y_true_kw, y_hat_kw)))
        maes.append (float(compute_mae (y_true_kw, y_hat_kw)))
        mses.append (float(compute_mse (y_true_kw, y_hat_kw)))
        hr1s.append (float(compute_hr  (y_true_kw, y_hat_kw, tol=1)))
        hr2s.append (float(compute_hr  (y_true_kw, y_hat_kw, tol=2)))
        hr3s.append (float(compute_hr  (y_true_kw, y_hat_kw, tol=3)))
    return {
        "pape_mean":               float(np.mean(papes)) if papes else float("nan"),
        "pape_std_across_clients": float(np.std(papes, ddof=1)) if len(papes) > 1 else 0.0,
        "mae_mean":                float(np.mean(maes)) if maes else float("nan"),
        "mse_kw2_mean":            float(np.mean(mses)) if mses else float("nan"),
        "hr@1_mean":               float(np.mean(hr1s)) if hr1s else float("nan"),
        "hr@2_mean":               float(np.mean(hr2s)) if hr2s else float("nan"),
        "hr@3_mean":               float(np.mean(hr3s)) if hr3s else float("nan"),
        "vq_util_mean":            util_sum / max(n_chunks, 1),
        "vq_ppl_mean":             ppl_sum  / max(n_chunks, 1),
        "n_clients":               int(len(papes)),
    }


# ---------------------------------------------------------------------------
# VQ-specific server aggregation (the whole point of this driver).
# ---------------------------------------------------------------------------


_VQ_CODEBOOK_KEY   = "stack_generic.vq.codebook"
_VQ_EMA_COUNT_KEY  = "stack_generic.vq.ema_count"
_VQ_EMA_WEIGHT_KEY = "stack_generic.vq.ema_weight"


@torch.no_grad()
def _extract_h_g_from_dict_model(
    model: NBEATSxAuxVQ,
    x: np.ndarray,
    *,
    batch_size: int,
    use_amp: bool,
) -> np.ndarray:
    """Forward `x` through a (random-init) NBEATSxAuxVQ and return h_generic.

    Bypasses src/fl/codebook_fl.py's `_extract_h_g_from_windows` because that
    helper assumes NBEATSxAux's tuple forward signature; our dict-returning
    wrapper needs a thin local extractor.
    """
    if x.shape[0] == 0:
        return np.zeros((0, D_MODEL), dtype=np.float32)
    model.eval()
    chunks = []
    for i in range(0, int(x.shape[0]), batch_size):
        xb = torch.from_numpy(x[i : i + batch_size]).to(DEVICE, non_blocking=True)
        with _wrap_amp(use_amp):
            out = model(xb)
        chunks.append(out["h_generic"].float().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _federated_kmeans_init_codebook(
    model: NBEATSxAuxVQ,
    splits: dict[str, dict],
    *,
    M: int,
    K_local: int,
    seed: int,
    batch_size: int,
    use_amp: bool,
) -> torch.Tensor:
    """Round-0 federated KMeans++ init for the codebook.

    Mirrors v05/v06 FedCB Stage-1+Stage-2 but inlined here so the driver
    stays self-contained:
      Stage 1 (per client): KMeans++(h_g, K_local) → (centroids, counts).
      Stage 2 (server):     weighted KMeans++(stacked centroids, M, sample_weight=counts).

    Federation-safe: raw h_g never leaves the client conceptually (here we
    extract on a single process for the simulation, but the *contract* is
    that only (centroids, counts) tuples upload).

    Returns: (M, D) fp32 codebook tensor on CPU. Falls back to None if every
    client has empty train_x (caller keeps the random init).
    """
    packets = []
    for sp in splits.values():
        h_g = _extract_h_g_from_dict_model(
            model, sp["train_x"], batch_size=batch_size, use_amp=use_amp,
        )
        if h_g.shape[0] == 0:
            continue
        K_eff = min(K_local, h_g.shape[0])
        if K_eff < 1:
            continue
        km = KMeans(
            n_clusters=K_eff, init="k-means++", n_init=3, random_state=seed
        ).fit(h_g)
        packets.append((
            km.cluster_centers_.astype(np.float32),
            np.bincount(km.labels_, minlength=K_eff).astype(np.int64),
        ))
    if not packets:
        return None
    P = np.vstack([c for c, _ in packets]).astype(np.float32)
    w = np.concatenate([cnts for _, cnts in packets]).astype(np.float64)
    if P.shape[0] < M:
        raise ValueError(
            f"_federated_kmeans_init_codebook: only {P.shape[0]} input centroids "
            f"vs M={M}; raise K_local or lower M."
        )
    km = KMeans(
        n_clusters=int(M), init="k-means++", n_init=10, random_state=seed,
    ).fit(P, sample_weight=w)
    return torch.from_numpy(km.cluster_centers_.astype(np.float32))


def _aggregate_vq_buffers(
    prev_codebook: torch.Tensor,
    local_states: list[dict[str, torch.Tensor]],
    ema_gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Plan v09-01 §3 mass-weighted aggregation + EMA blending for the codebook.

    Args:
        prev_codebook:  (M, D) — server codebook at the **start** of this round.
        local_states:   list of CPU state dicts uploaded by each client.
        ema_gamma:      blending weight on the previous codebook (default 0.95).

    Returns:
        new_codebook:   (M, D) — γ · prev + (1−γ) · mass-weighted client centroid;
                        dead entries (total_mass == 0) keep ``prev_codebook[c]``.
        new_ema_count:  (M,)   — Σ_i ema_count_local_i.
        new_ema_weight: (M, D) — Σ_i ema_weight_local_i.
    """
    # Stack client buffers (all CPU, float). Shapes: [N, M] and [N, M, D].
    counts = torch.stack(
        [sd[_VQ_EMA_COUNT_KEY].detach().to(torch.float64).cpu() for sd in local_states],
        dim=0,
    )
    weights = torch.stack(
        [sd[_VQ_EMA_WEIGHT_KEY].detach().to(torch.float64).cpu() for sd in local_states],
        dim=0,
    )
    new_ema_count  = counts.sum(dim=0)          # [M]
    new_ema_weight = weights.sum(dim=0)         # [M, D]

    prev = prev_codebook.detach().to(torch.float64).cpu()
    # Mass-weighted centroid per entry; safe where total_mass == 0 (fill prev).
    total_mass = new_ema_count.clamp_min(0.0)   # already non-negative, defensive
    mass_safe  = total_mass.clone()
    mass_safe[mass_safe <= 0] = 1.0             # avoid div-by-zero; overwritten below
    new_centroid = new_ema_weight / mass_safe.unsqueeze(1)
    dead_mask = (total_mass <= 0).unsqueeze(1)  # [M, 1]
    new_centroid = torch.where(dead_mask, prev, new_centroid)

    new_codebook = ema_gamma * prev + (1.0 - ema_gamma) * new_centroid
    # Dead entries: keep prev exactly (the blend above already equals prev
    # because new_centroid == prev there, but be explicit so the contract is
    # readable).
    new_codebook = torch.where(dead_mask, prev, new_codebook)

    return (
        new_codebook.to(prev_codebook.dtype),
        new_ema_count.to(torch.float32),
        new_ema_weight.to(prev_codebook.dtype),
    )


def _respawn_dead_codes(
    codebook: torch.Tensor,
    ema_count: torch.Tensor,
    ema_weight: torch.Tensor,
    *,
    n_min: float,
    rng: torch.Generator,
    noise_scale: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Replace under-used codebook entries with perturbed copies of the most-used one.

    Args:
        codebook:   (M, D) post-aggregation codebook (CPU).
        ema_count:  (M,)
        ema_weight: (M, D)
        n_min:      respawn threshold; any entry with ema_count[c] < n_min is
                    eligible for replacement.
        rng:        torch.Generator for deterministic noise.
        noise_scale: stddev of additive Gaussian noise relative to nothing
                    (multiplied directly — codebook entries are unitless z-space).

    Returns:
        (codebook, ema_count, ema_weight, n_respawned) — modified in-place
        copies (CPU, original dtypes).
    """
    cb  = codebook.clone()
    ec  = ema_count.clone()
    ew  = ema_weight.clone()
    dead_idx = torch.nonzero(ec < n_min, as_tuple=False).flatten()
    if dead_idx.numel() == 0:
        return cb, ec, ew, 0
    # Source = most-used entry. If all entries are dead (unlikely with mass-weighted
    # aggregation but defensive) fall back to entry 0.
    src = int(ec.argmax().item())
    src_vec = cb[src].clone()
    for c in dead_idx.tolist():
        noise = torch.randn(cb.shape[1], generator=rng, dtype=cb.dtype) * noise_scale
        cb[c]  = src_vec + noise
        ec[c]  = 1.0
        ew[c]  = cb[c].clone()
    return cb, ec, ew, int(dead_idx.numel())


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "v09 FedVQ — NBEATSxAux + federated VQ codebook on h_generic. "
            "Single FedAvg backbone + mass-weighted EMA-blended codebook aggregation "
            "+ dead-code respawn. Single seed × single cell per invocation."
        )
    )
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--local_epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--num_embeddings", type=int, default=32,
                    help="VQ codebook size M (v06 invariant).")
    ap.add_argument("--commitment_beta", type=float, default=0.25,
                    help="VQ-VAE commitment loss weight (van den Oord 2017).")
    ap.add_argument("--use_aux_head", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Enable peak-aux head and λ·peak_aux loss term.")
    ap.add_argument("--aux_lambda", type=float, default=0.3,
                    help="Weight on peak_aux loss (v06 default 0.3).")
    ap.add_argument("--hr_weight", type=float, default=0.1,
                    help="Inner hour-CE weight inside peak_aux (v06 default 0.1).")
    ap.add_argument("--ema_gamma", type=float, default=0.95,
                    help="Server codebook EMA blend: γ·prev + (1−γ)·new_centroid.")
    ap.add_argument("--init_mode", type=str, default="fedkmeans",
                    choices=["random", "fedkmeans"],
                    help="Round-0 codebook init. 'random' = VectorQuantizerEMA's "
                         "default randn*0.1 (collapse-prone). 'fedkmeans' = each "
                         "client runs local KMeans++ on its train h_g, server "
                         "merges with weighted KMeans++ to M centroids — same "
                         "pattern as v05/v06 FedCB Stage-1+Stage-2.")
    ap.add_argument("--init_K_local", type=int, default=2,
                    help="Per-client KMeans cluster count for fedkmeans init "
                         "(CLAUDE.md K_local default = 2; ignored if init_mode=random).")
    ap.add_argument("--respawn_period", type=int, default=5,
                    help="Run dead-code respawn every K rounds (1 disables periodicity gate; "
                         "0 disables respawn entirely).")
    ap.add_argument("--respawn_n_min", type=float, default=1.0,
                    help="ema_count threshold below which a code is respawned.")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--log_test_per_round", action="store_true",
                    help="Also evaluate the TEST split at the end of every round and "
                         "log it as row['test'] in round_log.jsonl. Off by default "
                         "(preserves legacy behaviour: only terminal test). Enables a "
                         "per-round TEST PAPE trajectory directly comparable to the "
                         "RoundCB (03) post-hoc curve.")
    ap.add_argument("--output_namespace", type=str, default="v09_round_vq_codebook")
    args = ap.parse_args()

    use_amp = not args.no_amp
    init_suffix = "-fedinit" if args.init_mode == "fedkmeans" else ""
    cell = ("V9-FedVQ-aux" if args.use_aux_head else "V9-FedVQ-noaux") + init_suffix
    out_dir = OUTPUT_DIR / args.output_namespace / f"seed{args.seed}" / cell
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "round_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    print(f"[{cell}] seed={args.seed}  R={args.rounds}  E={args.local_epochs}  "
          f"batch={args.batch_size}  M={args.num_embeddings}  beta={args.commitment_beta}  "
          f"γ={args.ema_gamma}  respawn_K={args.respawn_period}  amp={use_amp}")
    print(f"[{cell}] out_dir={out_dir}")

    # 1) Per-client splits (v06/v08 cache reused).
    print(f"[{cell}] building per-client 70/10/20 splits ...")
    splits = build_per_client_splits(seed=args.seed)
    n_clients = len(splits)
    print(f"[{cell}] {n_clients} apartments retained.")

    client_loaders: OrderedDict[str, DataLoader] = OrderedDict()
    client_weights: OrderedDict[str, float] = OrderedDict()
    for apt, sp in splits.items():
        ds = TensorDataset(
            torch.from_numpy(sp["train_x"]),
            torch.from_numpy(sp["train_y"]),
        )
        client_loaders[apt] = DataLoader(
            ds, batch_size=args.batch_size, shuffle=True, drop_last=False
        )
        client_weights[apt] = float(sp["train_x"].shape[0])

    # 2) Server init.
    server_model = _init_model(
        args.seed, args.num_embeddings, args.commitment_beta, args.use_aux_head
    )
    global_state = clone_state_dict(server_model.state_dict())

    # 2b) Round-0 codebook init. The default VectorQuantizerEMA init
    # (`torch.randn(M, D) * 0.1`) sits far from the fc4 output distribution,
    # so round 1's first forward routes most of the batch to a single entry
    # and triggers winner-take-all (observed: ema_count_top1_share = 0.90
    # at R1 with random init). Federated KMeans++ on each client's h_g
    # seeds the codebook in the actual representation manifold.
    if args.init_mode == "fedkmeans":
        print(f"[{cell}] init_mode=fedkmeans: federated KMeans++ init "
              f"(K_local={args.init_K_local}, M={args.num_embeddings}) ...")
        new_codebook = _federated_kmeans_init_codebook(
            server_model, splits,
            M=args.num_embeddings, K_local=args.init_K_local, seed=args.seed,
            batch_size=args.batch_size, use_amp=use_amp,
        )
        if new_codebook is None:
            print(f"[{cell}] WARNING: fedkmeans init produced no packets; "
                  f"falling back to random init.")
        else:
            global_state[_VQ_CODEBOOK_KEY] = new_codebook.to(
                global_state[_VQ_CODEBOOK_KEY].dtype
            )
            # Match ema_weight to the new codebook so the first-round EMA
            # blend doesn't drag everything back toward the stale randn init.
            global_state[_VQ_EMA_WEIGHT_KEY] = new_codebook.to(
                global_state[_VQ_EMA_WEIGHT_KEY].dtype
            )
            global_state[_VQ_EMA_COUNT_KEY] = torch.ones_like(
                global_state[_VQ_EMA_COUNT_KEY]
            )
            apply_state_dict(server_model, global_state)
            print(f"[{cell}] fedkmeans init done. codebook L2 norm "
                  f"mean={new_codebook.pow(2).sum(dim=1).sqrt().mean().item():.3f}")

    # RNG for deterministic respawn noise.
    respawn_rng = torch.Generator()
    respawn_rng.manual_seed(args.seed + 7919)  # arbitrary offset; kept seed-determined

    history: list[dict] = []
    cb_history: dict[str, list] = {
        "rounds": [], "codebook": [], "ema_count": [], "ema_weight": [],
    }
    prev_cb: torch.Tensor | None = None
    t0 = time.time()

    for r in range(1, args.rounds + 1):
        t_round = time.time()
        local_states: list[dict] = []
        local_weights: list[float] = []
        round_main_sum = round_aux_sum = round_commit_sum = 0.0
        round_util_sum = round_ppl_sum = 0.0
        round_batches = 0

        # Server-side prev codebook for this round (for EMA blend).
        prev_codebook_for_blend = global_state[_VQ_CODEBOOK_KEY].detach().clone()

        for apt, loader in client_loaders.items():
            apply_state_dict(server_model, global_state)
            # ----- Round-start EMA reset (mechanism 1) -----
            # Zero ema_count / ema_weight so this client's local EMA accumulates
            # only this round's batches. codebook stays for routing.
            with torch.no_grad():
                server_model.stack_generic.vq.ema_count.zero_()
                server_model.stack_generic.vq.ema_weight.zero_()
            # ------------------------------------------------
            optimizer = torch.optim.Adam(
                server_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
            diag = _local_train_one_client(
                server_model, loader, optimizer,
                n_epochs=args.local_epochs, use_amp=use_amp,
                use_aux_head=args.use_aux_head,
                aux_lambda=args.aux_lambda, hr_weight=args.hr_weight,
            )
            local_states.append(clone_state_dict(server_model.state_dict()))
            local_weights.append(client_weights[apt])
            round_main_sum   += diag["main_loss_mean"]   * diag["n_batches"]
            round_aux_sum    += diag["aux_loss_mean"]    * diag["n_batches"]
            round_commit_sum += diag["commit_loss_mean"] * diag["n_batches"]
            round_util_sum   += diag["vq_util_mean"]     * diag["n_batches"]
            round_ppl_sum    += diag["vq_ppl_mean"]      * diag["n_batches"]
            round_batches    += diag["n_batches"]

        # ----- Backbone aggregation: standard FedAvg by n_train_windows -----
        # This averages every float key including the VQ buffers. We then
        # overwrite the three VQ keys with mechanism-2 outputs below.
        global_state = weighted_average(local_states, local_weights)

        # ----- VQ aggregation (mechanism 2): mass-weighted + EMA-blended -----
        new_cb, new_ec, new_ew = _aggregate_vq_buffers(
            prev_codebook_for_blend, local_states, ema_gamma=args.ema_gamma,
        )

        # ----- Dead-code respawn (mechanism 3): every respawn_period rounds -----
        n_respawned = 0
        if args.respawn_period > 0 and (r % args.respawn_period == 0):
            new_cb, new_ec, new_ew, n_respawned = _respawn_dead_codes(
                new_cb, new_ec, new_ew,
                n_min=float(args.respawn_n_min), rng=respawn_rng,
            )

        # Write back into the (CPU) global_state and load.
        global_state[_VQ_CODEBOOK_KEY]   = new_cb
        global_state[_VQ_EMA_COUNT_KEY]  = new_ec
        global_state[_VQ_EMA_WEIGHT_KEY] = new_ew
        apply_state_dict(server_model, global_state)

        # ----- Snapshot + per-round diagnostics -----
        cb_now  = global_state[_VQ_CODEBOOK_KEY].detach().cpu().clone()
        ema_cnt = global_state[_VQ_EMA_COUNT_KEY].detach().cpu().clone()
        ema_w   = global_state[_VQ_EMA_WEIGHT_KEY].detach().cpu().clone()
        cb_history["rounds"].append(r)
        cb_history["codebook"].append(cb_now)
        cb_history["ema_count"].append(ema_cnt)
        cb_history["ema_weight"].append(ema_w)
        cb_drift = (
            float((cb_now - prev_cb).pow(2).sum().sqrt().item())
            if prev_cb is not None else 0.0
        )
        cnt_total = float(ema_cnt.sum().item())
        server_vq = {
            "codebook_drift_l2":    cb_drift,
            "ema_count_top1_share": float(ema_cnt.max().item() / cnt_total) if cnt_total > 0 else 0.0,
            "ema_count_active":     int((ema_cnt > 1e-3).sum().item()),
            "n_respawned":          int(n_respawned),
        }
        prev_cb = cb_now

        val_metrics = _eval_per_client(
            server_model, splits, "val",
            batch_size=args.batch_size, use_amp=use_amp,
        )
        test_metrics_round = (
            _eval_per_client(
                server_model, splits, "test",
                batch_size=args.batch_size, use_amp=use_amp,
            )
            if args.log_test_per_round else None
        )
        wall = time.time() - t_round
        row = {
            "round": r,
            "wall_seconds": float(wall),
            "train": {
                "main_loss_mean":   round_main_sum   / max(round_batches, 1),
                "aux_loss_mean":    round_aux_sum    / max(round_batches, 1),
                "commit_loss_mean": round_commit_sum / max(round_batches, 1),
                "vq_util_mean":     round_util_sum   / max(round_batches, 1),
                "vq_ppl_mean":      round_ppl_sum    / max(round_batches, 1),
                "n_batches":        int(round_batches),
            },
            "server_vq": server_vq,
            "val": val_metrics,
        }
        if test_metrics_round is not None:
            row["test"] = test_metrics_round
        history.append(row)
        with log_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(
            f"  round {r:3d}: main={row['train']['main_loss_mean']:.4f}  "
            f"aux={row['train']['aux_loss_mean']:.4f}  "
            f"commit={row['train']['commit_loss_mean']:.4f}  "
            f"util={row['train']['vq_util_mean']:.2f}  "
            f"ppl={row['train']['vq_ppl_mean']:.2f}  "
            f"cb_drift={cb_drift:.4f}  respawn={n_respawned}  "
            f"val.PAPE={val_metrics['pape_mean']:.2f}  wall={wall:.1f}s"
        )

    # Terminal test.
    test_metrics = _eval_per_client(
        server_model, splits, "test",
        batch_size=args.batch_size, use_amp=use_amp,
    )
    elapsed = time.time() - t0

    # Persist final outputs.
    torch.save(global_state, out_dir / "final_state_dict.pt")
    if cb_history["rounds"]:
        torch.save({
            "rounds":     cb_history["rounds"],
            "codebook":   torch.stack(cb_history["codebook"]),    # (R, M, D)
            "ema_count":  torch.stack(cb_history["ema_count"]),   # (R, M)
            "ema_weight": torch.stack(cb_history["ema_weight"]),  # (R, M, D)
        }, out_dir / "codebook_history.pt")
    result = {
        "cell": cell,
        "seed": int(args.seed),
        "n_clients": n_clients,
        "rounds": int(args.rounds),
        "local_epochs": int(args.local_epochs),
        "batch": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "num_embeddings": int(args.num_embeddings),
        "commitment_beta": float(args.commitment_beta),
        "use_aux_head": bool(args.use_aux_head),
        "aux_lambda": float(args.aux_lambda),
        "hr_weight": float(args.hr_weight),
        "ema_gamma": float(args.ema_gamma),
        "respawn_period": int(args.respawn_period),
        "respawn_n_min": float(args.respawn_n_min),
        "use_amp": bool(use_amp),
        "history": history,
        "val_terminal":  history[-1]["val"] if history else None,
        "test_terminal": test_metrics,
        "elapsed_seconds": float(elapsed),
        "comment": (
            "v09 plan §3 minimal-viable FedVQ. Round-start EMA reset + "
            "mass-weighted codebook aggregation with EMA blending + "
            "dead-code respawn every respawn_period rounds. Backbone uses "
            "standard FedAvg weighted by n_train_windows."
        ),
    }
    with (out_dir / "result.json").open("w") as fh:
        json.dump(result, fh, indent=2)
    print(
        f"[{cell}] done. test.PAPE={test_metrics['pape_mean']:.2f}  "
        f"util={test_metrics['vq_util_mean']:.2f}  "
        f"ppl={test_metrics['vq_ppl_mean']:.2f}  elapsed={elapsed:.0f}s"
    )


if __name__ == "__main__":
    main()
