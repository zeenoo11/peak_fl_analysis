"""Priority 1A — FedAvg with NBEATSxAux (backbone + aux_head federated jointly).

Why
---
The unified pFL paper headline is "v01-v03 method outperforms every v04
baseline by 11.8 kW PAPE". That gap conflates at least three independent
factors:

    1. centralised pretraining vs federated training
    2. with vs without auxiliary head (peak_aux MSE+CE)
    3. with vs without Peak-VQ + W5 hybrid correction

The parent v04 folder federates *MinimalNBEATSx only* (no aux head) and
applies Peak-VQ + W5 only as a separate G5 cross-cell on top of FedAvg /
FedRep. This script adds the missing ceteris-paribus row: federate the
**entire** NBEATSxAux (both backbone weights AND aux_head weights are
averaged across clients each round), so that the resulting (a) "fl_only"
metric isolates "centralised vs federated, with aux head" from
"with vs without aux head", and (b) the "with_codebook_*" metrics give
the FL-trained equivalent of the v01-v03 method.

Three result blocks (all on the same cold pool, single FL training pass):

- ``fl_only``                       — federated NBEATSxAux backbone, no
                                       Peak-VQ. This is the **Priority 1A**
                                       row asked for in the planning brief.
- ``with_codebook_HR_preserving``   — Phase C, W5 hybrid at v01 op-point 1.
                                       (â, ĥ) come from the *aux head*
                                       (NOT self-derived from ŷ) — the
                                       federated backbone learned an aux
                                       head, so we use it; this is exactly
                                       the v02 04_coldstart_eval.py flow.
- ``with_codebook_PAPE_aggressive`` — Phase C, W5 hybrid at v01 op-point 2.
                                       This is the "v01-v03 method but with
                                       FL training" comparison row.

Operating points are carried over from v01 §4.2 unchanged (CLAUDE.md
convention: do NOT re-tune (σ, α_v0, α_w1) on the cold split).

Loss
----
Each client's local SGD step uses the **same combined loss** the v01-v04
method uses for centralised training (CLAUDE.md §"Method"):

    L = MAE(y_hat, y) + λ * peak_aux_loss(amp_pred, hr_pred, y, hr_weight=0.1)

with ``λ = 0.3`` (project default per CLAUDE.md). The federated round loop
is otherwise FedAvg (full participation, weighted average by n_train_windows).

CLI
---
    uv run python experiments/v04_full_baseline_comparison/09_fix_rerun/02_fedavg_nbeatsx_aux.py --seed 42

Output
------
    outputs/v04_full_baseline_comparison/09_fix_rerun/seed{S}/fedavg_nbeatsx_aux/
        result.json
            ├── fl_only                     {pape, hr@1, hr@2, mae, n_cold_windows, n_cold_apts}
            ├── with_codebook_HR_preserving {sigma, alpha_v0, alpha_w1, metrics: {...}}
            ├── with_codebook_PAPE_aggressive {sigma, alpha_v0, alpha_w1, metrics: {...}}
            ├── vq_diagnostics              {utilization, perplexity, k_min, k_max, n_empty_clusters}
            ├── history (FL training)
            ├── config, seed, elapsed_seconds, gpu_at_start/end
        final_state_dict.pt   (NBEATSxAux full state dict; loadable into NBEATSxAux directly)

Wall-clock per seed: ~15 min on a 5070 Ti
    (FedAvg ~12 min + Phase B/C ~2 min + boilerplate ~1 min).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import HORIZON, OUTPUT_DIR, RANDOM_SEED, TRAIN_RATIO
from dataloader.splits import load_v02_split
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from eval.cold_helpers import (
    OPERATING_POINTS,
    gather_cold,
    gauss_template,
    metrics_z_to_kw,
    route_R0,
)
from fl.base import (
    DEVICE,
    ClientData,
    apply_state_dict,
    build_clients,
    client_loader,
    clone_state_dict,
    weighted_average,
)
from models.nbeatsx_aux import NBEATSxAux
from models.peak_aux_head import peak_aux_loss
from models.vq_kmeans import VectorQuantizerKMeans
from probes.peak_descriptor import extract_key

V04_FIX_OUT_ROOT = OUTPUT_DIR / "v04_full_baseline_comparison" / "09_fix_rerun"


def _gpu_snapshot() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        used, free, total, util = (int(s.strip()) for s in out.strip().split(","))
        return {"used_MiB": used, "free_MiB": free, "total_MiB": total, "util_pct": util}
    except Exception:
        return {"cpu_only": not torch.cuda.is_available()}


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def init_backbone_aux(seed: int) -> NBEATSxAux:
    """Seeded NBEATSxAux init. Mirrors fl.base.init_backbone but with aux head.

    We deliberately do NOT add this to ``src/fl/base.py`` because the rest of
    the v04 FL infrastructure assumes the 2-tuple forward of MinimalNBEATSx;
    adding NBEATSxAux as a peer initialiser would invite accidental misuse
    in the parent folder. This script keeps the helper local so the FL
    infrastructure for the parent folder is untouched.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    return NBEATSxAux(latent_source="h_generic").to(DEVICE)


def _local_step_aux(
    model: NBEATSxAux,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    use_amp: bool,
    aux_lambda: float = 0.3,
    hr_weight: float = 0.1,
) -> dict:
    """Run ``n_epochs`` local SGD on this client with the combined loss.

    L = MAE(y_hat, y) + aux_lambda * peak_aux_loss(amp_pred, hr_pred, y, hr_weight)
    """
    use_amp = use_amp and (DEVICE.type == "cuda")
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp else _NullCtx()
    )
    model.train()
    n_batches = 0
    sum_main = 0.0
    sum_aux = 0.0
    for _ in range(n_epochs):
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                y_hat, _hiddens, (amp_pred, hr_pred) = model(x)
                main = F.l1_loss(y_hat, y)
                aux = peak_aux_loss(amp_pred, hr_pred, y, hr_weight=hr_weight)
                loss = main + aux_lambda * aux
            loss.backward()
            optimizer.step()
            sum_main += float(main.item())
            sum_aux += float(aux.item())
            n_batches += 1
    return {
        "n_batches": n_batches,
        "main_loss_mean": sum_main / max(n_batches, 1),
        "aux_loss_mean": sum_aux / max(n_batches, 1),
    }


def fedavg_aux_round_loop(
    train_apts: list[str],
    cold_apts: list[str],
    *,
    rounds: int,
    local_epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    seed: int,
    use_amp: bool,
    aux_lambda: float,
    hr_weight: float,
) -> dict:
    """Phase A — federated training of NBEATSxAux with FedAvg.

    Returns a dict with ``history`` + ``final_state_dict`` (CPU). Cold
    metrics are NOT computed here — Phase C below handles cold inference
    after Peak-VQ fit.
    """
    clients: list[ClientData] = build_clients(train_apts)
    if not clients:
        raise RuntimeError("FedAvg-aux: no train clients (all apts missing?)")

    global_model = init_backbone_aux(seed=seed)
    global_state = clone_state_dict(global_model.state_dict())

    history: dict = {"rounds": [], "main_loss": [], "aux_loss": [], "n_clients": []}
    for r in range(1, rounds + 1):
        local_states: list[dict] = []
        local_weights: list[float] = []
        round_main_sum, round_aux_sum, round_n = 0.0, 0.0, 0
        for client in clients:
            apply_state_dict(global_model, global_state)
            optimizer = torch.optim.Adam(
                global_model.parameters(), lr=lr, weight_decay=weight_decay
            )
            loader = client_loader(client, batch_size, shuffle=True)
            diag = _local_step_aux(
                global_model, loader, optimizer,
                n_epochs=local_epochs, use_amp=use_amp,
                aux_lambda=aux_lambda, hr_weight=hr_weight,
            )
            local_states.append(clone_state_dict(global_model.state_dict()))
            local_weights.append(float(client.n_train_windows))
            round_main_sum += diag["main_loss_mean"] * diag["n_batches"]
            round_aux_sum += diag["aux_loss_mean"] * diag["n_batches"]
            round_n += diag["n_batches"]
        global_state = weighted_average(local_states, local_weights)
        history["rounds"].append(r)
        history["main_loss"].append(round_main_sum / max(round_n, 1))
        history["aux_loss"].append(round_aux_sum / max(round_n, 1))
        history["n_clients"].append(len(clients))
        print(f"  round {r:2d}: main={history['main_loss'][-1]:.4f}  "
              f"aux={history['aux_loss'][-1]:.4f}  n_clients={len(clients)}")

    return {
        "history": history,
        "final_state_dict": global_state,
        "n_train_clients": len(clients),
    }


def gather_train_segment_aux(
    apts: list[str],
    model: NBEATSxAux,
    batch: int,
    stride: int,
) -> dict[str, np.ndarray]:
    """Phase B helper — collect (h_g, y_hat_z, y_true_z, key) on each apt's
    train segment. Same shape as ``v02 03_fit_codebook.gather_train_segment``,
    just kept local to this script so the parent helpers stay untouched.
    """
    h_chunks, yhat_chunks, ytrue_chunks, key_chunks = [], [], [], []
    n_per_apt: list[int] = []
    model.eval()
    for apt in apts:
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            continue
        n = len(series)
        train_end = int(n * TRAIN_RATIO)
        seg = series[:train_end]
        m_ = float(seg.mean())
        s_ = float(seg.std()) if seg.std() > 1e-8 else 1.0
        ds = HouseholdDataset(seg, m_, s_, stride=stride)
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=batch, shuffle=False)
        per_apt = 0
        for x, y in loader:
            x_dev = x.to(DEVICE)
            with torch.no_grad():
                y_hat, hiddens, _aux = model(x_dev)
            h_chunks.append(hiddens["h_generic"].cpu().numpy())
            yhat_chunks.append(y_hat.cpu().numpy())
            ytrue_chunks.append(y.numpy())
            key_chunks.append(extract_key(x.numpy()))
            per_apt += len(x)
        n_per_apt.append(per_apt)
    return {
        "h_g": np.concatenate(h_chunks, axis=0).astype(np.float32),
        "y_hat_z": np.concatenate(yhat_chunks, axis=0).astype(np.float32),
        "y_true_z": np.concatenate(ytrue_chunks, axis=0).astype(np.float32),
        "key": np.concatenate(key_chunks, axis=0).astype(np.float32),
        "n_windows_per_apt": np.asarray(n_per_apt, dtype=np.int64),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="v04 09_fix_rerun: FedAvg with NBEATSxAux (Priority 1A).")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--aux_lambda", type=float, default=0.3,
                    help="Combined-loss weight on peak_aux (CLAUDE.md default = 0.3).")
    ap.add_argument("--hr_weight", type=float, default=0.1,
                    help="peak_aux internal hr-CE weight (CLAUDE.md default = 0.1).")
    ap.add_argument("--M", type=int, default=32, help="Codebook size for Phase B.")
    ap.add_argument("--stride_phase_bc", type=int, default=HORIZON,
                    help="Stride for Phase B (codebook fit) and Phase C (cold eval). v01/v02 default.")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    use_amp = not args.no_amp

    sp = load_v02_split(args.seed)
    train_apts, cold_apts = sp["train"], sp["cold"]
    out_dir = V04_FIX_OUT_ROOT / f"seed{args.seed}" / "fedavg_nbeatsx_aux"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v04 1A] seed={args.seed}  rounds={args.rounds}  "
          f"local_epochs={args.local_epochs}  batch={args.batch_size}  amp={use_amp}")
    gpu_start = _gpu_snapshot()
    print(f"[v04 1A] GPU @start: {gpu_start}")

    # ===== Phase A: federated training of NBEATSxAux =====
    t0 = time.time()
    fa = fedavg_aux_round_loop(
        train_apts, cold_apts,
        rounds=args.rounds, local_epochs=args.local_epochs,
        lr=args.lr, batch_size=args.batch_size, weight_decay=args.weight_decay,
        seed=args.seed, use_amp=use_amp,
        aux_lambda=args.aux_lambda, hr_weight=args.hr_weight,
    )
    fl_elapsed = time.time() - t0
    print(f"[v04 1A] FL training done in {fl_elapsed:.0f}s ({fl_elapsed/60:.1f} min)")

    # Load federated weights into a fresh model for downstream phases.
    model = NBEATSxAux(latent_source="h_generic").to(DEVICE).eval()
    apply_state_dict(model, fa["final_state_dict"])

    # ---- fl_only cold metrics: federated backbone, no Peak-VQ ----
    co = gather_cold(cold_apts, model, batch=args.batch_size, stride=args.stride_phase_bc, verbose_skips=False)
    fl_only_metrics = metrics_z_to_kw(co["y_true_z"], co["y_hat_z"], co["mean"], co["std"])
    fl_only_metrics["n_cold_windows"] = int(co["y_true_z"].shape[0])
    fl_only_metrics["n_cold_apts"] = int(len(np.unique(co["apt"])))
    print(f"[v04 1A] fl_only:  PAPE={fl_only_metrics['pape']:.2f}  "
          f"HR@1={fl_only_metrics['hr@1']:.1f}  HR@2={fl_only_metrics['hr@2']:.1f}")

    # ===== Phase B: Peak-VQ codebook fit on federated h_generic =====
    t_b = time.time()
    tr = gather_train_segment_aux(train_apts, model, batch=args.batch_size, stride=args.stride_phase_bc)
    print(f"[v04 1A] Phase B: {tr['h_g'].shape[0]} train windows for codebook fit")

    vq = VectorQuantizerKMeans(num_embeddings=args.M, embedding_dim=tr["h_g"].shape[1], random_state=args.seed)
    diag = vq.fit(torch.from_numpy(tr["h_g"]).float())
    print(f"[v04 1A] vq diag: util={diag['utilization']:.3f}  ppl={diag['perplexity']:.2f}  "
          f"k_min={diag['k_min']}  k_max={diag['k_max']}")

    centroids = vq.codebook.cpu().numpy()
    counts = vq.counts.cpu().numpy()
    h_t = torch.from_numpy(tr["h_g"]).float()
    with torch.no_grad():
        _, idx_t = vq(h_t)
    cluster_idx = idx_t.cpu().numpy().astype(np.int64)
    residuals = tr["y_true_z"] - tr["y_hat_z"]
    offsets = np.zeros((args.M, residuals.shape[1]), dtype=np.float32)
    for c in range(args.M):
        mask = cluster_idx == c
        if mask.any():
            offsets[c] = residuals[mask].mean(axis=0)

    key_pool = tr["key"].astype(np.float32)
    key_scaler = StandardScaler().fit(key_pool)
    key_pool_scaled = key_scaler.transform(key_pool).astype(np.float32)
    print(f"[v04 1A] Phase B done in {time.time() - t_b:.1f}s")

    # ===== Phase C: cold inference at both v01 op-points (R0 routing) =====
    cold_cluster = route_R0(
        co["key"],
        key_scaler.mean_.astype(np.float32),
        key_scaler.scale_.astype(np.float32),
        key_pool_scaled,
        cluster_idx,
    )
    cluster_offset = offsets[cold_cluster]

    out_per_op: dict = {}
    for op_name, op in OPERATING_POINTS.items():
        # (â, ĥ) come from the AUX HEAD — the federated backbone has a learned aux head
        # so we use its predictions directly (NOT y_hat-self-derived). This mirrors v02
        # 04_coldstart_eval.py exactly; the difference vs 04_peakvq_on_fl.py is that
        # 04_peakvq_on_fl runs on a FedAvg-trained MinimalNBEATSx (no aux head) and
        # therefore has to self-derive — we do not.
        g = gauss_template(co["pred_hr"], co["pred_amp"], sigma=op["sigma"])
        corrected_z = (
            co["y_hat_z"]
            + op["alpha_v0"] * cluster_offset
            + op["alpha_w1"] * g
        ).astype(np.float32)
        m = metrics_z_to_kw(co["y_true_z"], corrected_z, co["mean"], co["std"])
        delta = fl_only_metrics["pape"] - m["pape"]
        ratio = m["pape"] / fl_only_metrics["pape"] if fl_only_metrics["pape"] > 0 else float("nan")
        print(f"[v04 1A] {op_name:<16}: PAPE={m['pape']:.2f} (Δ={delta:+.2f} kW vs fl_only; ratio={ratio:.3f})")
        out_per_op[op_name] = {
            "sigma": op["sigma"], "alpha_v0": op["alpha_v0"], "alpha_w1": op["alpha_w1"],
            "metrics": m,
            "pape_delta_kw_vs_fl_only": delta,
            "pape_ratio_vs_fl_only": ratio,
        }

    # aux head accuracy on cold (mirrors v02 04 aux_diagnostics).
    cold_true_hr = co["y_true_z"].argmax(axis=1)
    aux_diag = {
        "top1": float((co["pred_hr"] == cold_true_hr).mean()),
        "within_1h": float((np.abs(co["pred_hr"] - cold_true_hr) <= 1).mean()),
        "within_2h": float((np.abs(co["pred_hr"] - cold_true_hr) <= 2).mean()),
    }

    elapsed = time.time() - t0
    gpu_end = _gpu_snapshot()

    out = {
        "algorithm": "fedavg_nbeatsx_aux",
        "seed": int(args.seed),
        "config": {
            "rounds": args.rounds, "local_epochs": args.local_epochs,
            "lr": args.lr, "batch_size": args.batch_size,
            "weight_decay": args.weight_decay, "use_amp": use_amp,
            "aux_lambda": args.aux_lambda, "hr_weight": args.hr_weight,
            "M": args.M, "stride_phase_bc": args.stride_phase_bc,
        },
        "history": fa["history"],
        "fl_only": fl_only_metrics,
        "with_codebook_HR_preserving":   out_per_op["HR-preserving"],
        "with_codebook_PAPE_aggressive": out_per_op["PAPE-aggressive"],
        "vq_diagnostics": {
            "utilization": float(diag["utilization"]),
            "perplexity": float(diag["perplexity"]),
            "k_min": int(diag["k_min"]),
            "k_max": int(diag["k_max"]),
            "n_empty_clusters": int((counts == 0).sum()),
        },
        "aux_diagnostics_on_cold": aux_diag,
        "n_train_clients": fa["n_train_clients"],
        "n_train_windows_phase_b": int(tr["h_g"].shape[0]),
        "n_cold_windows": fl_only_metrics["n_cold_windows"],
        "n_cold_apts": fl_only_metrics["n_cold_apts"],
        "elapsed_seconds": elapsed,
        "fl_elapsed_seconds": fl_elapsed,
        "gpu_at_start": gpu_start,
        "gpu_at_end": gpu_end,
        "comment": (
            "Priority 1A: FedAvg of NBEATSxAux with the same combined loss "
            "(MAE + 0.3·peak_aux) the centralised v01-v04 method uses. fl_only "
            "is the federated backbone's raw cold metric (no Peak-VQ). The two "
            "with_codebook_* blocks layer Peak-VQ + W5 hybrid at the v01 "
            "operating points UNCHANGED; (â, ĥ) come from the federated aux head "
            "(not self-derived). Comparing fl_only vs with_codebook_* isolates "
            "Peak-VQ's contribution under FL training; comparing fl_only vs the "
            "parent folder's fedavg cell isolates the aux-head training factor."
        ),
    }

    with open(out_dir / "result.json", "w") as fh:
        json.dump(out, fh, indent=2)
    torch.save(fa["final_state_dict"], out_dir / "final_state_dict.pt")
    # Codebook bundle (kept alongside in case downstream wants to recompute).
    np.savez(
        out_dir / "codebook.npz",
        codebook=centroids.astype(np.float32),
        counts=counts.astype(np.int64),
        offsets=offsets,
        cluster_idx=cluster_idx.astype(np.int32),
        key_pool=key_pool,
        key_pool_scaled=key_pool_scaled,
        key_scaler_mean=key_scaler.mean_.astype(np.float32),
        key_scaler_scale=key_scaler.scale_.astype(np.float32),
        n_windows_per_apt=tr["n_windows_per_apt"],
    )
    print(f"[v04 1A] saved -> {out_dir}")
    print(f"[v04 1A] total elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()


# Expected output (seed=42, GTX 5070 Ti):
#   - 20 FedAvg rounds × 80 clients × 2 local epochs ≈ 12 min,
#     Phase B + Phase C ≈ 1-2 min, total ≈ 14-15 min.
#   - fl_only PAPE in the 50-60 kW range (similar to parent fedavg cell).
#   - with_codebook_PAPE_aggressive: expected to be **comparable to** v02
#     centralised T2 (35.7 PAPE) — if it is, the headline gap is
#     attributable to "with vs without aux head" rather than "centralised
#     vs federated"; if it is significantly worse, the gap really is the
#     centralised-pretraining factor.
