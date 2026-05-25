"""V9-RoundVQ-FedAvg-naive — quick sanity check for round-wise VQ training.

(한글 요약)
plans/v09-01_round_wise_codebook.md 의 *round-wise federated codebook* 학습을
본격 구현하기 전, "FL round 안에서 VQ가 실제로 학습되는가" 만 확인하는 가장
단순한 single-driver smoke test.

본 실험은 plan v09-01 의 단순화 버전이다:
  - Backbone = ``NBEATSxVQ`` (src/models/nbeatsx_vq.py) — peak-aux head 없음.
    loss = MAE(ŷ, y) + L_commit 만 사용 (λ_aux 미적용).
  - Aggregation = naive FedAvg. ``VectorQuantizerEMA`` 의 buffer
    (``codebook``, ``ema_count``, ``ema_weight``) 가 state_dict 통째로
    weighted-average 되며, cluster identity alignment / mass-weighted
    aggregation / EMA blending / dead-code respawn 없음.
  - Per-round broadcast: server state_dict 통째 (`apply_state_dict` strict=True)
    → client EMA 누적이 매 라운드 server 버전으로 reset.

Sanity check 통과 조건 (no formal gate, 사용자가 round_log.jsonl 보고 판단):
  - 학습 중 NaN 없음.
  - per-round VQ utilization > 0, perplexity > 1 (codebook collapse 아님).
  - val.PAPE 가 라운드를 거치며 감소 trend.

통과 시 → plan v09-01 본 구현 (NBEATSxAux + cluster-mass weighted aggregation
+ EMA blending + dead-code respawn) 로 확장.

Per-seed argparse — single seed × single cell per invocation
(memory: feedback_argparse_per_seed).
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

from config import OUTPUT_DIR, RANDOM_SEED
from dataloader.per_client_split import build_per_client_splits
from fl.base import DEVICE, apply_state_dict, clone_state_dict, weighted_average
from models.nbeatsx_vq import NBEATSxVQ
from utils.metrics import compute_hr, compute_mae, compute_mse, compute_pape


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def _wrap_amp(use_amp: bool):
    if use_amp and DEVICE.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return _NullCtx()


def _init_model(seed: int, num_embeddings: int, commitment_beta: float) -> NBEATSxVQ:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = NBEATSxVQ(
        num_embeddings=num_embeddings,
        commitment_beta=commitment_beta,
    ).to(DEVICE)
    return model


def _local_train_one_client(
    model: NBEATSxVQ,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    n_epochs: int,
    use_amp: bool,
) -> dict:
    """One client × n_epochs of (MAE + L_commit) SGD, EMA codebook update inside VQ."""
    model.train()
    n_batches, sum_main, sum_commit = 0, 0.0, 0.0
    sum_util, sum_ppl = 0.0, 0.0
    for _ in range(n_epochs):
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _wrap_amp(use_amp):
                y_hat, vq_state = model(x)
                main = F.l1_loss(y_hat, y)
                commit = vq_state["commit_loss"]
                loss = main + commit
            loss.backward()
            optimizer.step()
            sum_main += float(main.item())
            sum_commit += float(commit.item())
            sum_util += float(vq_state["utilization"])
            sum_ppl += float(vq_state["perplexity"])
            n_batches += 1
    return {
        "n_batches": n_batches,
        "main_loss_mean":   sum_main   / max(n_batches, 1),
        "commit_loss_mean": sum_commit / max(n_batches, 1),
        "vq_util_mean":     sum_util   / max(n_batches, 1),
        "vq_ppl_mean":      sum_ppl    / max(n_batches, 1),
    }


@torch.no_grad()
def _eval_per_client(
    model: NBEATSxVQ,
    splits: dict[str, dict],
    split_key: str,
    *,
    batch_size: int,
    use_amp: bool,
) -> dict[str, float]:
    """Across-client mean of per-apt PAPE/HR/MAE/MSE on `split_key` ∈ {val, test}.

    Metrics are computed in kW (denormalised via per-apt train mean/std), matching
    the v06 RoundLogger convention.
    """
    model.eval()
    papes, maes, mses, hr1s, hr2s = [], [], [], [], []
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
                y_hat, vq_state = model(xb)
            yhat_chunks.append(y_hat.float().cpu().numpy())
            util_sum += float(vq_state["utilization"])
            ppl_sum  += float(vq_state["perplexity"])
            n_chunks += 1
        y_hat_z   = np.concatenate(yhat_chunks, axis=0).astype(np.float32)
        y_true_kw = (y * s_ + m_).astype(np.float32)
        y_hat_kw  = (y_hat_z * s_ + m_).astype(np.float32)
        papes.append(float(compute_pape(y_true_kw, y_hat_kw)))
        maes.append (float(compute_mae (y_true_kw, y_hat_kw)))
        mses.append (float(compute_mse (y_true_kw, y_hat_kw)))
        hr1s.append (float(compute_hr  (y_true_kw, y_hat_kw, tol=1)))
        hr2s.append (float(compute_hr  (y_true_kw, y_hat_kw, tol=2)))
    return {
        "pape_mean":               float(np.mean(papes)) if papes else float("nan"),
        "pape_std_across_clients": float(np.std(papes, ddof=1)) if len(papes) > 1 else 0.0,
        "mae_mean":                float(np.mean(maes)) if maes else float("nan"),
        "mse_kw2_mean":            float(np.mean(mses)) if mses else float("nan"),
        "hr@1_mean":               float(np.mean(hr1s)) if hr1s else float("nan"),
        "hr@2_mean":               float(np.mean(hr2s)) if hr2s else float("nan"),
        "vq_util_mean":            util_sum / max(n_chunks, 1),
        "vq_ppl_mean":             ppl_sum  / max(n_chunks, 1),
        "n_clients":               int(len(papes)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "v09 sanity check — NBEATSxVQ trained inside naive FedAvg rounds. "
            "MAE + commit_loss only (no peak-aux). Single seed × single cell per invocation."
        )
    )
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--rounds", type=int, default=10,
                    help="Smoke test default = 10 rounds. v09 plan target = 150.")
    ap.add_argument("--local_epochs", type=int, default=5,
                    help="v08-aligned default (E=5).")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--num_embeddings", type=int, default=32,
                    help="VQ codebook size M (v06 invariant).")
    ap.add_argument("--commitment_beta", type=float, default=0.25,
                    help="VQ-VAE commitment loss weight (van den Oord 2017).")
    ap.add_argument("--no_amp", action="store_true",
                    help="Disable bf16 autocast (auto-disabled on CPU).")
    ap.add_argument("--cell", type=str, default="V9-RoundVQ-FedAvg-naive")
    ap.add_argument("--output_namespace", type=str, default="v09_round_vq_codebook")
    args = ap.parse_args()

    use_amp = not args.no_amp
    out_dir = OUTPUT_DIR / args.output_namespace / f"seed{args.seed}" / args.cell
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "round_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    print(f"[{args.cell}] seed={args.seed}  R={args.rounds}  E={args.local_epochs}  "
          f"batch={args.batch_size}  M={args.num_embeddings}  beta={args.commitment_beta}  "
          f"amp={use_amp}")
    print(f"[{args.cell}] out_dir={out_dir}")

    # 1) Per-client splits (same disk cache as v06/v08).
    print(f"[{args.cell}] building per-client 70/10/20 splits ...")
    splits = build_per_client_splits(seed=args.seed)
    n_clients = len(splits)
    print(f"[{args.cell}] {n_clients} apartments retained.")

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
    server_model = _init_model(args.seed, args.num_embeddings, args.commitment_beta)
    global_state = clone_state_dict(server_model.state_dict())

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
        round_main_sum, round_commit_sum = 0.0, 0.0
        round_util_sum, round_ppl_sum, round_batches = 0.0, 0.0, 0

        for apt, loader in client_loaders.items():
            apply_state_dict(server_model, global_state)
            optimizer = torch.optim.Adam(
                server_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
            diag = _local_train_one_client(
                server_model, loader, optimizer,
                n_epochs=args.local_epochs, use_amp=use_amp,
            )
            local_states.append(clone_state_dict(server_model.state_dict()))
            local_weights.append(client_weights[apt])
            round_main_sum   += diag["main_loss_mean"]   * diag["n_batches"]
            round_commit_sum += diag["commit_loss_mean"] * diag["n_batches"]
            round_util_sum   += diag["vq_util_mean"]     * diag["n_batches"]
            round_ppl_sum    += diag["vq_ppl_mean"]      * diag["n_batches"]
            round_batches    += diag["n_batches"]

        # Naive FedAvg: VQ buffers (codebook / ema_count / ema_weight) are also
        # float and therefore averaged just like any parameter. plan v09-01 §3
        # replaces this with cluster-mass weighted aggregation + EMA blending.
        global_state = weighted_average(local_states, local_weights)
        apply_state_dict(server_model, global_state)

        # Snapshot post-aggregation server VQ buffers. Sizes are tiny (M=32
        # × D=64) so the full per-round history fits in memory; persisted
        # as a single codebook_history.pt at the end of training.
        cb_now  = global_state["stack_generic.vq.codebook"].detach().cpu().clone()
        ema_cnt = global_state["stack_generic.vq.ema_count"].detach().cpu().clone()
        ema_w   = global_state["stack_generic.vq.ema_weight"].detach().cpu().clone()
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
        }
        prev_cb = cb_now

        val_metrics = _eval_per_client(
            server_model, splits, "val",
            batch_size=args.batch_size, use_amp=use_amp,
        )
        wall = time.time() - t_round
        row = {
            "round": r,
            "wall_seconds": float(wall),
            "train": {
                "main_loss_mean":   round_main_sum   / max(round_batches, 1),
                "commit_loss_mean": round_commit_sum / max(round_batches, 1),
                "vq_util_mean":     round_util_sum   / max(round_batches, 1),
                "vq_ppl_mean":      round_ppl_sum    / max(round_batches, 1),
                "n_batches":        int(round_batches),
            },
            "server_vq": server_vq,
            "val": val_metrics,
        }
        history.append(row)
        with log_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(
            f"  round {r:2d}: train.main={row['train']['main_loss_mean']:.4f}  "
            f"commit={row['train']['commit_loss_mean']:.4f}  "
            f"util={row['train']['vq_util_mean']:.2f}  "
            f"ppl={row['train']['vq_ppl_mean']:.2f}  "
            f"cb_drift={cb_drift:.4f}  "
            f"val.PAPE={val_metrics['pape_mean']:.2f}  wall={wall:.1f}s"
        )

    test_metrics = _eval_per_client(
        server_model, splits, "test",
        batch_size=args.batch_size, use_amp=use_amp,
    )
    elapsed = time.time() - t0

    torch.save(global_state, out_dir / "final_state_dict.pt")
    if cb_history["rounds"]:
        torch.save({
            "rounds":     cb_history["rounds"],
            "codebook":   torch.stack(cb_history["codebook"]),    # (R, M, D)
            "ema_count":  torch.stack(cb_history["ema_count"]),   # (R, M)
            "ema_weight": torch.stack(cb_history["ema_weight"]),  # (R, M, D)
        }, out_dir / "codebook_history.pt")
    result = {
        "cell": args.cell,
        "seed": int(args.seed),
        "n_clients": n_clients,
        "rounds": int(args.rounds),
        "local_epochs": int(args.local_epochs),
        "batch": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "num_embeddings": int(args.num_embeddings),
        "commitment_beta": float(args.commitment_beta),
        "use_amp": bool(use_amp),
        "history": history,
        "val_terminal":  history[-1]["val"] if history else None,
        "test_terminal": test_metrics,
        "elapsed_seconds": float(elapsed),
        "comment": (
            "v09 quick sanity check. NBEATSx + VQ on h_generic; naive FedAvg "
            "(codebook + EMA buffers averaged); no peak-aux head; no respawn / "
            "no mass-weighted aggregation. plans/v09-01_round_wise_codebook.md "
            "full FedVQ pending."
        ),
    }
    with (out_dir / "result.json").open("w") as fh:
        json.dump(result, fh, indent=2)
    print(
        f"[{args.cell}] done. test.PAPE={test_metrics['pape_mean']:.2f}  "
        f"util={test_metrics['vq_util_mean']:.2f}  "
        f"ppl={test_metrics['vq_ppl_mean']:.2f}  elapsed={elapsed:.0f}s"
    )


if __name__ == "__main__":
    main()
