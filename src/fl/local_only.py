"""Local-only NBEATSx baseline (no FL) — v04 G1 lower bound.

For each held-out cold apt, train a fresh NBEATSx **on that apt's own
first-70% segment** with vanilla MAE-loss SGD; evaluate on the same
apt's eval segment. This is *not* a federated algorithm — it is the
"no-FL lower bound" referenced in the v04 plan §FL baselines (Tier 1):

    "what cold-PAPE can a cold apt achieve using only its own data?"

A FL or centralised method that beats Local-only justifies the cost
of FL.

Reference (loop sketch): the official ``main_local.py`` from
``rahulv0205/fedrep_experiments`` (cached in
``papers/literlature/fedrep_official/main_local.py``) — per-client
self-train with no aggregation. We re-implement directly because the
official is image-task code; only the training-loop shape is borrowed.

Hyperparameters
---------------
``rounds`` is reinterpreted as **total epochs of self-train per cold
apt** (FL rounds make no sense without aggregation). Default 20 epochs
matches the FedAvg total local-update budget (rounds=20 × E=2 → here
just 20 epochs total) so v04 reports comparable per-client compute.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import HORIZON, INPUT_SIZE, TRAIN_RATIO, VAL_RATIO
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from fl.base import (
    DEVICE,
    FLConfig,
    FLHistory,
    init_backbone,
    run_local_epochs,
)
from utils.metrics import compute_hr, compute_mae, compute_pape


@dataclass
class LocalOnlyConfig(FLConfig):
    """Local-only hyperparameters: ``rounds`` is reinterpreted as total epochs.

    The other FLConfig fields (``local_epochs``, ``clients_per_round``)
    are unused for Local-only; we keep them so the launcher script can
    pass a single dataclass uniformly across all FL algorithms.
    """

    pass  # uses FLConfig defaults; rounds = total epochs per cold apt


def _train_one_cold_apt(apt: str, cfg: LocalOnlyConfig) -> tuple[torch.nn.Module, dict] | None:
    """Self-train a fresh NBEATSx on this cold apt; return (model, train_log)."""
    try:
        series = load_apartment_hourly(apt).values.astype(np.float32)
    except FileNotFoundError:
        return None
    n = len(series)
    train_end = int(n * TRAIN_RATIO)
    seg_train = series[:train_end]
    m_ = float(seg_train.mean())
    s_ = float(seg_train.std()) if seg_train.std() > 1e-8 else 1.0
    train_set = HouseholdDataset(seg_train, m_, s_, stride=1)
    if len(train_set) == 0:
        return None
    loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)

    # Each cold apt gets a fresh NBEATSx initialised with the same seed
    # as every other apt (so cross-apt comparisons are fair).
    model = init_backbone(seed=cfg.seed)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    diag = run_local_epochs(model, loader, optimizer, n_epochs=cfg.rounds)
    return model, {
        "apt": apt,
        "n_train_windows": len(train_set),
        "n_batches": diag["n_batches"],
        "train_loss": diag["main_loss_mean"],
        "mean": m_,
        "std": s_,
    }


def _eval_one_cold_apt(model: torch.nn.Module, apt: str, mean: float, std: float) -> dict:
    """Inference on this apt's own eval segment (warm-start z-norm = its own train)."""
    series = load_apartment_hourly(apt).values.astype(np.float32)
    n = len(series)
    train_end = int(n * TRAIN_RATIO)
    seg = series[:train_end]
    ds = HouseholdDataset(seg, mean, std, stride=HORIZON)
    if len(ds) == 0:
        return {"pape": float("nan"), "hr@1": float("nan"), "hr@2": float("nan"),
                "mae": float("nan"), "n_windows": 0}
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    model.eval()
    true_z, pred_z = [], []
    with torch.no_grad():
        for x, y in loader:
            y_hat, _ = model(x.to(DEVICE))
            true_z.append(y.numpy())
            pred_z.append(y_hat.cpu().numpy())
    t_z = np.concatenate(true_z, axis=0)
    p_z = np.concatenate(pred_z, axis=0)
    t_kw = t_z * std + mean
    p_kw = p_z * std + mean
    return {
        "pape": float(compute_pape(t_kw, p_kw)),
        "hr@1": float(compute_hr(t_kw, p_kw, tol=1)),
        "hr@2": float(compute_hr(t_kw, p_kw, tol=2)),
        "mae": float(compute_mae(t_kw, p_kw)),
        "n_windows": int(t_z.shape[0]),
    }


def train_local_only(
    train_apts: list[str],         # unused — kept for signature parity with FL algorithms
    cold_apts: list[str],
    cfg: LocalOnlyConfig,
) -> dict:
    """Train one fresh NBEATSx per cold apt; aggregate cold metrics across apts.

    Output schema mirrors the FL algorithms' ``cold_metrics`` so v04's
    aggregator can read all baselines uniformly. We additionally
    report per-apt metrics for debugging.
    """
    per_apt_metrics: list[dict] = []
    per_apt_logs: list[dict] = []
    pooled_pape, pooled_hr1, pooled_hr2, pooled_mae = [], [], [], []
    pooled_n = 0

    for apt in cold_apts:
        out = _train_one_cold_apt(apt, cfg)
        if out is None:
            continue
        model, log = out
        m = _eval_one_cold_apt(model, apt, log["mean"], log["std"])
        m["apt"] = apt
        per_apt_metrics.append(m)
        per_apt_logs.append(log)
        pooled_pape.append((m["pape"], m["n_windows"]))
        pooled_hr1.append((m["hr@1"], m["n_windows"]))
        pooled_hr2.append((m["hr@2"], m["n_windows"]))
        pooled_mae.append((m["mae"], m["n_windows"]))
        pooled_n += int(m["n_windows"])

    def _wmean(pairs: list[tuple[float, int]]) -> float:
        if not pairs or sum(n for _, n in pairs) == 0:
            return float("nan")
        return float(sum(v * n for v, n in pairs) / sum(n for _, n in pairs))

    cold_metrics = {
        "pape": _wmean(pooled_pape),
        "hr@1": _wmean(pooled_hr1),
        "hr@2": _wmean(pooled_hr2),
        "mae": _wmean(pooled_mae),
        "n_cold_windows": pooled_n,
        "n_cold_apts": len(per_apt_metrics),
    }

    history = FLHistory()
    # No real "rounds" in Local-only; we record one entry per cold apt for parity.
    for log in per_apt_logs:
        history.append(
            round_idx=0,
            train_loss=log["train_loss"],
            n_clients=1,
            extra={"apt": log["apt"], "n_train_windows": log["n_train_windows"]},
        )

    return {
        "algorithm": "local_only",
        "config": cfg.__dict__,
        "history": history.as_dict(),
        "cold_metrics": cold_metrics,
        "per_apt_metrics": per_apt_metrics,
        "n_train_clients": 0,    # by definition no FL training clients
        "final_state_dict": None,  # one model per apt; not aggregated
    }
