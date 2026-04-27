"""Federated-learning simulator scaffolding shared across v04 FL baselines.

All v04 FL algorithms (FedAvg / FedProx / FedRep / Ditto / Local-only) share
the same simulation primitives:

- **Client = one apt.** Each train apt holds its own ``HouseholdDataset``
  built on its own first-70% segment with per-apt z-norm.
- **Backbone = MinimalNBEATSx** (no peak_aux head). v04's FL axis isolates
  "what is federated" so we strip the auxiliary head; the v01-v03 method
  with peak_aux + Peak-VQ is its own row in the comparison table.
- **Round** = each selected client runs ``E`` local epochs of MAE-loss SGD
  on its own dataset, then the server aggregates the resulting weights.
- **Held-out cold apts.** 20 cold apts are never clients (decision 1 in
  the v04 plan). They are forwarded after FL training for cold inference.

The helpers here are deliberately small and *algorithm-agnostic*. Per-
algorithm differences (proximal term, head split, dual-model
regularisation, …) live in the per-algorithm files in this directory.

Public surface
--------------
- ``FLConfig``                  — common hyperparameters (rounds, local_epochs, lr, batch).
- ``ClientData``                — one apt's train/val Datasets + denorm stats.
- ``build_clients(apts, ...)``  — load every apt's series and wrap it.
- ``client_loader(client, batch, shuffle=True)`` — convenience DataLoader.
- ``init_backbone(seed)``       — seeded MinimalNBEATSx initialiser.
- ``flatten_state_dict(sd)`` / ``apply_state_dict(model, sd)``  — convenience
  for sending parameters around (deep-clone helpers).
- ``weighted_average(state_dicts, weights)`` — FedAvg-style aggregation.
- ``local_step_mae(model, batch, optimizer)``  — one MAE training step;
  per-algorithm files wrap this with extra terms (proximal / l2-toward-global).
- ``evaluate_cold(model, cold_apts, ...)``  — cold-side inference under the
  v02 protocol (warm-start z-norm, stride=24, denormalise to kW), returning
  PAPE / HR@1 / HR@2 / MAE.

Output JSON contract is the same across all FL baselines so v04's
``07_aggregate.py`` can read them uniformly.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import HORIZON, INPUT_SIZE, TRAIN_RATIO, VAL_RATIO
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from models.nbeatsx import MinimalNBEATSx
from utils.metrics import compute_hr, compute_mae, compute_pape

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class FLConfig:
    """Common FL hyperparameters; per-algorithm files extend with extra fields.

    Defaults follow the FedAvg literature norm cited in v04 plan §Open-B
    (rounds=20, local_epochs=2, batch=256, lr=1e-3) — adjust during a
    convergence dry-run if needed.
    """

    rounds: int = 20
    local_epochs: int = 2
    lr: float = 1e-3
    batch_size: int = 256
    weight_decay: float = 1e-5
    seed: int = 42

    # Aggregation: number of clients sampled per round. 0 means "all" (every
    # train apt participates in every round). UMass has only 80 train apts so
    # full participation is cheap.
    clients_per_round: int = 0


# ============================================================================
# Client data
# ============================================================================


@dataclass
class ClientData:
    """One apt's local datasets + per-apt z-norm statistics."""

    apt: str
    train_set: HouseholdDataset
    val_set: HouseholdDataset
    mean: float
    std: float
    n_train_windows: int


def build_clients(apts: Iterable[str]) -> list[ClientData]:
    """Build one ``ClientData`` per train apt under the v01/v02/v03 protocol.

    - per-apt z-norm uses each apt's own first ``TRAIN_RATIO`` segment,
    - sliding windows with stride=1 (same as v01/v02 02_train_arms.py).

    Apartments whose CSV is missing are silently skipped.
    """
    clients: list[ClientData] = []
    for apt in apts:
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            continue
        n = len(series)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
        seg_train = series[:train_end]
        seg_val = series[train_end:val_end]
        m_ = float(seg_train.mean())
        s_ = float(seg_train.std()) if seg_train.std() > 1e-8 else 1.0
        train_set = HouseholdDataset(seg_train, m_, s_, stride=1)
        val_set = HouseholdDataset(seg_val, m_, s_, stride=1)
        clients.append(
            ClientData(
                apt=apt,
                train_set=train_set,
                val_set=val_set,
                mean=m_,
                std=s_,
                n_train_windows=len(train_set),
            )
        )
    return clients


def client_loader(client: ClientData, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(client.train_set, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ============================================================================
# Backbone init / state-dict utilities
# ============================================================================


def init_backbone(seed: int) -> MinimalNBEATSx:
    """Seeded init of the FL backbone (MinimalNBEATSx, no aux head)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    return MinimalNBEATSx().to(DEVICE)


def clone_state_dict(state_dict: dict[str, torch.Tensor]) -> OrderedDict:
    """Deep clone with ``.detach().cpu().clone()`` per tensor."""
    return OrderedDict(
        (k, v.detach().cpu().clone()) for k, v in state_dict.items()
    )


def apply_state_dict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load a state dict onto the model (moves tensors to the model's device)."""
    device = next(model.parameters()).device
    model.load_state_dict({k: v.to(device) for k, v in state_dict.items()}, strict=True)


def weighted_average(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: list[float],
) -> OrderedDict:
    """FedAvg-style weighted average of a list of state dicts.

    Args
    ----
    state_dicts : per-client state dicts (CPU tensors recommended).
    weights     : per-client weights (e.g. n_train_windows). Will be
                  normalised to sum to 1 inside the function.

    Returns the averaged state dict (CPU tensors). Tensors that are not
    floating-point (e.g. ``Long`` buffers, ``BoolTensor`` masks) are taken
    from the first state dict verbatim — averaging int tensors makes no
    sense for FL.
    """
    if len(state_dicts) == 0:
        raise ValueError("weighted_average: empty state_dicts")
    if len(state_dicts) != len(weights):
        raise ValueError(
            f"weighted_average: {len(state_dicts)} state_dicts vs {len(weights)} weights"
        )
    total = float(sum(weights))
    if total <= 0:
        raise ValueError(f"weighted_average: weights sum to {total}; expected > 0")
    norm = [w / total for w in weights]
    out: OrderedDict = OrderedDict()
    for k, v0 in state_dicts[0].items():
        if not v0.is_floating_point():
            # Non-float tensors (counts, indices, buffers) — keep first.
            out[k] = v0.detach().cpu().clone()
            continue
        acc = torch.zeros_like(v0, dtype=torch.float64).cpu()
        for sd, w in zip(state_dicts, norm):
            acc = acc + sd[k].detach().cpu().to(torch.float64) * w
        out[k] = acc.to(v0.dtype)
    return out


# ============================================================================
# Local training step
# ============================================================================


def local_step_mae(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """One MAE-loss training step on a single batch. Returns the loss value.

    The forward call assumes ``MinimalNBEATSx`` (returns ``(y_hat, hiddens)``)
    so we keep the FL backbone consistent across all algorithms in this
    directory.
    """
    model.train()
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    y_hat, _ = model(x)
    loss = F.l1_loss(y_hat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def run_local_epochs(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    extra_loss_fn: callable | None = None,
) -> dict:
    """Run ``n_epochs`` of local SGD on this client's loader.

    ``extra_loss_fn(model, x, y, y_hat) -> torch.Tensor`` lets per-algorithm
    files add a regulariser to the loss (e.g. FedProx proximal term, Ditto's
    pull-toward-global, …) without re-implementing the loop.

    Returns a small diagnostic dict.
    """
    model.train()
    n_batches = 0
    sum_main = 0.0
    sum_extra = 0.0
    for _ in range(n_epochs):
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y_hat, _ = model(x)
            main = F.l1_loss(y_hat, y)
            loss = main
            if extra_loss_fn is not None:
                extra = extra_loss_fn(model, x, y, y_hat)
                loss = loss + extra
                sum_extra += float(extra.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_main += float(main.item())
            n_batches += 1
    return {
        "n_batches": n_batches,
        "main_loss_mean": sum_main / max(n_batches, 1),
        "extra_loss_mean": sum_extra / max(n_batches, 1) if extra_loss_fn else None,
    }


# ============================================================================
# Cold-side evaluation (mirrors v02 04 cold protocol)
# ============================================================================


class _ColdWindowDataset(Dataset):
    """Cold apt sliding windows with denorm stats kept alongside."""

    def __init__(self, series: np.ndarray, mean: float, std: float, stride: int = HORIZON) -> None:
        self.ds = HouseholdDataset(series, mean, std, stride=stride)
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        return self.ds[idx]


def evaluate_cold(
    model: torch.nn.Module,
    cold_apts: Iterable[str],
    batch_size: int = 256,
    stride: int = HORIZON,
) -> dict:
    """Cold-side inference under the v02 protocol; returns PAPE / HR@1 / HR@2 / MAE in kW.

    For each cold apt:
      1. warm-start z-norm on its own first ``TRAIN_RATIO`` segment,
      2. sliding windows of stride ``stride`` (default = horizon = 24),
      3. forward (eval mode, no_grad) -> ``y_hat_z``,
      4. denormalise to kW with that apt's (mean, std) and accumulate.

    Returns ``{"pape", "hr@1", "hr@2", "mae", "n_cold_windows", "n_cold_apts"}``.
    """
    model.eval()
    true_chunks, pred_chunks, mean_chunks, std_chunks = [], [], [], []
    n_apts_seen = 0
    for apt in cold_apts:
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            continue
        n = len(series)
        train_end = int(n * TRAIN_RATIO)
        seg = series[:train_end]
        m_ = float(seg.mean())
        s_ = float(seg.std()) if seg.std() > 1e-8 else 1.0
        ds = _ColdWindowDataset(seg, m_, s_, stride=stride)
        if len(ds) == 0:
            continue
        n_apts_seen += 1
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        for x, y in loader:
            with torch.no_grad():
                y_hat, _ = model(x.to(DEVICE))
            true_chunks.append(y.numpy())
            pred_chunks.append(y_hat.cpu().numpy())
            mean_chunks.append(np.full(len(y), m_, dtype=np.float32))
            std_chunks.append(np.full(len(y), s_, dtype=np.float32))
    if not true_chunks:
        return {"pape": float("nan"), "hr@1": float("nan"), "hr@2": float("nan"),
                "mae": float("nan"), "n_cold_windows": 0, "n_cold_apts": 0}
    t_z = np.concatenate(true_chunks, axis=0)
    p_z = np.concatenate(pred_chunks, axis=0)
    m_arr = np.concatenate(mean_chunks, axis=0)
    s_arr = np.concatenate(std_chunks, axis=0)
    t_kw = t_z * s_arr[:, None] + m_arr[:, None]
    p_kw = p_z * s_arr[:, None] + m_arr[:, None]
    return {
        "pape": float(compute_pape(t_kw, p_kw)),
        "hr@1": float(compute_hr(t_kw, p_kw, tol=1)),
        "hr@2": float(compute_hr(t_kw, p_kw, tol=2)),
        "mae": float(compute_mae(t_kw, p_kw)),
        "n_cold_windows": int(t_z.shape[0]),
        "n_cold_apts": int(n_apts_seen),
    }


# ============================================================================
# History container
# ============================================================================


@dataclass
class FLHistory:
    """Per-round diagnostic capture for any FL algorithm in this directory."""

    rounds: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    n_clients: list[int] = field(default_factory=list)
    extra: list[dict] = field(default_factory=list)

    def append(self, *, round_idx: int, train_loss: float, n_clients: int, extra: dict | None = None) -> None:
        self.rounds.append(round_idx)
        self.train_loss.append(train_loss)
        self.n_clients.append(n_clients)
        self.extra.append(extra or {})

    def as_dict(self) -> dict:
        return {
            "rounds": self.rounds,
            "train_loss": self.train_loss,
            "n_clients": self.n_clients,
            "extra": self.extra,
        }
