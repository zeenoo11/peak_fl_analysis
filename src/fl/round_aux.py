"""Unified NBEATSxAux federated round-loop for v06 round-dynamics experiments.

Why a new module
----------------
v06 (`plans/v06-01_round_dynamics.md` §"백본 / hyperparameter") requires
all 5 FL cells (FedAvg, FedProx, FedRep, Ditto, FedProto) to share the
same backbone — **NBEATSxAux with combined loss
``L = MAE(ŷ, y) + 0.3 · peak_aux(ŷ, y; hr_weight=0.1)``** — so the
round-by-round comparison is apples-to-apples on the *same* backbone the
conference Proposed cell is built on (``src/fl/fedavg_aux.py``).

The existing per-algorithm files in ``src/fl/`` (``fedprox.py``,
``fedrep.py``, ``ditto.py``, ``fedproto.py``) are built on
``MinimalNBEATSx`` + MAE-only loss and are kept untouched here for v04 /
v05 reproducibility (their reported numbers were produced under that
recipe). v06's 5-algo driver imports *this* module instead.

What this module provides
-------------------------
- ``run_fl_aux(algorithm, ...)`` — single entry point dispatching by name
  to one of {fedavg, fedprox, fedrep, ditto, fedproto} round loops, all
  on NBEATSxAux + combined loss.
- Each round-loop accepts an ``on_round_end(round_idx, server_state_post,
  client_states_pre, comm_stats, wall_seconds)`` callback that v06's
  ``RoundLogger`` plugs in. The callback is called *after* server
  aggregation but *before* the next-round broadcast.
- ``comm_stats`` includes both ``upload_bytes_round`` and
  ``broadcast_bytes_round``. Default magnitude = ``|θ| × 4 × n_clients``
  (upload) and ``|θ| × 4`` (broadcast), with per-algorithm refinements
  noted inline (e.g. FedRep broadcasts encoder only, FedProto adds the
  prototype payload).

Design choices
--------------
- Each helper builds ``ClientData`` only once (via ``build_clients`` with
  the apts list, falling back to a v06-supplied per-client split that
  bypasses ``build_clients`` entirely — see ``train_apt_data`` arg).
- The "client states pre-aggregation" passed to ``on_round_end`` is the
  list of end-of-local state dicts from this round (drift = ||θ_i -
  θ_global^{round-start}||₂, mean over i).
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

from config import D_MODEL
from fl.base import (
    DEVICE,
    ClientData,
    _NullCtx,
    apply_state_dict,
    clone_state_dict,
    weighted_average,
)
from fl.fedavg_aux import _local_step_aux, init_backbone_aux
from models.nbeatsx_aux import NBEATSxAux
from models.peak_aux_head import peak_aux_loss


# ---------------------------------------------------------------------------
# Helpers — bytes / loaders / clients
# ---------------------------------------------------------------------------


def _state_dict_bytes(sd: dict[str, torch.Tensor]) -> int:
    """Sum of tensor sizes in bytes (respects per-dtype element size)."""
    return sum(int(v.numel()) * int(v.element_size()) for v in sd.values())


def _build_clients_from_v06_splits(
    splits: dict[str, dict],
) -> list[ClientData]:
    """Construct ``ClientData`` objects directly from v06 per-client split blocks
    (avoids ``fl.base.build_clients`` re-loading from disk).
    """
    clients: list[ClientData] = []
    for apt, sp in splits.items():
        # Wrap the precomputed (x, y) tensors as a TensorDataset; this gives
        # the same iteration interface as HouseholdDataset.
        train_x = torch.from_numpy(sp["train_x"])
        train_y = torch.from_numpy(sp["train_y"])
        val_x   = torch.from_numpy(sp["val_x"])
        val_y   = torch.from_numpy(sp["val_y"])
        train_set = TensorDataset(train_x, train_y)
        val_set   = TensorDataset(val_x, val_y)
        clients.append(
            ClientData(
                apt=apt,
                train_set=train_set,
                val_set=val_set,
                mean=float(sp["mean"]),
                std=float(sp["std"]),
                n_train_windows=int(len(train_set)),
            )
        )
    return clients


def _client_loader(client: ClientData, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(client.train_set, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _wrap_amp(use_amp: bool):
    use_amp = use_amp and (DEVICE.type == "cuda")
    return (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp else _NullCtx()
    )


# ---------------------------------------------------------------------------
# FedAvg-aux (matches src/fl/fedavg_aux.py — re-implemented here so the
# on_round_end hook fits cleanly without monkey-patching fedavg_aux.py)
# ---------------------------------------------------------------------------


def _fedavg_round_loop_aux(
    clients: list[ClientData],
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
    on_round_end: Optional[Callable] = None,
) -> dict:
    """FedAvg over NBEATSxAux with combined MAE+peak_aux loss + round-end hook."""
    global_model = init_backbone_aux(seed=seed)
    global_state = clone_state_dict(global_model.state_dict())

    history: dict = {"rounds": [], "main_loss": [], "aux_loss": [], "n_clients": []}
    n_clients = len(clients)
    sd_bytes = _state_dict_bytes(global_state)

    for r in range(1, rounds + 1):
        t_round = time.time()
        round_start_state = clone_state_dict(global_state)
        local_states: list[dict] = []
        local_weights: list[float] = []
        round_main_sum, round_aux_sum, round_n = 0.0, 0.0, 0
        for client in clients:
            apply_state_dict(global_model, global_state)
            optimizer = torch.optim.Adam(
                global_model.parameters(), lr=lr, weight_decay=weight_decay
            )
            loader = _client_loader(client, batch_size, shuffle=True)
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
        history["n_clients"].append(n_clients)
        wall = time.time() - t_round
        print(f"  round {r:2d}: main={history['main_loss'][-1]:.4f}  "
              f"aux={history['aux_loss'][-1]:.4f}  n_clients={n_clients}  wall={wall:.1f}s")

        if on_round_end is not None:
            apply_state_dict(global_model, global_state)
            on_round_end(
                round_idx=r,
                model=global_model,
                server_state_pre=round_start_state,
                client_states=local_states,
                comm_stats={
                    "upload_bytes_round": sd_bytes * n_clients,
                    "broadcast_bytes_round": sd_bytes,
                },
                wall_seconds=wall,
                train_stats={
                    "loss_mean_last_epoch": history["main_loss"][-1],
                    "n_steps_round": int(round_n),
                },
            )

    return {
        "history": history,
        "final_state_dict": global_state,
        "n_train_clients": n_clients,
    }


# ---------------------------------------------------------------------------
# FedProx-aux
# ---------------------------------------------------------------------------


def _local_step_aux_with_quadratic_anchor(
    model: NBEATSxAux,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    use_amp: bool,
    aux_lambda: float,
    hr_weight: float,
    snapshot: dict[str, torch.Tensor],
    coef: float,
    anchor_key: str,
) -> dict:
    """``L = MAE + λ·peak_aux + (coef/2)·||θ - snapshot||²``.

    Shared body for FedProx (snapshot = round-start global, coef = μ) and
    Ditto's personal-model step (snapshot = round-start global, coef = lam).
    ``anchor_key`` names the third loss-mean slot in the returned dict
    (``"prox_loss_mean"`` vs ``"pull_loss_mean"``).
    """
    amp_ctx = _wrap_amp(use_amp)
    model.train()
    n_batches = 0
    sum_main, sum_aux, sum_anchor = 0.0, 0.0, 0.0
    snap = {k: v.detach().clone() for k, v in snapshot.items()}
    for _ in range(n_epochs):
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                y_hat, _h, (amp_pred, hr_pred) = model(x)
                main = F.l1_loss(y_hat, y)
                aux = peak_aux_loss(amp_pred, hr_pred, y, hr_weight=hr_weight)
                anchor = torch.zeros((), device=DEVICE)
                for n_, p in model.named_parameters():
                    if not p.requires_grad or not p.is_floating_point():
                        continue
                    ref = snap[n_].to(DEVICE)
                    anchor = anchor + ((p - ref) ** 2).sum()
                loss = main + aux_lambda * aux + 0.5 * coef * anchor
            loss.backward()
            optimizer.step()
            sum_main += float(main.item())
            sum_aux += float(aux.item())
            sum_anchor += float(anchor.item())
            n_batches += 1
    return {
        "n_batches": n_batches,
        "main_loss_mean": sum_main / max(n_batches, 1),
        "aux_loss_mean":  sum_aux  / max(n_batches, 1),
        anchor_key:       sum_anchor / max(n_batches, 1),
    }


def _local_step_aux_with_prox(
    model: NBEATSxAux,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    use_amp: bool,
    aux_lambda: float,
    hr_weight: float,
    snapshot: dict[str, torch.Tensor],
    mu: float,
) -> dict:
    """``L = MAE + λ·peak_aux + (μ/2)·||θ - θ_global^{round-start}||²``."""
    return _local_step_aux_with_quadratic_anchor(
        model, loader, optimizer, n_epochs, use_amp,
        aux_lambda, hr_weight, snapshot, coef=mu, anchor_key="prox_loss_mean",
    )


def _fedprox_round_loop_aux(
    clients: list[ClientData],
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
    mu: float = 0.01,
    on_round_end: Optional[Callable] = None,
) -> dict:
    """FedProx (loss-augmentation form) over NBEATSxAux + combined loss."""
    global_model = init_backbone_aux(seed=seed)
    global_state = clone_state_dict(global_model.state_dict())
    history = {"rounds": [], "main_loss": [], "aux_loss": [], "prox_loss": [], "n_clients": []}
    n_clients = len(clients)
    sd_bytes = _state_dict_bytes(global_state)

    for r in range(1, rounds + 1):
        t_round = time.time()
        round_start_state = clone_state_dict(global_state)
        local_states: list[dict] = []
        local_weights: list[float] = []
        round_main_sum, round_aux_sum, round_prox_sum, round_n = 0.0, 0.0, 0.0, 0
        for client in clients:
            apply_state_dict(global_model, global_state)
            optimizer = torch.optim.Adam(
                global_model.parameters(), lr=lr, weight_decay=weight_decay
            )
            loader = _client_loader(client, batch_size, shuffle=True)
            diag = _local_step_aux_with_prox(
                global_model, loader, optimizer,
                n_epochs=local_epochs, use_amp=use_amp,
                aux_lambda=aux_lambda, hr_weight=hr_weight,
                snapshot=round_start_state, mu=mu,
            )
            local_states.append(clone_state_dict(global_model.state_dict()))
            local_weights.append(float(client.n_train_windows))
            round_main_sum += diag["main_loss_mean"] * diag["n_batches"]
            round_aux_sum  += diag["aux_loss_mean"]  * diag["n_batches"]
            round_prox_sum += diag["prox_loss_mean"] * diag["n_batches"]
            round_n += diag["n_batches"]
        global_state = weighted_average(local_states, local_weights)
        history["rounds"].append(r)
        history["main_loss"].append(round_main_sum / max(round_n, 1))
        history["aux_loss"].append(round_aux_sum / max(round_n, 1))
        history["prox_loss"].append(round_prox_sum / max(round_n, 1))
        history["n_clients"].append(n_clients)
        wall = time.time() - t_round
        print(f"  round {r:2d}: main={history['main_loss'][-1]:.4f}  "
              f"aux={history['aux_loss'][-1]:.4f}  prox={history['prox_loss'][-1]:.4f}  wall={wall:.1f}s")

        if on_round_end is not None:
            apply_state_dict(global_model, global_state)
            on_round_end(
                round_idx=r,
                model=global_model,
                server_state_pre=round_start_state,
                client_states=local_states,
                comm_stats={
                    "upload_bytes_round": sd_bytes * n_clients,
                    "broadcast_bytes_round": sd_bytes,
                },
                wall_seconds=wall,
                train_stats={
                    "loss_mean_last_epoch": history["main_loss"][-1],
                    "n_steps_round": int(round_n),
                },
            )

    return {
        "history": history,
        "final_state_dict": global_state,
        "n_train_clients": n_clients,
    }


# ---------------------------------------------------------------------------
# FedRep-aux  (encoder/head split: heads = backbone.stack_*.proj.* AND aux_head.*)
#
# In NBEATSxAux the "head" naturally extends to ``aux_head.*`` (the per-client
# auxiliary peak head), so v06 FedRep-aux personalises both:
#   - backbone.stack_*.proj.{weight, bias}  (forecast head)
#   - aux_head.*                            (peak head)
# Encoder = everything else (the shared NBEATSx representation).
# ---------------------------------------------------------------------------


def _is_head_param_aux(name: str) -> bool:
    return (".proj." in name) or name.startswith("aux_head.")


def _set_requires_grad_aux(model: NBEATSxAux, *, train_head: bool) -> None:
    for n_, p in model.named_parameters():
        if _is_head_param_aux(n_):
            p.requires_grad = train_head
        else:
            p.requires_grad = not train_head


def _split_state_dict_aux(sd: dict) -> tuple[OrderedDict, OrderedDict]:
    enc, head = OrderedDict(), OrderedDict()
    for k, v in sd.items():
        (head if _is_head_param_aux(k) else enc)[k] = v
    return enc, head


def _merge_aux(enc: dict, head: dict) -> OrderedDict:
    out = OrderedDict()
    out.update(enc)
    out.update(head)
    return out


def _local_fedrep_aux(
    model: NBEATSxAux,
    loader: DataLoader,
    *,
    n_epochs_total: int,
    head_epochs: int,
    lr: float,
    weight_decay: float,
    use_amp: bool,
    aux_lambda: float,
    hr_weight: float,
) -> dict:
    """Phase 1 (head_epochs): train head only. Phase 2 (rep_epochs): train encoder only."""
    rep_epochs = max(0, n_epochs_total - head_epochs)
    sum_main, sum_aux, n_batches = 0.0, 0.0, 0

    if head_epochs > 0:
        _set_requires_grad_aux(model, train_head=True)
        opt = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=lr, weight_decay=weight_decay,
        )
        d = _local_step_aux(model, loader, opt, n_epochs=head_epochs, use_amp=use_amp,
                            aux_lambda=aux_lambda, hr_weight=hr_weight)
        sum_main += d["main_loss_mean"] * d["n_batches"]
        sum_aux  += d["aux_loss_mean"]  * d["n_batches"]
        n_batches += d["n_batches"]

    if rep_epochs > 0:
        _set_requires_grad_aux(model, train_head=False)
        opt = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=lr, weight_decay=weight_decay,
        )
        d = _local_step_aux(model, loader, opt, n_epochs=rep_epochs, use_amp=use_amp,
                            aux_lambda=aux_lambda, hr_weight=hr_weight)
        sum_main += d["main_loss_mean"] * d["n_batches"]
        sum_aux  += d["aux_loss_mean"]  * d["n_batches"]
        n_batches += d["n_batches"]

    # Re-enable all gradients for the next caller.
    for p in model.parameters():
        p.requires_grad = True

    return {
        "main_loss_mean": sum_main / max(n_batches, 1),
        "aux_loss_mean":  sum_aux  / max(n_batches, 1),
        "n_batches": n_batches,
    }


def _fedrep_round_loop_aux(
    clients: list[ClientData],
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
    head_epochs: int = 1,
    on_round_end: Optional[Callable] = None,
) -> dict:
    """FedRep over NBEATSxAux. Encoder is federated; per-client heads (forecast
    proj heads + aux head) are kept locally and never broadcast.
    """
    global_model = init_backbone_aux(seed=seed)
    init_sd = clone_state_dict(global_model.state_dict())
    encoder_state, init_head = _split_state_dict_aux(init_sd)
    client_heads: dict[str, OrderedDict] = {
        c.apt: OrderedDict((k, v.clone()) for k, v in init_head.items()) for c in clients
    }
    history = {"rounds": [], "main_loss": [], "aux_loss": [], "n_clients": []}
    n_clients = len(clients)
    encoder_bytes = _state_dict_bytes(encoder_state)
    head_bytes = _state_dict_bytes(init_head)
    full_bytes = encoder_bytes + head_bytes  # used for drift snapshot referencing

    for r in range(1, rounds + 1):
        t_round = time.time()
        # Reference for drift = global FULL state at round start = encoder + mean(client_heads).
        # But "per-client drift" naturally compares to the encoder + per-client head this client
        # actually started the round with — equal to the _broadcast_ encoder + each client's own
        # head_state. We use: drift_i = ||θ_i^{end} − (encoder + client_heads[i]^{round-start})||
        # which means snapshot is per-client. We compute drift inside the loop.
        round_start_full_per_client: dict[str, OrderedDict] = {
            c.apt: clone_state_dict(_merge_aux(encoder_state, client_heads[c.apt])) for c in clients
        }

        new_encoders: list[dict] = []
        encoder_weights: list[float] = []
        local_states: list[dict] = []  # for round-end logging (drift)
        round_main_sum, round_aux_sum, round_n = 0.0, 0.0, 0

        for client in clients:
            local_sd = _merge_aux(encoder_state, client_heads[client.apt])
            apply_state_dict(global_model, local_sd)
            loader = _client_loader(client, batch_size, shuffle=True)
            diag = _local_fedrep_aux(
                global_model, loader,
                n_epochs_total=local_epochs, head_epochs=head_epochs,
                lr=lr, weight_decay=weight_decay, use_amp=use_amp,
                aux_lambda=aux_lambda, hr_weight=hr_weight,
            )
            new_sd = clone_state_dict(global_model.state_dict())
            enc_k, head_k = _split_state_dict_aux(new_sd)
            new_encoders.append(enc_k)
            encoder_weights.append(float(client.n_train_windows))
            client_heads[client.apt] = head_k
            local_states.append(new_sd)
            round_main_sum += diag["main_loss_mean"] * diag["n_batches"]
            round_aux_sum  += diag["aux_loss_mean"]  * diag["n_batches"]
            round_n += diag["n_batches"]

        encoder_state = weighted_average(new_encoders, encoder_weights)
        history["rounds"].append(r)
        history["main_loss"].append(round_main_sum / max(round_n, 1))
        history["aux_loss"].append(round_aux_sum / max(round_n, 1))
        history["n_clients"].append(n_clients)
        wall = time.time() - t_round
        print(f"  round {r:2d}: main={history['main_loss'][-1]:.4f}  "
              f"aux={history['aux_loss'][-1]:.4f}  wall={wall:.1f}s")

        if on_round_end is not None:
            # Build a "global FULL state" for the logger by combining the new
            # encoder with the mean-head across clients (same convention as
            # ``src/fl/fedrep.py`` cold inference).
            head_states = list(client_heads.values())
            head_w = [1.0] * len(head_states)
            mean_head = weighted_average(head_states, head_w)
            global_full = _merge_aux(encoder_state, mean_head)
            apply_state_dict(global_model, global_full)
            # drift_l2 reference: round-start FULL state per client, computed
            # outside the helper. We pass round_start_full_per_client (ordered
            # the same way as local_states) and let the logger compute the
            # mean L2.
            # Logger expects a single ``server_state_pre``, so we provide
            # ``encoder_state_pre + mean(client_heads^{round-start})`` as the
            # "global anchor" — matches the cold-inference convention.
            mean_head_pre = weighted_average(
                [round_start_full_per_client[c.apt] for c in clients],
                [1.0] * n_clients,
            )
            on_round_end(
                round_idx=r,
                model=global_model,
                server_state_pre=mean_head_pre,
                client_states=local_states,
                comm_stats={
                    # FedRep broadcasts encoder only; uploads encoder only too.
                    "upload_bytes_round": encoder_bytes * n_clients,
                    "broadcast_bytes_round": encoder_bytes,
                },
                wall_seconds=wall,
                train_stats={
                    "loss_mean_last_epoch": history["main_loss"][-1],
                    "n_steps_round": int(round_n),
                },
            )

    # Return the cold-style FULL state dict (encoder + mean head across all clients).
    head_states = list(client_heads.values())
    head_w = [1.0] * len(head_states)
    mean_head = weighted_average(head_states, head_w)
    final_full = _merge_aux(encoder_state, mean_head)
    return {
        "history": history,
        "final_state_dict": final_full,
        "encoder_state_dict": encoder_state,
        "n_train_clients": n_clients,
    }


# ---------------------------------------------------------------------------
# Ditto-aux  (two-model loop: federated global + per-client personal v_k)
# ---------------------------------------------------------------------------


def _local_step_aux_with_pull(
    model: NBEATSxAux,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    use_amp: bool,
    aux_lambda: float,
    hr_weight: float,
    snapshot: dict[str, torch.Tensor],
    lam: float,
) -> dict:
    """Personal-model loss for Ditto:
    ``L = MAE(ŷ(v_k), y) + λ·peak_aux(...) + (lam/2) · ||v_k − w_global^{round-start}||²``.
    """
    return _local_step_aux_with_quadratic_anchor(
        model, loader, optimizer, n_epochs, use_amp,
        aux_lambda, hr_weight, snapshot, coef=lam, anchor_key="pull_loss_mean",
    )


def _ditto_round_loop_aux(
    clients: list[ClientData],
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
    lam: float = 0.1,
    on_round_end: Optional[Callable] = None,
) -> dict:
    """Ditto over NBEATSxAux. Logger sees the *global* model state (FedAvg
    leg). Personal models (one per client) are kept locally and the final
    return reports the mean-personal as ``final_state_dict`` (matching the
    v04 ``src/fl/ditto.py`` convention).
    """
    global_model   = init_backbone_aux(seed=seed)
    personal_model = init_backbone_aux(seed=seed)
    global_state   = clone_state_dict(global_model.state_dict())
    personal_states: dict[str, OrderedDict] = {
        c.apt: clone_state_dict(global_state) for c in clients
    }
    history = {"rounds": [], "main_loss": [], "aux_loss": [],
               "personal_main": [], "personal_pull": [], "n_clients": []}
    n_clients = len(clients)
    sd_bytes = _state_dict_bytes(global_state)

    for r in range(1, rounds + 1):
        t_round = time.time()
        round_start_state = clone_state_dict(global_state)
        local_globals: list[dict] = []
        agg_weights: list[float] = []
        gm_main_s, gm_aux_s, gm_n = 0.0, 0.0, 0
        pm_main_s, pm_aux_s, pm_pull_s, pm_n = 0.0, 0.0, 0.0, 0

        for client in clients:
            loader = _client_loader(client, batch_size, shuffle=True)

            # --- Step 1: train GLOBAL model on this client ---
            apply_state_dict(global_model, global_state)
            opt_g = torch.optim.Adam(global_model.parameters(), lr=lr, weight_decay=weight_decay)
            d_g = _local_step_aux(global_model, loader, opt_g, n_epochs=local_epochs,
                                  use_amp=use_amp, aux_lambda=aux_lambda, hr_weight=hr_weight)
            local_globals.append(clone_state_dict(global_model.state_dict()))
            agg_weights.append(float(client.n_train_windows))
            gm_main_s += d_g["main_loss_mean"] * d_g["n_batches"]
            gm_aux_s  += d_g["aux_loss_mean"]  * d_g["n_batches"]
            gm_n += d_g["n_batches"]

            # --- Step 2: train PERSONAL model with pull-to-global ---
            apply_state_dict(personal_model, personal_states[client.apt])
            opt_p = torch.optim.Adam(personal_model.parameters(), lr=lr, weight_decay=weight_decay)
            d_p = _local_step_aux_with_pull(
                personal_model, loader, opt_p,
                n_epochs=local_epochs, use_amp=use_amp,
                aux_lambda=aux_lambda, hr_weight=hr_weight,
                snapshot=round_start_state, lam=lam,
            )
            personal_states[client.apt] = clone_state_dict(personal_model.state_dict())
            pm_main_s += d_p["main_loss_mean"] * d_p["n_batches"]
            pm_aux_s  += d_p["aux_loss_mean"]  * d_p["n_batches"]
            pm_pull_s += d_p["pull_loss_mean"] * d_p["n_batches"]
            pm_n += d_p["n_batches"]

        global_state = weighted_average(local_globals, agg_weights)
        history["rounds"].append(r)
        history["main_loss"].append(gm_main_s / max(gm_n, 1))
        history["aux_loss"].append(gm_aux_s  / max(gm_n, 1))
        history["personal_main"].append(pm_main_s / max(pm_n, 1))
        history["personal_pull"].append(pm_pull_s / max(pm_n, 1))
        history["n_clients"].append(n_clients)
        wall = time.time() - t_round
        print(f"  round {r:2d}: gm_main={history['main_loss'][-1]:.4f}  "
              f"pm_main={history['personal_main'][-1]:.4f}  "
              f"pm_pull={history['personal_pull'][-1]:.4f}  wall={wall:.1f}s")

        if on_round_end is not None:
            # Logger sees the mean of personal models (matches v04 ditto cold
            # convention; this is the model that yields the headline cold
            # PAPE in conference Table line 200).
            personal_list = list(personal_states.values())
            mean_personal = weighted_average(personal_list, [1.0] * len(personal_list))
            apply_state_dict(global_model, mean_personal)  # reuse global_model as carrier
            on_round_end(
                round_idx=r,
                model=global_model,
                server_state_pre=round_start_state,  # global-side reference (drift on global leg)
                client_states=local_globals,
                comm_stats={
                    # Ditto broadcasts/uploads only the global model (personal stays local).
                    "upload_bytes_round": sd_bytes * n_clients,
                    "broadcast_bytes_round": sd_bytes,
                },
                wall_seconds=wall,
                train_stats={
                    "loss_mean_last_epoch": history["main_loss"][-1],
                    "n_steps_round": int(gm_n),
                },
            )
            # Restore global_model → global_state so the next round broadcasts
            # the right thing.
            apply_state_dict(global_model, global_state)

    personal_list = list(personal_states.values())
    mean_personal = weighted_average(personal_list, [1.0] * len(personal_list))
    return {
        "history": history,
        "final_state_dict": mean_personal,
        "global_state_dict": global_state,
        "n_train_clients": n_clients,
    }


# ---------------------------------------------------------------------------
# FedProto-aux  (per-cluster prototype federation on h_generic)
# ---------------------------------------------------------------------------


def _local_step_aux_with_proto(
    model: NBEATSxAux,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    use_amp: bool,
    aux_lambda: float,
    hr_weight: float,
    global_prototypes: torch.Tensor,
    lambda_proto: float,
) -> dict:
    """``L = MAE + λ·peak_aux + λ_proto · MSE(h_g, global_proto[nearest])``."""
    amp_ctx = _wrap_amp(use_amp)
    model.train()
    n_batches = 0
    sum_main, sum_aux, sum_proto = 0.0, 0.0, 0.0
    proto_t = global_prototypes.detach()
    for _ in range(n_epochs):
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                y_hat, hiddens, (amp_pred, hr_pred) = model(x)
                main = F.l1_loss(y_hat, y)
                aux  = peak_aux_loss(amp_pred, hr_pred, y, hr_weight=hr_weight)
                h_g = hiddens["h_generic"]
                d2 = (
                    h_g.float().pow(2).sum(1, keepdim=True)
                    - 2.0 * h_g.float() @ proto_t.t()
                    + proto_t.pow(2).sum(1)
                )
                c_batch = d2.argmin(dim=1)
                target = proto_t[c_batch]
                proto_loss = F.mse_loss(h_g.float(), target)
                loss = main + aux_lambda * aux + lambda_proto * proto_loss
            loss.backward()
            optimizer.step()
            sum_main += float(main.item())
            sum_aux  += float(aux.item())
            sum_proto += float(proto_loss.item())
            n_batches += 1
    return {
        "n_batches": n_batches,
        "main_loss_mean":  sum_main  / max(n_batches, 1),
        "aux_loss_mean":   sum_aux   / max(n_batches, 1),
        "proto_loss_mean": sum_proto / max(n_batches, 1),
    }


def _gather_h_g(model: NBEATSxAux, loader: DataLoader, use_amp: bool) -> np.ndarray:
    amp_ctx = _wrap_amp(use_amp)
    h_chunks = []
    model.eval()
    with torch.no_grad():
        for x, _y in loader:
            x = x.to(DEVICE, non_blocking=True)
            with amp_ctx:
                _yh, hiddens, _aux = model(x)
            h_chunks.append(hiddens["h_generic"].float().cpu().numpy())
    return np.concatenate(h_chunks, axis=0).astype(np.float32) if h_chunks else np.zeros((0, D_MODEL), dtype=np.float32)


def _initial_prototypes_aux(
    model: NBEATSxAux,
    clients: list[ClientData],
    K: int,
    batch_size: int,
    use_amp: bool,
    seed: int,
) -> np.ndarray:
    h_pool = []
    for client in clients:
        loader = DataLoader(client.train_set, batch_size=batch_size, shuffle=False)
        h = _gather_h_g(model, loader, use_amp=use_amp)
        if len(h) > 10_000:
            rng = np.random.default_rng(seed)
            h = h[rng.choice(len(h), 10_000, replace=False)]
        h_pool.append(h)
    h_all = np.concatenate(h_pool, axis=0)
    km = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=seed).fit(h_all)
    return km.cluster_centers_.astype(np.float32)


def _local_h_g_centroids_aux(
    model: NBEATSxAux,
    loader: DataLoader,
    K: int,
    init_centroids: np.ndarray,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    h_arr = _gather_h_g(model, loader, use_amp=use_amp)
    if len(h_arr) < K:
        if len(h_arr) <= 1:
            counts = np.zeros(K, dtype=np.int64)
            return init_centroids.astype(np.float32), counts
        km = KMeans(n_clusters=len(h_arr), init="k-means++", n_init=10).fit(h_arr)
        centroids = init_centroids.copy()
        centroids[: len(h_arr)] = km.cluster_centers_.astype(np.float32)
        labels = km.labels_
        counts = np.zeros(K, dtype=np.int64)
        bc = np.bincount(labels, minlength=len(h_arr)).astype(np.int64)
        counts[: len(h_arr)] = bc
        return centroids.astype(np.float32), counts
    km = KMeans(n_clusters=K, init=init_centroids.astype(np.float32), n_init=1).fit(h_arr)
    counts = np.bincount(km.labels_, minlength=K).astype(np.int64)
    return km.cluster_centers_.astype(np.float32), counts


def _aggregate_prototypes(
    centroids_by_client: list[np.ndarray],
    counts_by_client: list[np.ndarray],
) -> np.ndarray:
    K, D = centroids_by_client[0].shape
    out = np.zeros((K, D), dtype=np.float32)
    for c in range(K):
        weights = np.array([cnts[c] for cnts in counts_by_client], dtype=np.float64)
        if weights.sum() <= 0:
            out[c] = np.mean([cs[c] for cs in centroids_by_client], axis=0)
            continue
        weights = weights / weights.sum()
        out[c] = np.sum([cs[c] * w for cs, w in zip(centroids_by_client, weights)], axis=0)
    return out


def _fedproto_round_loop_aux(
    clients: list[ClientData],
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
    K: int = 32,
    lambda_proto: float = 0.1,
    on_round_end: Optional[Callable] = None,
) -> dict:
    """FedProto over NBEATSxAux + combined loss + prototype-aligned regulariser."""
    global_model = init_backbone_aux(seed=seed)
    global_state = clone_state_dict(global_model.state_dict())

    print(f"[FedProto-aux] computing initial prototypes (K={K}) from pooled h_g...")
    apply_state_dict(global_model, global_state)
    global_prototypes_np = _initial_prototypes_aux(
        global_model, clients, K=K, batch_size=batch_size, use_amp=use_amp, seed=seed,
    )
    n_clients = len(clients)
    sd_bytes = _state_dict_bytes(global_state)
    proto_bytes = K * D_MODEL * 4  # K × D × fp32 bytes — roughly 8 KB for K=32

    history = {"rounds": [], "main_loss": [], "aux_loss": [], "proto_loss": [], "n_clients": []}
    for r in range(1, rounds + 1):
        t_round = time.time()
        round_start_state = clone_state_dict(global_state)
        proto_anchor = torch.from_numpy(global_prototypes_np).to(DEVICE)

        local_states: list[dict] = []
        local_weights: list[float] = []
        local_centroids: list[np.ndarray] = []
        local_counts: list[np.ndarray] = []
        round_main_s, round_aux_s, round_proto_s, round_n = 0.0, 0.0, 0.0, 0

        for client in clients:
            apply_state_dict(global_model, global_state)
            optimizer = torch.optim.Adam(
                global_model.parameters(), lr=lr, weight_decay=weight_decay
            )
            loader = _client_loader(client, batch_size, shuffle=True)
            diag = _local_step_aux_with_proto(
                global_model, loader, optimizer,
                n_epochs=local_epochs, use_amp=use_amp,
                aux_lambda=aux_lambda, hr_weight=hr_weight,
                global_prototypes=proto_anchor, lambda_proto=lambda_proto,
            )
            no_shuffle = DataLoader(client.train_set, batch_size=batch_size, shuffle=False)
            cents, cnts = _local_h_g_centroids_aux(
                global_model, no_shuffle, K=K, init_centroids=global_prototypes_np, use_amp=use_amp,
            )
            local_states.append(clone_state_dict(global_model.state_dict()))
            local_weights.append(float(client.n_train_windows))
            local_centroids.append(cents)
            local_counts.append(cnts)
            round_main_s  += diag["main_loss_mean"]  * diag["n_batches"]
            round_aux_s   += diag["aux_loss_mean"]   * diag["n_batches"]
            round_proto_s += diag["proto_loss_mean"] * diag["n_batches"]
            round_n += diag["n_batches"]

        global_state = weighted_average(local_states, local_weights)
        global_prototypes_np = _aggregate_prototypes(local_centroids, local_counts)
        history["rounds"].append(r)
        history["main_loss"].append(round_main_s / max(round_n, 1))
        history["aux_loss"].append(round_aux_s   / max(round_n, 1))
        history["proto_loss"].append(round_proto_s / max(round_n, 1))
        history["n_clients"].append(n_clients)
        wall = time.time() - t_round
        print(f"  round {r:2d}: main={history['main_loss'][-1]:.4f}  "
              f"aux={history['aux_loss'][-1]:.4f}  proto={history['proto_loss'][-1]:.4f}  wall={wall:.1f}s")

        if on_round_end is not None:
            apply_state_dict(global_model, global_state)
            on_round_end(
                round_idx=r,
                model=global_model,
                server_state_pre=round_start_state,
                client_states=local_states,
                comm_stats={
                    # FedProto uploads backbone + per-client centroids; broadcasts backbone + prototypes.
                    "upload_bytes_round":    sd_bytes * n_clients + proto_bytes * n_clients,
                    "broadcast_bytes_round": sd_bytes + proto_bytes,
                },
                wall_seconds=wall,
                train_stats={
                    "loss_mean_last_epoch": history["main_loss"][-1],
                    "n_steps_round": int(round_n),
                },
            )

    return {
        "history": history,
        "final_state_dict": global_state,
        "global_prototypes": global_prototypes_np,
        "n_train_clients": n_clients,
    }


# ---------------------------------------------------------------------------
# Public dispatch entry
# ---------------------------------------------------------------------------


_DISPATCH = {
    "fedavg":   _fedavg_round_loop_aux,
    "fedprox":  _fedprox_round_loop_aux,
    "fedrep":   _fedrep_round_loop_aux,
    "ditto":    _ditto_round_loop_aux,
    "fedproto": _fedproto_round_loop_aux,
}


def run_fl_aux(
    algorithm: str,
    splits: dict[str, dict],
    *,
    rounds: int,
    local_epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    seed: int,
    use_amp: bool,
    aux_lambda: float = 0.3,
    hr_weight: float = 0.1,
    on_round_end: Optional[Callable] = None,
    **algo_kwargs,
) -> dict:
    """Top-level v06 FL dispatch.

    Builds ``ClientData`` from the v06 per-client split blocks (no disk
    re-read), then calls the algorithm-specific round-loop on NBEATSxAux +
    combined MAE+peak_aux loss.

    Algorithm-specific extras
    -------------------------
    - fedprox  : ``mu`` (default 0.01).
    - fedrep   : ``head_epochs`` (default 1, paper §5 short-horizon recipe).
    - ditto    : ``lam`` (default 0.1).
    - fedproto : ``K`` (default 32), ``lambda_proto`` (default 0.1).
    """
    algorithm = algorithm.lower()
    if algorithm not in _DISPATCH:
        raise ValueError(f"run_fl_aux: unknown algorithm {algorithm!r}; choices: {sorted(_DISPATCH)}")
    clients = _build_clients_from_v06_splits(splits)
    if not clients:
        raise RuntimeError(f"run_fl_aux({algorithm}): no clients constructed from splits")
    fn = _DISPATCH[algorithm]
    return fn(
        clients,
        rounds=rounds, local_epochs=local_epochs, lr=lr,
        batch_size=batch_size, weight_decay=weight_decay,
        seed=seed, use_amp=use_amp,
        aux_lambda=aux_lambda, hr_weight=hr_weight,
        on_round_end=on_round_end,
        **algo_kwargs,
    )
