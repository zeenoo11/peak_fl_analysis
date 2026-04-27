"""Ditto (Li et al., ICML'21) on the v04 NBEATSx backbone.

Reference: T. Li, S. Hu, A. Beirami, V. Smith, "Ditto: Fair and Robust
Federated Learning Through Personalization", ICML'21. arxiv:2012.04221.
Official code (TF1) cached in ``papers/literlature/ditto_official/``;
see ``ditto.py``'s ``LocalUpdateDitto.train`` for the authoritative
two-model loop.

Algorithm (per round, per client)
---------------------------------
Ditto maintains **two** models per client:

1. ``w_global``  — the FedAvg-trained global model. Each round, every
   client locally trains *this* model with vanilla MAE-loss SGD; the
   server then aggregates as in FedAvg.
2. ``v_k``       — a per-client personal model. Each round, every
   client locally trains ``v_k`` with the regularised loss

       L_personal(v_k) = MAE(y_hat(v_k), y) + (lam/2) * ||v_k - w_global||²

   where ``w_global`` is the round-start global snapshot. ``v_k`` is
   never broadcast back to the server.

The two trainings are interleaved; the v04 implementation runs them
in the same round. Cold-side inference uses the **mean v_k across train
clients** as the personal model (held-out cold apts have no v_k of
their own); the alternative — using ``w_global`` — is identical to
FedAvg and would not exercise Ditto's personalisation. Mean-personal
matches the FedRep cold-inference choice for symmetry.

Defaults follow Ditto §5: ``lam=0.1``.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass

import torch

from fl.base import (
    ClientData,
    FLConfig,
    FLHistory,
    apply_state_dict,
    build_clients,
    client_loader,
    clone_state_dict,
    evaluate_cold,
    init_backbone,
    run_local_epochs,
    weighted_average,
)


@dataclass
class DittoConfig(FLConfig):
    """Ditto hyperparameters: FLConfig + ``lam`` for the personal-model regulariser."""

    lam: float = 0.1


def _make_pull_loss(reference_state: dict, lam: float):
    """``extra_loss_fn`` adding (lam/2)·||theta - reference||² for the personal model.

    Identical formulation to FedProx's prox term, but the *direction* is
    different: in FedProx the prox pulls each client's local copy of
    the **global** model toward the round-start global snapshot; in
    Ditto it pulls the **personal** model toward that same snapshot.
    """
    snapshot = {k: v.detach().clone() for k, v in reference_state.items()}

    def extra(model, x, y, y_hat):
        device = next(model.parameters()).device
        pull = torch.zeros((), device=device)
        for n, p in model.named_parameters():
            if not p.requires_grad or not p.is_floating_point():
                continue
            ref = snapshot[n].to(device)
            pull = pull + ((p - ref) ** 2).sum()
        return 0.5 * lam * pull

    return extra


def train_ditto(
    train_apts: list[str],
    cold_apts: list[str],
    cfg: DittoConfig,
) -> dict:
    """Run Ditto for ``cfg.rounds`` rounds and return a result dict.

    Cold inference uses the mean of all per-client personal models
    ``v_k`` as the cold-side default model.
    """
    clients: list[ClientData] = build_clients(train_apts)
    if len(clients) == 0:
        raise RuntimeError("Ditto: no train clients")

    global_model = init_backbone(seed=cfg.seed)
    global_state = clone_state_dict(global_model.state_dict())

    # Per-client personal models v_k, all initialised to the same global init.
    personal_states: dict[str, OrderedDict] = {
        c.apt: clone_state_dict(global_state) for c in clients
    }
    # We re-use a single nn.Module instance for personal-model training
    # to save GPU memory; copy the per-client v_k into it on demand.
    personal_model = init_backbone(seed=cfg.seed)

    history = FLHistory()
    for r in range(1, cfg.rounds + 1):
        local_globals: list[dict] = []
        agg_weights: list[float] = []
        round_main_g_sum = 0.0
        round_main_g_n = 0
        round_main_p_sum = 0.0
        round_pull_sum = 0.0
        round_main_p_n = 0

        participating = clients
        if cfg.clients_per_round > 0 and cfg.clients_per_round < len(clients):
            torch.manual_seed(cfg.seed * 10_000 + r)
            idx = torch.randperm(len(clients))[: cfg.clients_per_round].tolist()
            participating = [clients[i] for i in idx]

        # Personal-model regulariser anchors at the round-start global snapshot.
        pull_extra = _make_pull_loss(global_state, lam=cfg.lam)

        for client in participating:
            loader = client_loader(client, cfg.batch_size, shuffle=True)

            # --- Step 1: train the GLOBAL model on this client (FedAvg routine) ---
            apply_state_dict(global_model, global_state)
            opt_g = torch.optim.Adam(
                global_model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
            d_g = run_local_epochs(global_model, loader, opt_g, n_epochs=cfg.local_epochs, use_amp=cfg.use_amp)
            local_globals.append(clone_state_dict(global_model.state_dict()))
            agg_weights.append(float(client.n_train_windows))
            round_main_g_sum += d_g["main_loss_mean"] * d_g["n_batches"]
            round_main_g_n += d_g["n_batches"]

            # --- Step 2: train this client's PERSONAL model with pull-to-global ---
            apply_state_dict(personal_model, personal_states[client.apt])
            opt_p = torch.optim.Adam(
                personal_model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
            d_p = run_local_epochs(
                personal_model, loader, opt_p,
                n_epochs=cfg.local_epochs,
                extra_loss_fn=pull_extra,
                use_amp=cfg.use_amp,
            )
            personal_states[client.apt] = clone_state_dict(personal_model.state_dict())
            round_main_p_sum += d_p["main_loss_mean"] * d_p["n_batches"]
            if d_p["extra_loss_mean"] is not None:
                round_pull_sum += d_p["extra_loss_mean"] * d_p["n_batches"]
            round_main_p_n += d_p["n_batches"]

        # Server aggregates the global model only (FedAvg-style).
        global_state = weighted_average(local_globals, agg_weights)
        history.append(
            round_idx=r,
            train_loss=round_main_g_sum / max(round_main_g_n, 1),  # global-side main loss
            n_clients=len(participating),
            extra={
                "personal_main_loss_mean": round_main_p_sum / max(round_main_p_n, 1),
                "personal_pull_loss_mean": round_pull_sum / max(round_main_p_n, 1),
            },
        )

    # Cold inference: mean of all personal models.
    personal_list = list(personal_states.values())
    personal_weights = [1.0] * len(personal_list)
    mean_personal = weighted_average(personal_list, personal_weights)
    apply_state_dict(personal_model, mean_personal)
    cold_metrics_personal = evaluate_cold(personal_model, cold_apts, use_amp=cfg.use_amp)

    # Also report the global model's cold metrics for reference (it equals what
    # FedAvg would produce; useful as a sanity check that personalisation helps).
    apply_state_dict(global_model, global_state)
    cold_metrics_global = evaluate_cold(global_model, cold_apts, use_amp=cfg.use_amp)

    return {
        "algorithm": "ditto",
        "config": cfg.__dict__,
        "history": history.as_dict(),
        "cold_metrics": cold_metrics_personal,            # primary report = personal
        "cold_metrics_global_reference": cold_metrics_global,
        "n_train_clients": len(clients),
        "final_state_dict": mean_personal,
        "global_state_dict": global_state,
    }
