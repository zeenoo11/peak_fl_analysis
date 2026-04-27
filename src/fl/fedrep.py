"""FedRep (Collins et al., ICML'21) on the v04 NBEATSx backbone.

Reference: L. Collins et al., "Exploiting Shared Representations for
Personalized Federated Learning", ICML'21. arxiv:2102.07078. The
``lgcollins/FedRep`` original repo is unavailable; we follow the
``rahulv0205/fedrep_experiments`` mirror cached in
``papers/literlature/fedrep_official/`` (MIT licensed; description
explicitly says "original source ... lgcollins/FedRep.git").

Algorithm (one round, per client)
---------------------------------
The total local budget is ``E`` epochs. Within those:
1. **Head epochs** (first ``E - E_rep``): freeze the **shared encoder**,
   train only the **per-client head**. (Personalisation step.)
2. **Representation epochs** (last ``E_rep``): freeze the head, train
   the encoder. (Shared-representation step.)

After all clients finish, the server averages **only the encoder
weights** across clients; per-client heads stay local. Client-specific
heads are kept across rounds — they are NOT broadcast back from the
server.

NBEATSx encoder/head split
--------------------------
- **encoder** (shared): every ``stack_*.fc{1..4}.{weight,bias}``.
  These produce the per-stack hidden representations
  (h_trend, h_seasonal, h_generic).
- **head** (per-client): every ``stack_*.proj.{weight,bias}``.
  These project hidden -> basis coefficients (theta), i.e. the actual
  forecast/backcast output layer of each stack.

Cold inference
--------------
For a held-out cold apt, the cold-side **does not own a personalised
head**. We use the **average head across train clients** as the
default head at cold inference — a standard FedRep cold-start choice
when the cold client has no labelled data. This matches the v04 plan
note "cold = held-out, no per-client head trained for cold".

(An optional sub-row using a K-shot personalised head would be the
v03 F2a analogue; deferred.)
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

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
class FedRepConfig(FLConfig):
    """FedRep hyperparameters: FLConfig + head/representation epoch split.

    Defaults follow the FedRep paper §5 short-horizon recipe:
    ``local_epochs=2`` total; ``head_epochs=1`` (head first, representation second).
    """

    head_epochs: int = 1   # ``E - E_rep`` in the paper notation; rep_epochs = local_epochs - head_epochs


# --- encoder / head split for NBEATSx ---


def _is_head_param(name: str) -> bool:
    """``stack_*.proj.*`` are per-client heads; everything else is shared encoder."""
    return ".proj." in name


def _set_requires_grad(model: torch.nn.Module, *, train_head: bool) -> None:
    """Freeze encoder OR head depending on ``train_head``."""
    for n, p in model.named_parameters():
        if _is_head_param(n):
            p.requires_grad = train_head
        else:
            p.requires_grad = not train_head


def _split_state_dict(sd: dict) -> tuple[OrderedDict, OrderedDict]:
    """Split a state dict into (encoder_part, head_part) via parameter name."""
    enc = OrderedDict()
    head = OrderedDict()
    for k, v in sd.items():
        (head if _is_head_param(k) else enc)[k] = v
    return enc, head


def _merge(encoder_sd: dict, head_sd: dict) -> OrderedDict:
    out = OrderedDict()
    out.update(encoder_sd)
    out.update(head_sd)
    return out


# --- per-client local FedRep loop ---


def _local_fedrep(
    model: torch.nn.Module,
    loader,
    cfg: FedRepConfig,
) -> dict:
    """Run FedRep's two-phase local training. Returns a small diag dict.

    Phase 1: train head only (encoder frozen) for ``head_epochs``.
    Phase 2: train encoder only (head frozen) for ``rep_epochs``.
    """
    rep_epochs = max(0, cfg.local_epochs - cfg.head_epochs)
    sum_main = 0.0
    n_batches = 0

    # Phase 1 (head): one optimizer over head parameters only.
    if cfg.head_epochs > 0:
        _set_requires_grad(model, train_head=True)
        opt_head = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        d = run_local_epochs(model, loader, opt_head, n_epochs=cfg.head_epochs, use_amp=cfg.use_amp)
        sum_main += d["main_loss_mean"] * d["n_batches"]
        n_batches += d["n_batches"]

    # Phase 2 (representation): optimizer over encoder parameters only.
    if rep_epochs > 0:
        _set_requires_grad(model, train_head=False)
        opt_enc = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        d = run_local_epochs(model, loader, opt_enc, n_epochs=rep_epochs, use_amp=cfg.use_amp)
        sum_main += d["main_loss_mean"] * d["n_batches"]
        n_batches += d["n_batches"]

    # Re-enable all gradients before the next round (cleanup hygiene).
    for p in model.parameters():
        p.requires_grad = True

    return {"main_loss_mean": sum_main / max(n_batches, 1), "n_batches": n_batches}


def train_fedrep(
    train_apts: list[str],
    cold_apts: list[str],
    cfg: FedRepConfig,
) -> dict:
    """Run FedRep for ``cfg.rounds`` rounds and return a result dict.

    Server keeps a global encoder state. Each client also keeps its own
    head state across rounds (never broadcast). At cold inference we use
    the **mean head** across train clients as the default head.
    """
    clients: list[ClientData] = build_clients(train_apts)
    if len(clients) == 0:
        raise RuntimeError("FedRep: no train clients")

    # Server state: encoder only.
    global_model = init_backbone(seed=cfg.seed)
    init_sd = clone_state_dict(global_model.state_dict())
    encoder_state, init_head = _split_state_dict(init_sd)

    # Per-client head states. All initialised to the same head as the global init,
    # then diverge as each client trains its own head.
    client_heads: dict[str, OrderedDict] = {
        c.apt: OrderedDict((k, v.clone()) for k, v in init_head.items()) for c in clients
    }

    history = FLHistory()
    for r in range(1, cfg.rounds + 1):
        new_encoders: list[dict] = []
        encoder_weights: list[float] = []
        round_loss_sum = 0.0
        round_loss_n = 0

        participating = clients
        if cfg.clients_per_round > 0 and cfg.clients_per_round < len(clients):
            torch.manual_seed(cfg.seed * 10_000 + r)
            idx = torch.randperm(len(clients))[: cfg.clients_per_round].tolist()
            participating = [clients[i] for i in idx]

        for client in participating:
            # Send current global encoder + this client's own head.
            local_sd = _merge(encoder_state, client_heads[client.apt])
            apply_state_dict(global_model, local_sd)
            loader = client_loader(client, cfg.batch_size, shuffle=True)

            diag = _local_fedrep(global_model, loader, cfg)
            round_loss_sum += diag["main_loss_mean"] * diag["n_batches"]
            round_loss_n += diag["n_batches"]

            # Extract updated encoder and head from this client.
            new_sd = clone_state_dict(global_model.state_dict())
            enc_k, head_k = _split_state_dict(new_sd)
            new_encoders.append(enc_k)
            encoder_weights.append(float(client.n_train_windows))
            # Save the client's updated head (kept locally — NOT broadcast back).
            client_heads[client.apt] = head_k

        # Server aggregates encoder only.
        encoder_state = weighted_average(new_encoders, encoder_weights)
        history.append(
            round_idx=r,
            train_loss=round_loss_sum / max(round_loss_n, 1),
            n_clients=len(participating),
        )

    # Cold inference: use the *mean head* across all (final) client heads as the
    # default head for held-out cold apts (paper-standard FedRep cold-start).
    head_states = list(client_heads.values())
    head_weights = [1.0] * len(head_states)
    mean_head = weighted_average(head_states, head_weights)
    cold_state = _merge(encoder_state, mean_head)
    apply_state_dict(global_model, cold_state)
    cold_metrics = evaluate_cold(global_model, cold_apts, use_amp=cfg.use_amp)

    return {
        "algorithm": "fedrep",
        "config": cfg.__dict__,
        "history": history.as_dict(),
        "cold_metrics": cold_metrics,
        "n_train_clients": len(clients),
        "final_state_dict": cold_state,        # encoder + mean-head
        "encoder_state_dict": encoder_state,   # also expose raw encoder
    }
