"""FedProx (Li et al., MLSys'20) on the v04 NBEATSx backbone.

Reference: T. Li et al., "Federated Optimization in Heterogeneous Networks",
MLSys'20. arxiv:1812.06127. Official code (TF1) cached in
``papers/literlature/fedprox_official/`` — see ``pgd.py`` for the
authoritative perturbed-gradient-descent definition.

Algorithm difference from FedAvg
--------------------------------
At each local SGD step, the loss adds a proximal regulariser pulling
the local weights toward the global snapshot at the start of the round:

    L_local(w) = MAE(y_hat, y) + (mu/2) * ||w - w_global||²

The official ``PerturbedGradientDescent._apply_dense`` applies this as
``var <- var - lr * (grad + mu * (var - vstar))`` where ``vstar`` is
the round-start global snapshot. Equivalent to adding the prox term
to the loss and using vanilla SGD/Adam.

We follow the loss-augmentation form because PyTorch's autograd makes
it cleaner; ``mu`` defaults to 0.01 (FedProx paper §5 default).

Cold inference identical to FedAvg.
"""

from __future__ import annotations

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
class FedProxConfig(FLConfig):
    """FedProx hyperparameters: FLConfig + ``mu`` for the proximal term."""

    mu: float = 0.01


def _make_prox_loss(global_state: dict, mu: float):
    """Return an ``extra_loss_fn`` that adds (mu/2)·||w - w_global||² each step.

    The closure captures the round-start global snapshot; per-step we sum
    squared diffs across all *floating-point* parameters (matches the
    official PerturbedGradientDescent: only updates the float tensor
    update rule).
    """
    # Stash a CPU copy of the snapshot once; we move per-key tensors to the
    # parameter's device on the fly inside the closure.
    snapshot = {k: v.detach().clone() for k, v in global_state.items()}

    def extra(model, x, y, y_hat):
        device = next(model.parameters()).device
        prox = torch.zeros((), device=device)
        for n, p in model.named_parameters():
            if not p.requires_grad or not p.is_floating_point():
                continue
            ref = snapshot[n].to(device)
            prox = prox + ((p - ref) ** 2).sum()
        return 0.5 * mu * prox

    return extra


def train_fedprox(
    train_apts: list[str],
    cold_apts: list[str],
    cfg: FedProxConfig,
) -> dict:
    """Run FedProx for ``cfg.rounds`` rounds and return a result dict.

    Output schema is the same as FedAvg's so the v04 aggregator can read
    them uniformly.
    """
    clients: list[ClientData] = build_clients(train_apts)
    if len(clients) == 0:
        raise RuntimeError("FedProx: no train clients")

    global_model = init_backbone(seed=cfg.seed)
    global_state = clone_state_dict(global_model.state_dict())

    history = FLHistory()
    for r in range(1, cfg.rounds + 1):
        local_states: list[dict] = []
        local_weights: list[float] = []
        round_loss_sum = 0.0
        round_extra_sum = 0.0
        round_loss_n = 0

        participating = clients
        if cfg.clients_per_round > 0 and cfg.clients_per_round < len(clients):
            torch.manual_seed(cfg.seed * 10_000 + r)
            idx = torch.randperm(len(clients))[: cfg.clients_per_round].tolist()
            participating = [clients[i] for i in idx]

        # The proximal anchor is the round-start global snapshot,
        # identical for every client of that round.
        prox_extra = _make_prox_loss(global_state, mu=cfg.mu)

        for client in participating:
            apply_state_dict(global_model, global_state)
            optimizer = torch.optim.Adam(
                global_model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
            loader = client_loader(client, cfg.batch_size, shuffle=True)
            diag = run_local_epochs(
                global_model, loader, optimizer,
                n_epochs=cfg.local_epochs,
                extra_loss_fn=prox_extra,
                use_amp=cfg.use_amp,
            )
            local_states.append(clone_state_dict(global_model.state_dict()))
            local_weights.append(float(client.n_train_windows))
            round_loss_sum += diag["main_loss_mean"] * diag["n_batches"]
            if diag["extra_loss_mean"] is not None:
                round_extra_sum += diag["extra_loss_mean"] * diag["n_batches"]
            round_loss_n += diag["n_batches"]

        global_state = weighted_average(local_states, local_weights)
        history.append(
            round_idx=r,
            train_loss=round_loss_sum / max(round_loss_n, 1),
            n_clients=len(participating),
            extra={"prox_loss_mean": round_extra_sum / max(round_loss_n, 1)},
        )

    apply_state_dict(global_model, global_state)
    cold_metrics = evaluate_cold(global_model, cold_apts, use_amp=cfg.use_amp)

    return {
        "algorithm": "fedprox",
        "config": cfg.__dict__,
        "history": history.as_dict(),
        "cold_metrics": cold_metrics,
        "n_train_clients": len(clients),
        "final_state_dict": global_state,
    }
