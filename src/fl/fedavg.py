"""FedAvg (McMahan et al., AISTATS'17) on the v04 NBEATSx backbone.

Reference: H. B. McMahan et al., "Communication-Efficient Learning of
Deep Networks from Decentralized Data", AISTATS'17.
arxiv:1602.05629. Official-style code (TF1) cached in
``papers/literlature/fedavg_official/`` (FedProx repo's flearn.trainers
package — McMahan FedAvg has no canonical PyTorch reference, this is
the closest reproduction).

Algorithm (one round)
---------------------
1. Server broadcasts ``w_global`` to all (or sampled) clients.
2. Each client ``k`` runs ``E`` local epochs of MAE-loss SGD on its own
   train segment, producing local weights ``w_k``.
3. Server aggregates: ``w_global = sum_k (n_k / n_total) * w_k``,
   weighted by local sample count.

Identical to v04 ``base.run_local_epochs`` per client + the
``base.weighted_average`` aggregator. No per-algorithm extras.

Cold inference: forward only, denormalise to kW
(``base.evaluate_cold``).
"""

from __future__ import annotations

import torch

from fl.base import (
    DEVICE,
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


def train_fedavg(
    train_apts: list[str],
    cold_apts: list[str],
    cfg: FLConfig,
) -> dict:
    """Run FedAvg for ``cfg.rounds`` rounds and return a result dict.

    Returns
    -------
    {
        "algorithm": "fedavg",
        "config": dict,
        "history": list,                        # per-round diagnostic
        "cold_metrics": {pape, hr@1, hr@2, mae, n_cold_windows, n_cold_apts},
        "n_train_clients": int,
        "final_state_dict": OrderedDict,        # final aggregated global weights (CPU)
    }
    """
    clients: list[ClientData] = build_clients(train_apts)
    if len(clients) == 0:
        raise RuntimeError("FedAvg: no train clients (all apts missing?)")

    # Single global model. Each round: clone global -> local train per client -> aggregate.
    global_model = init_backbone(seed=cfg.seed)
    global_state = clone_state_dict(global_model.state_dict())

    history = FLHistory()
    for r in range(1, cfg.rounds + 1):
        local_states: list[dict] = []
        local_weights: list[float] = []
        round_loss_sum = 0.0
        round_loss_n = 0

        # Optional client sampling (default = all clients participate).
        participating = clients
        if cfg.clients_per_round > 0 and cfg.clients_per_round < len(clients):
            # deterministic per-round sample
            torch.manual_seed(cfg.seed * 10_000 + r)
            idx = torch.randperm(len(clients))[: cfg.clients_per_round].tolist()
            participating = [clients[i] for i in idx]

        for client in participating:
            apply_state_dict(global_model, global_state)  # send global -> client
            optimizer = torch.optim.Adam(
                global_model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
            loader = client_loader(client, cfg.batch_size, shuffle=True)
            diag = run_local_epochs(
                global_model, loader, optimizer, n_epochs=cfg.local_epochs
            )
            local_states.append(clone_state_dict(global_model.state_dict()))
            local_weights.append(float(client.n_train_windows))
            round_loss_sum += diag["main_loss_mean"] * diag["n_batches"]
            round_loss_n += diag["n_batches"]

        global_state = weighted_average(local_states, local_weights)
        round_train_loss = round_loss_sum / max(round_loss_n, 1)
        history.append(
            round_idx=r,
            train_loss=round_train_loss,
            n_clients=len(participating),
        )

    # Load final global weights and evaluate on cold apts.
    apply_state_dict(global_model, global_state)
    cold_metrics = evaluate_cold(global_model, cold_apts)

    return {
        "algorithm": "fedavg",
        "config": cfg.__dict__,
        "history": history.as_dict(),
        "cold_metrics": cold_metrics,
        "n_train_clients": len(clients),
        "final_state_dict": global_state,
    }
