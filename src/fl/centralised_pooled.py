"""Centralised pooled SGD on NBEATSxAux for v06 V6-Dyn-A reference.

(한글 요약)
v06 의 *상한선* (G1, plan §"Goals") — 100가구 train 윈도우를 단일 pool 로
합쳐 NBEATSxAux 를 학습. 라운드 (= epoch) 끝에 ``on_round_end`` 를 호출해서
v06 의 ``RoundLogger`` 가 다른 5종 FL cell 과 같은 jsonl 행을 쓸 수 있도록
한다. comm_stats 는 모두 0 (federated 가 아니므로 client-server 통신 없음).

Public surface
--------------
- ``centralised_pooled_train(splits, *, n_epochs, lr, batch_size, ...)``

Loss
----
``L = MAE(ŷ, y) + λ · peak_aux(ŷ, y; hr_weight)`` — same as fedavg_aux.
Optimiser: Adam(lr=1e-3, weight_decay=1e-5).

Output
------
Returns ``{"history": {...}, "final_state_dict": OrderedDict, "n_clients":
int, "elapsed_seconds": float}``. ``final_state_dict`` is on CPU and
loadable with ``strict=True`` into a fresh NBEATSxAux.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from fl.base import DEVICE, _NullCtx, clone_state_dict
from fl.fedavg_aux import init_backbone_aux
from models.nbeatsx_aux import NBEATSxAux
from models.peak_aux_head import peak_aux_loss


def _make_pooled_loader(
    splits: dict[str, dict], batch_size: int, shuffle: bool = True
) -> tuple[DataLoader, int]:
    """Build a single DataLoader over the ConcatDataset of all clients' train sets."""
    datasets = []
    for apt, sp in splits.items():
        x = torch.from_numpy(sp["train_x"])
        y = torch.from_numpy(sp["train_y"])
        if x.shape[0] > 0:
            datasets.append(TensorDataset(x, y))
    if not datasets:
        raise RuntimeError("centralised_pooled_train: no train windows in splits")
    pooled = ConcatDataset(datasets)
    return DataLoader(pooled, batch_size=batch_size, shuffle=shuffle, drop_last=False), len(datasets)


def centralised_pooled_train(
    splits: dict[str, dict],
    *,
    n_epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    aux_lambda: float = 0.3,
    hr_weight: float = 0.1,
    seed: int = 42,
    use_amp: bool = True,
    on_round_end: Optional[Callable] = None,
) -> dict:
    """Pool 100 clients' train sets into one DataLoader and train NBEATSxAux.

    Per-epoch the optional ``on_round_end`` callback is invoked with:
        on_round_end(
            round_idx=epoch_idx (1-based),
            model=model,
            server_state_pre=None,
            client_states=None,
            comm_stats={'upload_bytes_round': 0, 'broadcast_bytes_round': 0},
            wall_seconds=epoch_wall_seconds,
            train_stats={'loss_mean_last_epoch': ..., 'n_steps_round': ...},
        )
    so the v06 RoundLogger writes the same JSONL schema as the FL cells.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    use_amp_eff = use_amp and (DEVICE.type == "cuda")
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp_eff else _NullCtx()
    )

    model = init_backbone_aux(seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pooled_loader, n_clients = _make_pooled_loader(splits, batch_size, shuffle=True)

    history = {"epochs": [], "main_loss": [], "aux_loss": [], "n_steps": []}
    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        model.train()
        t_epoch = time.time()
        sum_main, sum_aux, n_batches = 0.0, 0.0, 0
        for x, y in pooled_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                y_hat, _h, (amp_pred, hr_pred) = model(x)
                main = F.l1_loss(y_hat, y)
                aux = peak_aux_loss(amp_pred, hr_pred, y, hr_weight=hr_weight)
                loss = main + aux_lambda * aux
            loss.backward()
            optimizer.step()
            sum_main += float(main.item())
            sum_aux  += float(aux.item())
            n_batches += 1

        history["epochs"].append(epoch)
        history["main_loss"].append(sum_main / max(n_batches, 1))
        history["aux_loss"].append(sum_aux  / max(n_batches, 1))
        history["n_steps"].append(int(n_batches))
        wall_epoch = time.time() - t_epoch
        print(f"  epoch {epoch:2d}: main={history['main_loss'][-1]:.4f}  "
              f"aux={history['aux_loss'][-1]:.4f}  steps={n_batches}  wall={wall_epoch:.1f}s")

        if on_round_end is not None:
            on_round_end(
                round_idx=epoch,
                model=model,
                server_state_pre=None,
                client_states=None,
                comm_stats={"upload_bytes_round": 0, "broadcast_bytes_round": 0},
                wall_seconds=wall_epoch,
                train_stats={
                    "loss_mean_last_epoch": history["main_loss"][-1],
                    "n_steps_round": int(n_batches),
                },
            )

    elapsed = time.time() - t0
    final_sd = clone_state_dict(model.state_dict())
    return {
        "history": history,
        "final_state_dict": final_sd,
        "n_clients": int(n_clients),
        "elapsed_seconds": float(elapsed),
    }
