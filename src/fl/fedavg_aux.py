"""FedAvg over NBEATSxAux (backbone + aux head federated jointly).

(한글 요약)
v04 09_fix_rerun의 ``02_fedavg_nbeatsx_aux.py``가 처음 도입한 *NBEATSxAux 전체*
(backbone + aux head)를 FedAvg하는 학습 루프를, conference 발표용 파이프라인과
앞으로의 v05+ 작업이 모두 같은 본체를 import해서 쓸 수 있도록 ``src/fl/``로
승격한 모듈이다.

Why is this in src/fl/ now?
---------------------------
v04 09_fix_rerun의 ``02_fedavg_nbeatsx_aux.py`` 헤더 노트는 이 helpers를 일부러
스크립트 내부에 둔 이유를 적어 두었다 — v04의 다른 FL 인프라가 ``MinimalNBEATSx``의
2-tuple forward 만 가정하기 때문에 NBEATSxAux 초기화를 기존 ``init_backbone`` 옆에
나란히 두면 잘못 호출될 위험이 있었다. 그러나 conference 파이프라인 (Phase A
backbone 학습)과 잠재적인 v05+ 후속 실험 모두 이 NBEATSxAux용 FedAvg 본체를
재활용해야 한다. inline 두 번 복사하면 drift가 생기므로 ``src/fl/`` 안의 별도
모듈로 추출했다.

**v04 reproducibility 보호**: 기존 v04 스크립트(``02_fedavg_nbeatsx_aux.py``)는
바꾸지 않는다. v04 결과 reproducibility를 위해 v04 internals는 그대로 local copy
유지. 본 모듈은 *forward use only* — 새로 만들어지는 conference / v05+ 드라이버가
import해서 쓴다.

Public surface
--------------
- ``init_backbone_aux(seed)``     — seeded NBEATSxAux init (latent_source='h_generic').
- ``_local_step_aux(model, ...)`` — single-client local SGD with combined loss
                                    ``L = MAE(ŷ, y) + λ · peak_aux_loss(â, ĥ, y)``.
- ``fedavg_aux_round_loop(...)``  — full FedAvg loop over all train clients;
                                    returns ``{history, final_state_dict, n_train_clients}``.

Style matches ``src/fl/base.py`` — functional helpers, no per-algorithm class.
The combined loss weights (``aux_lambda=0.3``, ``hr_weight=0.1``) are CLAUDE.md
defaults; the caller can override but the v04 09_fix_rerun anchor used these
exact values.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fl.base import (
    ClientData,
    DEVICE,
    apply_state_dict,
    build_clients,
    client_loader,
    clone_state_dict,
    weighted_average,
)
from models.nbeatsx_aux import NBEATSxAux
from models.peak_aux_head import peak_aux_loss


class _NullCtx:
    """Trivial no-op context manager used when AMP is disabled (CPU / --no_amp)."""

    def __enter__(self): return self

    def __exit__(self, *a): return False


def init_backbone_aux(seed: int) -> NBEATSxAux:
    """Seeded init of an NBEATSxAux (latent_source='h_generic') on ``DEVICE``.

    Mirrors ``fl.base.init_backbone`` but for the aux-head variant — kept
    here (rather than added to ``fl.base``) so that ``fl.base.init_backbone``
    keeps its strict MinimalNBEATSx contract for the rest of the v04 FL
    infrastructure.
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
    """Run ``n_epochs`` of local SGD on this client with the combined loss.

    ``L = MAE(y_hat, y) + aux_lambda · peak_aux_loss(amp_pred, hr_pred, y, hr_weight)``

    bf16 autocast is enabled when ``use_amp`` and ``DEVICE.type == 'cuda'``;
    on CPU or with ``use_amp=False`` the loop runs in fp32. Returns a small
    diagnostic dict (``main_loss_mean``, ``aux_loss_mean``, ``n_batches``).
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
    clients: list[ClientData] | None = None,
) -> dict:
    """Phase A — federated training of NBEATSxAux with FedAvg.

    Parameters
    ----------
    train_apts    : list of apt names. ``build_clients`` is called internally
                    to load each apt's series (same protocol as v04 / v05).
                    Ignored when ``clients`` is provided directly.
    rounds        : number of FedAvg rounds.
    local_epochs  : local SGD epochs per round per client.
    lr            : Adam learning rate.
    batch_size    : per-client batch size.
    weight_decay  : Adam weight decay.
    seed          : seed for backbone init + numpy.
    use_amp       : enable bf16 autocast on CUDA (silently falls back on CPU).
    aux_lambda    : combined-loss weight on ``peak_aux_loss`` (CLAUDE.md=0.3).
    hr_weight     : ``peak_aux_loss`` internal CE weight on hour classifier
                    (CLAUDE.md=0.1).
    clients       : optional pre-built ``list[ClientData]`` for tests; when
                    None, ``build_clients(train_apts)`` is called.

    Returns
    -------
    dict with keys ``history`` (per-round diagnostic), ``final_state_dict``
    (CPU OrderedDict, loadable with ``strict=True`` into a fresh NBEATSxAux),
    and ``n_train_clients``. Cold metrics are NOT computed here; the caller
    handles Phase B/C downstream.

    Notes
    -----
    Full participation per round (UMass has only 80 train apts so this is
    cheap). FedAvg weighting = ``client.n_train_windows`` per client.
    """
    if clients is None:
        clients = build_clients(train_apts)
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
        print(
            f"  round {r:2d}: main={history['main_loss'][-1]:.4f}  "
            f"aux={history['aux_loss'][-1]:.4f}  n_clients={len(clients)}"
        )

    return {
        "history": history,
        "final_state_dict": global_state,
        "n_train_clients": len(clients),
    }
