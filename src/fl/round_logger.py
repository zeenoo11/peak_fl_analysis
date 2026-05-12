"""Per-round (or per-epoch) logger for v06 round-dynamics experiments.

Writes one JSONL row per round to ``round_log.jsonl`` following the schema in
``plans/v06-01_round_dynamics.md`` §3 ("라운드별 logging contract"):

    {
      "round": 7, "epoch_equivalent": 14.0,
      "val":   {"pape_mean", "pape_std_across_clients",
                "hr@1_mean", "hr@2_mean", "mae_mean", "mse_kw2_mean",
                "n_clients", "n_windows_total"},
      "test":  {... same shape as `val` ...}        # only when test_data attached
      "train": {"loss_mean_last_epoch", "n_steps_round"},
      "comm":  {"upload_bytes_round", "upload_bytes_cum",
                "broadcast_bytes_round", "broadcast_bytes_cum"},
      "drift_l2": ...,
      "wall_seconds_round": ...,
    }

The terminal (test) row is appended with ``round = -1`` (``log_terminal``).

Round-by-round test trajectory (option A)
-----------------------------------------
When ``test_data`` is attached to the logger (the v06 driver always passes it),
``log_round`` also forwards over the test windows on every round and writes a
``test`` block alongside ``val``. This matches the McMahan 2017 FedAvg
Figure 2 / Li 2020 FedProx Figure 6 paper convention of plotting the test
trajectory rather than (or in addition to) the val trajectory. Cost per round
≈ 2× the val forward (test ≈ 20% of the timeline vs val ≈ 10%); only enable
when a full multi-seed re-run is acceptable. When ``test_data is None`` the
``test`` key is omitted (back-compat with older callers / older jsonl rows).

Usage
-----
The logger is constructed once per cell run; the FL helper passes
``logger.log_round`` as ``on_round_end=...``. The model passed to
``log_round`` is forwarded over each apt's val (or test, for the terminal
row) windows; per-apt PAPE / HR@1 / HR@2 / MAE / MSE(kW²) are computed by
denormalising z-space back to kW with the apt's ``(mean, std)``.

across-apt aggregation = simple mean / sample-std (ddof=1) over apts —
each apt counts once (not weighted by window count) so single-apt
"outlier 효과" 가 across-client mean 에 보이지 않게 over-influence 하지
않음. This matches plan §3 ("100가구 평균이 ``*_mean``, across-client
표준편차가 ``*_std_across_clients``").

Resumability
------------
This logger writes one row per round and does not track its own
resume cursor; the Phase 1 drivers run each cell front-to-back and
overwrite the output directory on re-launch.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import HORIZON
from fl.base import DEVICE
from utils.metrics import compute_hr, compute_mae, compute_mse, compute_pape


def _to_kw(z: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Denormalise z-space windows back to kW (per-apt mean/std)."""
    return z * std + mean


@torch.no_grad()
def _forward_apt(
    model: nn.Module,
    x: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """Forward windows through a model that may return a 2-tuple (y_hat, hiddens)
    or a 3-tuple (y_hat, hiddens, aux). Returns y_hat as a numpy array (fp32).
    """
    model.eval()
    n = len(x)
    out: list[np.ndarray] = []
    for i in range(0, n, batch_size):
        xb = torch.from_numpy(x[i : i + batch_size]).to(DEVICE)
        ret = model(xb)
        # Tuple unpack: support both MinimalNBEATSx (y_hat, hiddens) and
        # NBEATSxAux (y_hat, hiddens, (amp_pred, hr_pred)).
        y_hat = ret[0] if isinstance(ret, tuple) else ret
        out.append(y_hat.float().cpu().numpy())
    if not out:
        return np.zeros((0, HORIZON), dtype=np.float32)
    return np.concatenate(out, axis=0)


def _per_client_eval(
    model: nn.Module,
    eval_data: dict[str, dict],
    splits: dict[str, dict],
    batch_size: int = 512,
) -> dict:
    """Run forward on each apt's eval (val/test) windows and compute per-apt
    PAPE / HR@1 / HR@2 / MAE / MSE(kW²); return across-apt mean + std.
    """
    pape_vals: list[float] = []
    hr1_vals:  list[float] = []
    hr2_vals:  list[float] = []
    mae_vals:  list[float] = []
    mse_vals:  list[float] = []
    n_windows_total = 0
    for apt, blk in eval_data.items():
        x = blk["x"]; y = blk["y"]
        if x.shape[0] == 0:
            continue
        sp = splits[apt]
        m_, s_ = float(sp["mean"]), float(sp["std"])
        y_hat_z = _forward_apt(model, x, batch_size=batch_size)
        y_kw = _to_kw(y, m_, s_)
        yh_kw = _to_kw(y_hat_z, m_, s_)
        pape_vals.append(float(compute_pape(y_kw, yh_kw)))
        hr1_vals.append(float(compute_hr(y_kw, yh_kw, tol=1)))
        hr2_vals.append(float(compute_hr(y_kw, yh_kw, tol=2)))
        mae_vals.append(float(compute_mae(y_kw, yh_kw)))
        mse_vals.append(float(compute_mse(y_kw, yh_kw)))
        n_windows_total += int(x.shape[0])

    if not pape_vals:
        return {
            "pape_mean": float("nan"), "pape_std_across_clients": float("nan"),
            "hr@1_mean": float("nan"), "hr@2_mean": float("nan"),
            "mae_mean": float("nan"), "mse_kw2_mean": float("nan"),
            "n_clients": 0, "n_windows_total": 0,
        }
    return {
        "pape_mean":               float(np.mean(pape_vals)),
        "pape_std_across_clients": float(np.std(pape_vals, ddof=1)) if len(pape_vals) > 1 else 0.0,
        "hr@1_mean":               float(np.mean(hr1_vals)),
        "hr@2_mean":               float(np.mean(hr2_vals)),
        "mae_mean":                float(np.mean(mae_vals)),
        "mse_kw2_mean":            float(np.mean(mse_vals)),
        "n_clients":               len(pape_vals),
        "n_windows_total":         n_windows_total,
    }


def _drift_l2(
    server_state_pre: dict[str, torch.Tensor],
    client_states: Optional[Iterable[dict[str, torch.Tensor]]],
) -> float:
    """``mean_i ||θ_i - θ_global_pre||₂`` over float-tensor parameters.

    Returns 0.0 when ``client_states`` is None (e.g. centralised cell —
    "drift" is not defined without per-client snapshots).
    """
    if client_states is None:
        return 0.0
    refs = {k: v for k, v in server_state_pre.items() if v.is_floating_point()}
    if not refs:
        return 0.0
    drifts: list[float] = []
    for cs in client_states:
        sq = 0.0
        for k, v_ref in refs.items():
            v_loc = cs[k]
            sq += float(((v_loc.detach().cpu().to(torch.float64) - v_ref.detach().cpu().to(torch.float64)) ** 2).sum())
        drifts.append(float(np.sqrt(sq)))
    return float(np.mean(drifts)) if drifts else 0.0


class RoundLogger:
    """Round-by-round JSONL logger. One instance per cell run.

    Parameters
    ----------
    log_path  : path to ``round_log.jsonl`` (parent dir is created if needed).
    splits    : ``dict[apt -> per_client_split block]`` containing ``mean``
                and ``std`` for every apt the eval data covers.
    val_data  : ``dict[apt -> {'x': np.ndarray, 'y': np.ndarray}]`` for val
                forwards. Subset of ``splits`` keys.
    test_data : optional ``dict[apt -> {'x', 'y'}]`` for the terminal row.
    batch_size: forward batch size for eval (default 512 to match training).
    """

    def __init__(
        self,
        log_path: Path,
        splits: dict[str, dict],
        val_data: dict[str, dict],
        test_data: Optional[dict[str, dict]] = None,
        batch_size: int = 512,
    ) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.splits = splits
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self._upload_cum = 0
        self._broadcast_cum = 0
        # Open in append mode — caller is responsible for clearing if a
        # full-restart is desired.
        self._fh = self.log_path.open("a", buffering=1)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "RoundLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def log_round(
        self,
        round_idx: int,
        model: nn.Module,
        server_state_pre: Optional[dict[str, torch.Tensor]] = None,
        client_states: Optional[list[dict[str, torch.Tensor]]] = None,
        comm_stats: Optional[dict] = None,
        wall_seconds: float = 0.0,
        train_stats: Optional[dict] = None,
        epoch_equivalent: Optional[float] = None,
    ) -> dict:
        """Run val forward → compute metrics → write one JSONL row.

        Parameters
        ----------
        round_idx        : 1-based round index (or epoch index for centralised).
        model            : current global / centralised model. Will be put in
                           eval() before forward.
        server_state_pre : round-start (= broadcast) global state, used as the
                           reference for ``drift_l2``. Pass None to skip drift.
        client_states    : per-client end-of-local state dicts. Pass None for
                           centralised (drift = 0).
        comm_stats       : dict with ``upload_bytes_round``,
                           ``broadcast_bytes_round``. Defaults to zeros.
        wall_seconds     : wall-clock time of this round (compute + aggregation).
        train_stats      : optional dict {'loss_mean_last_epoch': ..., 'n_steps_round': ...}.
        epoch_equivalent : optional float for centralised-vs-FL alignment.

        Returns the row dict it just wrote (also persists to disk).
        """
        comm = comm_stats or {}
        upload_round = int(comm.get("upload_bytes_round", 0))
        broadcast_round = int(comm.get("broadcast_bytes_round", 0))
        self._upload_cum += upload_round
        self._broadcast_cum += broadcast_round

        val_block = _per_client_eval(model, self.val_data, self.splits, batch_size=self.batch_size)

        if server_state_pre is not None and client_states is not None:
            drift = _drift_l2(server_state_pre, client_states)
        else:
            drift = 0.0

        train_block = train_stats or {"loss_mean_last_epoch": None, "n_steps_round": None}

        row = {
            "round": int(round_idx),
            "epoch_equivalent": float(epoch_equivalent) if epoch_equivalent is not None else float(round_idx),
            "val":   val_block,
            "train": train_block,
            "comm":  {
                "upload_bytes_round":    upload_round,
                "upload_bytes_cum":      int(self._upload_cum),
                "broadcast_bytes_round": broadcast_round,
                "broadcast_bytes_cum":   int(self._broadcast_cum),
            },
            "drift_l2": float(drift),
            "wall_seconds_round": float(wall_seconds),
        }
        # Optional round-level test forward (paper convention: plot test
        # trajectory). Only run when test_data is attached.
        if self.test_data is not None:
            test_block = _per_client_eval(
                model, self.test_data, self.splits, batch_size=self.batch_size
            )
            row["test"] = test_block
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()
        return row

    def log_terminal(
        self,
        model: nn.Module,
        comm_total: Optional[dict] = None,
        wall_total: float = 0.0,
    ) -> dict:
        """Append the terminal eval row with ``round = -1`` (single row, plan §3 spec).

        Evaluates both val and test splits in one call and writes a single JSONL row
        containing both ``"val"`` and ``"test"`` blocks. ``comm_total`` is reported
        with both ``*_round`` and ``*_cum`` populated to the same total.
        """
        comm = comm_total or {}
        upload_total = int(comm.get("upload_bytes_round", self._upload_cum))
        broadcast_total = int(comm.get("broadcast_bytes_round", self._broadcast_cum))

        val_block = _per_client_eval(model, self.val_data, self.splits, batch_size=self.batch_size)
        test_block = (
            _per_client_eval(model, self.test_data, self.splits, batch_size=self.batch_size)
            if self.test_data is not None else None
        )
        row: dict = {
            "round": -1,
            "epoch_equivalent": -1.0,
            "val": val_block,
            "train": {"loss_mean_last_epoch": None, "n_steps_round": None},
            "comm": {
                "upload_bytes_round":    upload_total,
                "upload_bytes_cum":      upload_total,
                "broadcast_bytes_round": broadcast_total,
                "broadcast_bytes_cum":   broadcast_total,
            },
            "drift_l2": 0.0,
            "wall_seconds_round": float(wall_total),
        }
        if test_block is not None:
            row["test"] = test_block
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()
        return row
