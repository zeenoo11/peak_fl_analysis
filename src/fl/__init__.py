"""v04 federated-learning baselines.

Five training routines, one per algorithm. All return the same result-
dict schema so v04's ``07_aggregate.py`` can read them uniformly:

    {
        "algorithm":        str,            # one of {fedavg, fedprox, fedrep, ditto, local_only}
        "config":           dict,           # the FLConfig (or subclass) used
        "history":          dict,           # per-round diagnostic
        "cold_metrics":     dict,           # {pape, hr@1, hr@2, mae, n_cold_windows, n_cold_apts}
        "n_train_clients":  int,
        "final_state_dict": OrderedDict | None,
        ...optional algorithm-specific extras...
    }

Backbone: MinimalNBEATSx (no peak_aux head). v04 method axis "FL
backbone = NBEATSx, correction = none" — Peak-VQ correction is the
v01-v03 contribution and lives in its own row, not on top of these
baselines (G5 cross-cell adds Peak-VQ on top of FedAvg / FedRep
separately).
"""

from fl.base import (
    DEVICE,
    ClientData,
    FLConfig,
    FLHistory,
    apply_state_dict,
    build_clients,
    client_loader,
    clone_state_dict,
    evaluate_clients_val,
    evaluate_cold,
    init_backbone,
    run_local_epochs,
    weighted_average,
)
from fl.ditto import DittoConfig, train_ditto
from fl.fedavg import train_fedavg
from fl.fedproto import FedProtoConfig, train_fedproto
from fl.fedprox import FedProxConfig, train_fedprox
from fl.fedrep import FedRepConfig, train_fedrep
from fl.local_only import LocalOnlyConfig, train_local_only

__all__ = [
    # base
    "DEVICE",
    "ClientData",
    "FLConfig",
    "FLHistory",
    "apply_state_dict",
    "build_clients",
    "client_loader",
    "clone_state_dict",
    "evaluate_clients_val",
    "evaluate_cold",
    "init_backbone",
    "run_local_epochs",
    "weighted_average",
    # algorithms
    "train_fedavg",
    "train_fedprox",  "FedProxConfig",
    "train_fedrep",   "FedRepConfig",
    "train_ditto",    "DittoConfig",
    "train_local_only", "LocalOnlyConfig",
    "train_fedproto", "FedProtoConfig",
]
