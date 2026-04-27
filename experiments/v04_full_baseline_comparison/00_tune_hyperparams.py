"""v04 hyperparameter tuning — single algorithm × single param grid.

Tuning policy (decision 2026-04-28)
-----------------------------------
- Use **train apts' val segment** to score grid points (per-apt z-norm
  using the same train segment used for FL training; identical to v01
  02_train_arms.py's eval_per_apt). The cold split is **never seen**
  during tuning, so v01 §5.4.1's selection-bias concern stays closed.
- **One seed (=42) only.** Grid search runs at one seed; the final 3-seed
  sweep then uses the chosen default everywhere.
- **Score = train-apts val-segment PAPE (kW)**. HR@k and MAE are also
  recorded but the decision rule is "lowest val PAPE wins".
- **Cold metrics also recorded** for parity diagnostic, **but cold is
  not used for the decision**. Cold numbers exist in the JSON only as
  a sanity check.

Per-seed CLI (matches v02/v03/v04 conventions):

    uv run python experiments/v04_full_baseline_comparison/00_tune_hyperparams.py \\
        --algorithm fedprox --grid_param mu --grid 0.001 0.01 0.1 1.0 --seed 42

    uv run python experiments/v04_full_baseline_comparison/00_tune_hyperparams.py \\
        --algorithm fedrep --grid_param head_epochs --grid 1 2 --seed 42

    uv run python experiments/v04_full_baseline_comparison/00_tune_hyperparams.py \\
        --algorithm ditto --grid_param lam --grid 0.01 0.1 0.5 1.0 --seed 42

Output:

    outputs/v04_full_baseline_comparison/tuning/{algorithm}_{grid_param}.json

Local-only excluded from this script (rounds=20 default per user
decision; tuning grid would dominate wall-clock).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import OUTPUT_DIR, RANDOM_SEED
from dataloader.splits import load_v02_split
from fl import (
    DittoConfig,
    FedProxConfig,
    FedRepConfig,
    apply_state_dict,
    build_clients,
    evaluate_clients_val,
    init_backbone,
    train_ditto,
    train_fedprox,
    train_fedrep,
)

V04_OUT_ROOT = OUTPUT_DIR / "v04_full_baseline_comparison"
TUNING_DIR = V04_OUT_ROOT / "tuning"

# Mapping of (algorithm, grid_param) -> (config class, train_fn, valid_param_check)
ALGORITHMS = {
    ("fedprox", "mu"): (FedProxConfig, train_fedprox, float),
    ("fedrep", "head_epochs"): (FedRepConfig, train_fedrep, int),
    ("ditto", "lam"): (DittoConfig, train_ditto, float),
}


def _parse_grid_value(s: str, cast) -> float | int:
    return cast(s)


def main() -> None:
    ap = argparse.ArgumentParser(description="Tune one v04 FL hyperparameter on seed=42 train val.")
    ap.add_argument("--algorithm", required=True, choices=["fedprox", "fedrep", "ditto"])
    ap.add_argument("--grid_param", required=True, help="One of: mu (fedprox), head_epochs (fedrep), lam (ditto).")
    ap.add_argument("--grid", required=True, nargs="+", help="Grid values (string-parsed by the param's cast).")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no_amp", action="store_true", help="Disable bf16 autocast (default = enabled).")
    args = ap.parse_args()

    key = (args.algorithm, args.grid_param)
    if key not in ALGORITHMS:
        raise SystemExit(f"unsupported (algorithm, grid_param): {key}; see ALGORITHMS")
    cfg_cls, train_fn, cast = ALGORITHMS[key]
    grid_values = [_parse_grid_value(s, cast) for s in args.grid]

    sp = load_v02_split(args.seed)
    train_apts, cold_apts = sp["train"], sp["cold"]
    print(f"[tune] seed={args.seed}  train={len(train_apts)}  cold={len(cold_apts)}")
    print(f"[tune] algorithm={args.algorithm}  grid_param={args.grid_param}  grid={grid_values}")
    print(f"[tune] rounds={args.rounds}  local_epochs={args.local_epochs}  batch_size={args.batch_size}  lr={args.lr}")

    # Pre-build the val clients once so each grid point evaluates the *same*
    # apt list and z-norm stats.
    val_clients = build_clients(train_apts)
    print(f"[tune] {len(val_clients)} val clients prebuilt")

    results = []
    for gv in grid_values:
        print(f"\n=== {args.algorithm}.{args.grid_param} = {gv} ===")
        kwargs = dict(
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
            use_amp=not args.no_amp,
        )
        kwargs[args.grid_param] = gv
        cfg = cfg_cls(**kwargs)

        t0 = time.time()
        out = train_fn(train_apts, cold_apts, cfg)
        elapsed = time.time() - t0

        # Re-evaluate on val *clients*. We need a model to load `out["final_state_dict"]` into.
        # init_backbone seed doesn't matter for evaluation; we just need an MinimalNBEATSx instance.
        model = init_backbone(seed=args.seed)
        apply_state_dict(model, out["final_state_dict"])
        val_metrics = evaluate_clients_val(model, val_clients, batch_size=args.batch_size, use_amp=not args.no_amp)

        # Cold parity (already computed inside train_fn, just re-extract).
        cold_metrics = out["cold_metrics"]

        print(f"  elapsed={elapsed:.0f}s  val_PAPE={val_metrics['pape']:.2f}  val_HR@1={val_metrics['hr@1']:.1f}")
        print(f"  cold parity: PAPE={cold_metrics['pape']:.2f}  HR@1={cold_metrics['hr@1']:.1f}")

        results.append({
            "grid_value": gv,
            "config": asdict(cfg),
            "elapsed_seconds": elapsed,
            "val_metrics": val_metrics,        # *the* tuning score
            "cold_metrics_parity": cold_metrics,
            "history_summary": {
                "rounds": out["history"]["rounds"],
                "train_loss": out["history"]["train_loss"],
            },
        })

    # Best grid value: lowest val PAPE.
    best = min(results, key=lambda r: r["val_metrics"]["pape"])
    summary = {
        "algorithm": args.algorithm,
        "grid_param": args.grid_param,
        "grid": grid_values,
        "seed": args.seed,
        "fixed": {
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "use_amp": not args.no_amp,
        },
        "results": results,
        "best": {
            "grid_value": best["grid_value"],
            "val_pape": best["val_metrics"]["pape"],
            "val_hr@1": best["val_metrics"]["hr@1"],
            "cold_pape_parity": best["cold_metrics_parity"]["pape"],
        },
    }

    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TUNING_DIR / f"{args.algorithm}_{args.grid_param}.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[tune] best {args.algorithm}.{args.grid_param} = {best['grid_value']}  (val_PAPE={best['val_metrics']['pape']:.2f})")
    print(f"[tune] saved -> {out_path}")


if __name__ == "__main__":
    main()
