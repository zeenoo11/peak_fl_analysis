# v07 Draft Manifest

| File | Role |
|---|---|
| `v07_loss_weight_sensitivity.md` | Main paper — λ_aux sweep (§3) + codebook×λ ablation (§4) + hr_weight sweep (§5) |
| `figures/` | Local copies of the figures rendered into the paper |

## Figures referenced in the paper

| Figure | Source | Description |
|---|---|---|
| **F-aux** | `outputs/v07_loss_budget_sweeps/figures/F-aux.png` | §3 — λ_aux ∈ {0, 0.05, 0.1, 0.2, 0.3} × 6 algorithms × 3 seeds. Centralised interior optimum at λ=0.1; 5 FL cells monotone increasing |
| **F-codebook×λ** | `outputs/v07_loss_budget_sweeps/figures/F-codebook-vs-lambda.png` | §4 — centralised backbone λ ∈ {0, 0.1, 0.3} × {before, after} federated codebook. Codebook absorbs backbone λ; MAEonly is post-codebook strict optimum |
| **F-hr** | `outputs/v07_loss_budget_sweeps/figures/F-hr.png` | §5 — hr_weight ∈ {0.05, 0.1, 0.5, 1.0} × 6 algorithms × 3 seeds at λ_aux=0.1. Centralised robust; 5 FL cells monotone decreasing (small effect) |

## Numeric anchors

- v07-A multi-seed: `outputs/v07_loss_budget_sweeps/aux_sweep_summary.json` (algo × λ matrix)
- v07-A1 codebook lifts (centralised × λ ∈ {0, 0.1, 0.3}):
    - `outputs/v06_round_dynamics/seed{S}/V6-Dyn-A_centralised{,-MAEonly}/codebook_lift.json` (v06 reuse)
    - `outputs/v07_loss_budget_sweeps/seed{S}/V6-Dyn-A_centralised-aux0.1/codebook_lift.json` (v07-A1 new)
- v07-A2 hr-sweep cells: `outputs/v07_loss_budget_sweeps/seed{S}/V6-Dyn-{...}-aux0.1-hr{V}/result.json`

## Reproducibility commands

`experiments/v07_loss_budget_sweeps/Readme.md` is the canonical command index;
the v07 paper §8 gives the end-to-end pipeline. Sections of v07's command
reference:

- **0503** — v07-A 54 runs: `01_run_aux_sweep.py --seeds 42 123 7 --lambdas 0.05 0.1 0.2` (3-way per-seed parallel ≈ 5h32m).
- **0504a** — v07-A1 codebook stacking on centralised λ=0.1 (3 runs): `08_codebook_stacking.py --cell V6-Dyn-A_centralised-aux0.1 --output_namespace v07_loss_budget_sweeps`. Reuses v06's `λ=0` and `λ=0.3` codebook lifts (already on disk).
- **0504b** — v07-A2 54 runs: `02_run_hr_weight_sweep.py --seeds 42 123 7 --hr_weights 0.05 0.5 1.0` (≈ 5h, 3-way parallel).
- **Aggregate + figures**: `05_aggregate_aux.py --seeds 42 123 7` then `08_make_figures.py --section all`.

## Cross-version references

- v06 paper draft: `papers/v06_draft/v06_round_dynamics.md` — establishes
  the round-level FL protocol, Phase 1 (FL training dynamics) and Phase 2
  (codebook stacking) results, and the `(λ_aux=0 + codebook)` hypothesis
  on FL cells (v06 §5.3). v07 generalises §5.3 to centralised and probes
  the loss-weighting axis.
- v07 plan: `plans/v07-01_loss_and_budget_sweeps.md` — original three-axis
  scoping (v07-A loss-weight, v07-B round budget, v07-C round-trajectory
  codebook). This draft covers v07-A only (loss-weight axis); v07-B and
  v07-C are deferred per §6.3.

## Open follow-ups (deferred)

- **v07-B**: round / local-epoch budget sweep (`E ∈ {1, 2, 5, 10, 20}` at
  fixed `T=80`, FedSGD reference). Needs new driver
  `03_fedsgd.py`. Estimated 30–50 h sequential.
- **v07-C**: round-trajectory codebook (Phase 1 with `--checkpoint_every 5`,
  re-stack codebook on rounds {5, 10, 15, 20}). Needs `src/fl/round_logger.py`
  modification. Estimated 6–24 h.

## Tests

`tests/test_v07_aux_sweep.py` (10 cases) covers cell-name suffix mapping
across `_aux_suffix` × `_hr_suffix`, the `--output_namespace` argparse
contract on the v06 drivers, and the v07 aggregator's cell-name parser.
Full repository regression: `pytest tests/` → 42 passed (no v06 regressions).
