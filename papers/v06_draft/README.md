# v06 Draft Manifest

| File | Role |
|---|---|
| `v06_round_dynamics.md` | Main paper — Phase 1 (round-level FL training dynamics) + Phase 2 (post-hoc codebook stacking) |
| `figures/` | Local copies of the figures rendered into the paper |

## Figures referenced in the paper

| Figure | Source | Description |
|---|---|---|
| F1   | `outputs/v06_round_dynamics/figures/F1_round_vs_val_pape.png` | round vs val PAPE — 6 cells × seed bands |
| F1b  | `outputs/v06_round_dynamics/figures/F1b_round_vs_test_pape.png`| round vs test PAPE (Option A trajectory) |
| F1c  | `outputs/v06_round_dynamics/figures/F1c_round_vs_train_loss.png`| round vs train loss |
| F2   | `outputs/v06_round_dynamics/figures/F2_bytes_vs_val_pape.png` | comm cost vs val PAPE Pareto |
| F3   | `outputs/v06_round_dynamics/figures/F3_drift_vs_round.png`    | client drift L2 trajectory |
| F4   | `outputs/v06_round_dynamics/figures/F4_round_vs_test_pape_MAEonly.png`| round vs test PAPE — MAE-only ablation |
| F5   | `outputs/v06_round_dynamics/figures/F5_round_vs_train_loss_MAEonly.png`| round vs train loss — MAE-only ablation |
| **F6** | `outputs/v06_round_dynamics/figures/F6_codebook_lift.png` | **Phase 2 — codebook lift on test PAPE** |
| **F7** | `outputs/v06_round_dynamics/figures/F7_alpha_pareto.png`  | **§5.4 — α_v0 PAPE/MAE Pareto curve** (4 alpha points × 6 cells) |
| **F8** | `outputs/v06_round_dynamics/figures/F8_klocal_sweep.png` | **§5.5 — K_local sweep on FL cells** (diminishing returns past K=2) |

## Numeric anchors

- Phase 1 multi-seed: `outputs/v06_round_dynamics/multiseed_summary.json`
- Phase 2 multi-seed: `outputs/v06_round_dynamics/codebook_lift_summary.json`
- Trajectory tensors: `outputs/v06_round_dynamics/trajectories.npz`

## Reproducibility commands

`experiments/v06_round_dynamics/Readme.md` is the canonical command index. Sections:

- **0501** — Phase 1 default 18 runs (6 cells × 3 seeds, λ_aux=0.3)
- **0502** — Phase 1 MAE-only ablation 18 runs (-MAEonly suffix) + Option A trajectory re-run
- **0503** — Phase 2 codebook stacking 18 runs (6 cells × 3 seeds, α=1.0, K=2)
- **0504 (Phase-2 ablations, 117 runs)** — three follow-up sweeps for §5.3 / §5.4 / §5.5:
    - **§5.3 MAEonly codebook**: 18 runs (`--cell V6-Dyn-{...}-MAEonly`)
    - **§5.4 α_v0 sweep**: 54 runs (`--alpha_v0 {0.5, 1.5, 2.0} --ablation_suffix _alpha{V}`)
    - **§5.5 K_local sweep**: 45 runs FL only (`--K_local {1, 4, 8} --ablation_suffix _K{K}`)
    - Aggregator: ad-hoc inline (per-file json read; no separate aggregator script — α/K
      are summarised directly in the paper §5.4 / §5.5 tables)
    - Figures: `11_make_ablation_figures.py` (F7 + F8)

All drivers take `--seed S` per invocation; multi-seed sweep is the launcher's responsibility.
