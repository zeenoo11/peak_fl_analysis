# Paper artifacts — `papers/`

This directory holds paper drafts, figures, and literature references for
the project. Each paper version lives under its own `v{NN}_draft/`
subdirectory; planning docs are in `../plans/v{NN}-{seq}_*.md` and
reproducible code is under `../experiments/v{NN}_*/`.

## Layout

```
papers/
├── README.md            # this file (manifest across versions)
├── literlature/         # shared literature notes (sic, kept as-is)
├── v01_draft/
│   ├── README.md        # v01-specific compile / reproduce guide
│   ├── v01_peak_vq.md   # markdown master draft
│   ├── v01_peak_vq.tex  # IEEE LaTeX
│   └── figures/         # F0..F6 PNGs
├── v02_draft/           # md draft + figures done; tex pending
├── v03_draft/           # placeholder, v03 blocked on v02
├── v04_draft/           # placeholder, v04 final-comparison version (branch: v04)
├── v05_draft/           # FedCB hierarchical federated codebook
├── v06_draft/           # round-level FL training dynamics + Phase 2 codebook
├── v07_draft/           # loss-weight sensitivity sweep (λ_aux, hr_weight)
├── pfl_unified/         # cross-version unified pFL paper (v02/v04/v05)
└── conference_draft/    # KIIE conference final pipeline (FL-only)
```

## Version status

| Version | Title | Plan | Experiments | Outputs | Paper |
|---|---|---|---|---|---|
| **v01** | Peak-Aware VQ for cold-start residential load forecasting (50:50, centralized) | `plans/v01-01_peak_from_latent_test.md` | `experiments/v01_peak_from_latent/` | `outputs/v01_peak_from_latent/` | **complete** — `v01_draft/v01_peak_vq.{md,tex}` |
| **v02** | FL-aligned 80:20 zero-shot under PFL framing | `plans/v02-01_fl_8020_ratio.md` | `experiments/v02_fl_8020_ratio/` | `outputs/v02_fl_8020_ratio/` | **md + figures done** — `v02_draft/v02_fl_8020_ratio.md` (tex pending) |
| **v03** | K-shot personalized FL, F2 ablation (head / last-layer / LoRA) | `plans/v03-01_kshot_pfl.md` | `experiments/v03_kshot_pfl/` | `outputs/v03_kshot_pfl/` | scaffolding (blocked on v02) |
| **v04** | Full baseline comparison: FL × Neural Forecasting × Foundation Models | `plans/v04-01_full_baseline_comparison.md` | `experiments/v04_full_baseline_comparison/` | `outputs/v04_full_baseline_comparison/` | scaffolding (branch: `v04`) |
| **v05** | FedCB — 2-stage hierarchical federated codebook + FedAvg-Aux backbone | `plans/v05-01_fedcb_codebook.md` | `experiments/v05_fedcb_codebook/` | `outputs/v05_fedcb_codebook/` | **complete** — `v05_draft/v05_fedcb.md` |
| **v06** | Round-level FL training dynamics + Phase 2 post-hoc codebook | `plans/v06-01_round_dynamics.md` | `experiments/v06_round_dynamics/` | `outputs/v06_round_dynamics/` | **drafted** — `v06_draft/v06_round_dynamics.md` (8 figures F1–F8) |
| **v07** | Loss-weight sensitivity sweep (λ_aux × hr_weight) on the v06 protocol | `plans/v07-01_loss_and_budget_sweeps.md` | `experiments/v07_loss_budget_sweeps/` | `outputs/v07_loss_budget_sweeps/` | **drafted** — `v07_draft/v07_loss_weight_sensitivity.md` (3 figures: F-aux, F-codebook×λ, F-hr; v07-B and v07-C deferred) |

## Per-version compile / reproduce guides

- v01 — see `v01_draft/README.md`.
- v02 / v03 — drafts not yet written; reproduce via the scripts listed in
  the corresponding `experiments/v{NN}_*/README.md` once results are in.

## Conventions

- Each new version increments the major number when the *evaluation
  protocol or framing* changes (50:50 → 80:20 PFL → K-shot PFL → full
  baseline comparison).
- Method components frozen across v02/v03/v04: NBEATSx + peak_aux head +
  W5 hybrid correction. What changes is split, framing, cold-side
  adaptation, and (in v04) the *baselines* placed alongside.
- All papers use the same operating points carried forward from v01:
  - HR-preserving: σ=3.0, α_v0=1.0, α_w1=0.1
  - PAPE-aggressive: σ=3.0, α_v0=1.5, α_w1=0.5
