# v02 — FL-aligned 80:20 zero-shot evaluation under PFL framing

Successor to `experiments/v01_peak_from_latent/`. Reframes v01's
centralized backbone as a frozen shared encoder (FedHiP pattern), changes
the train:cold split from 50:50 to **80:20**, and adds a routing-mechanism
ablation. Cold inference is **zero-shot** — no client-side training. K-shot
personalization (F2a/F2b/F2c) is in `experiments/v03_kshot_pfl/`.

See `plans/v02-01_fl_8020_ratio.md` for the full motivation and
experimental plan.

## Status

Scaffolding only — scripts to be added.

## Planned script order

| # | Script | Purpose |
|---|---|---|
| 01 | `01_make_split.py` | Generate stratified 80:20 split, write `outputs/v02_fl_8020_ratio/splits/v02_8020.yaml`. |
| 02 | `02_train_arms.py` | Train T0 (no peak_aux) and T2 (with peak_aux) at 80:20, 3 seeds. Adapter over v01's `01_train_arms.py`. |
| 03 | `03_fit_codebook.py` | Fit M=32 KMeans codebook on T2 latents per seed. |
| 04 | `04_coldstart_eval.py` | Cold zero-shot inference, both operating points × {R0 KEY-NN, R1 h_g direct}. |
| 05 | `05_E1_ablation.py` | peak_aux ON/OFF V0-mechanism ablation at 80:20 (matches v01 §4.3). |
| 06 | `06_aggregate_seeds.py` | Multi-seed mean ± std summary. |
| 07 | `07_make_v02_figures.py` | Generate F1–F4 figures for `papers/figures/`. |

## Key delta from v01

- **Split**: 50:50 → **80:20** (single split, no sweep).
- **Routing**: v01 R0 (5-d KEY-NN) **+ R1** (64-d `h_g_cold` direct nearest-centroid) ablation.
- **Framing**: backbone is treated as a centrally pretrained, frozen
  shared encoder (no client-side weight updates anywhere).

## Outputs

`outputs/v02_fl_8020_ratio/` — see plan §"Outputs" for tree.

## What is NOT in scope

- K-shot fine-tuning (any of F2a/b/c) → **v03**.
- Whole-backbone fine-tune (F1) → v04.
- Federated KMeans / DP-KMeans → v04+.
- 2nd dataset → v04+.
- New W-mechanisms — method frozen at v01's W5 hybrid.
