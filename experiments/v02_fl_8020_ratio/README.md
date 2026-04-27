# v02 — FL-aligned 80:20 zero-shot evaluation under PFL framing

Successor to `experiments/v01_peak_from_latent/`. Reframes v01's
centralized backbone as a frozen shared encoder (FedHiP pattern), changes
the train:cold split from 50:50 to **80:20**, and adds a routing-mechanism
ablation. Cold inference is **zero-shot** — no client-side training. K-shot
personalization (F2a/F2b/F2c) is in `experiments/v03_kshot_pfl/`.

See `plans/v02-01_fl_8020_ratio.md` for the full motivation and
experimental plan.

## Status

**Implementation + 3-seed sweep complete (2026-04-28).** All G1–G4 numbers in `outputs/v02_fl_8020_ratio/multiseed_summary.json`, all five figures in `papers/v02_draft/figures/`. Paper draft (`papers/v02_draft/v02_fl_8020_ratio.{md,tex}`) is the next step.

All scripts run **per-seed** (`--seed S`); the {42, 123, 7} sweep is driven by a launcher / executor, never hardcoded inside the scripts.

### v02 headline result (3 seeds, R0 routing, mean ± sample std with ddof=1)

| Cell | Cold PAPE (kW) | HR@1 (%) | vs v01 50:50 |
|---|---|---|---|
| baseline (NBEATSxAux uncorrected) | 54.01 ± 1.95 | 27.1 ± 2.26 | v01 baseline 55.17 |
| Hybrid HR-preserving | **43.72 ± 0.59** | 26.6 ± 2.79 | v01 ≈ 45.34 → improved |
| Hybrid PAPE-aggressive | **35.70 ± 0.60** | 26.3 ± 2.63 | v01 37.62 ± 0.45 → improved |

| Goal | result | judgement |
|---|---|---|
| G1 — 80:20 PAPE improvement survives | −19.0% / −33.8% (vs v01 −18% / −32%) | ✅ |
| G2 — R1 vs R0 routing | PAPE indistinguishable; Latent-Route HR@1 +0.5–0.6 pp | ≈ (slight Latent-Route advantage) |
| G3 — multi-seed std | Hybrid PAPE σ 0.42–0.81 (sample std, n=3) | ✅ comparable to v01 §4.4 |
| G4 — Hybrid dominance survives | synergy +3.47 ± 0.40 / +3.24 ± 0.71 PAPE-kW | ✅ Hybrid still dominates |
| E1 — peak_aux on CMO effect | +11.9 ± 11.2 pp (v01: +18.6 pp) | ⚠ mean attenuated, σ very large (Vanilla codebook fragmentation) |
| codebook health (NBEATSxAux) | k_min 137 ± 34 (v01 threshold 113), util 1.000, ppl 27.84 ± 1.14 | ✅ aggregate; ⚠ seed 123 = 98 individually |

## Script order

| # | Script | Purpose | State |
|---|---|---|---|
| 01 | `01_make_split.py` | Generate stratified 80:20 split per seed → `outputs/v02_fl_8020_ratio/splits/v02_8020_seed{S}.yaml` and a refreshed `split_summary.json` (KMeans random_state = `--seed`, KL gate retries with seed+1). | **done** — final KL ≤ 0.33 across 3 seeds; **seed 123 retried** with seed+1 (initial KL > 0.5 → 0.208 after retry); seeds 42 and 7 passed first try. cold-set Jaccard 0.08–0.14. |
| 02 | `02_train_arms.py` | Train T0 (no peak_aux) and T2 (with peak_aux) at 80:20 for one seed. Adapter over v01's `01_train_arms.py`. | **3 seeds done** (T0 / T2 each early-stopped before 30 epochs). |
| 03 | `03_fit_codebook.py` | Fit M=32 KMeans codebook on **T2 latents only** per seed (consumed by 04 and 06; 05 fits its own per-arm codebook on T0/T2 latents independently because the two backbones live in different latent spaces). Saves centroids + per-cluster offsets + KEY pool + scaler params. | **3 seeds done** (k_min 137 ± 34 ≥ v01 threshold 113 in aggregate; seed 123 individually 98). |
| 04 | `04_coldstart_eval.py` | Cold zero-shot inference, **both operating points × {R0 KEY-NN, R1 h_g direct}** — main G1/G2 numbers. | **3 seeds done**. |
| 05 | `05_E1_ablation.py` | **E1: peak_aux ON/OFF on V0 mechanism** — T0 / T2 × V0-only. Mirrors v01 §4.3 exactly so the +18.6 pp headline can be re-tested at 80:20. | **3 seeds done** (+11.9 ± 11.2 pp; per-seed swing 3.6–24.7 driven by Vanilla-side codebook collapse). |
| 06 | `06_W_component_ablation.py` | **W5 component decomposition** — T2 × {V0-only, W1a-only, W5-hybrid} on R0 routing. Asks "in the 80:20 PFL setting, does the W5 hybrid still dominate over each component alone?" New in v02 (v01 §4.6 iter4 only did this on the 50:50 split). | **3 seeds done** (synergy +3.47 ± 0.40 / +3.24 ± 0.71 PAPE-kW; W5 dominates V0/W1a in every seed). |
| 07 | `07_aggregate_seeds.py` | Multi-seed mean ± std summary across 04 / 05 / 06 outputs. | **done** — `multiseed_summary.json`. |
| 08 | `08_make_v02_figures.py` | Generate F1–F5 figures for `papers/v02_draft/figures/`. | **done** — F1–F5 PNGs rendered. |

### What 05 vs 06 isolates

- **05 (E1)** holds the *correction mechanism* fixed (V0 only, the cleanest one — no aux predictions involved) and varies the *backbone* (peak_aux ON ↔ OFF). It is the v01 +18.6 pp PAPE headline check.
- **06 (W component)** holds the *backbone* fixed (T2) and varies the *correction* among {V0-only, W1a-only, W5-hybrid} on the same R0 routing. It asks whether the v01 iter4 ranking (W5 dominates) survives the 80:20 split.

Together they form an orthogonal 2D ablation; merging them would obscure which axis drives any change.

**Cross-script consistency check.** 04 (R0, Hybrid op-point) and 06 (T2 × {V0, W1a, **W5=Hybrid**}, R0) both compute the W5 cell using the same codebook, the same op-point constants, and the same R0 routing. As a sanity check we verified that the W5 PAPE in both 04 R0 and 06 is **bit-identical** (diff = 0.00e+00) across all 3 seeds × 2 op-points = 6 cells. This is what we would expect — both should reduce to the same arithmetic — but it confirms the duplicated helpers in 04 / 06 (which a future refactor will lift to a shared `src/eval/` module) have not silently drifted.

## Key delta from v01

- **Split**: 50:50 → **80:20** (single split, no sweep).
- **Routing**: v01 R0 (5-d KEY-NN) **+ R1** (64-d `h_g_cold` direct nearest-centroid) ablation.
- **Framing**: backbone is treated as a centrally pretrained, frozen
  shared encoder (no client-side weight updates anywhere).

## Naming reference

Pulled together from v01 paper (`papers/v01_draft/v01_peak_vq.md` §3.3, §3.4, §4.1, §4.3, §4.6).

### Backbone arms (`02_train_arms.py`)

| Arm | Model | Loss | peak_aux head |
|---|---|---|---|
| **T0** | `MinimalNBEATSx` | MAE only | — |
| **T2** | `NBEATSxAux(latent_source='h_generic')` | MAE + λ·peak_aux (λ=0.3) | yes (32-d hidden, amp scalar + hr 24-class) |

T1 (h_concat probe of T0) and T3 (h_concat NBEATSxAux) are v01-only — out of v02 scope.

### Correction mechanisms (W5 family)

For an input window `x` with frozen forward `ŷ_base, h_g, (â, ĥ) = T2(x)` and KEY-routed cluster `c*`:

| Symbol | Formula | Role |
|---|---|---|
| **Baseline** | `ŷ_base` (no correction) | reference pure forecast — both T0 and T2 baselines reported in v01 §4.1 / §4.2 |
| **V0** | `ŷ_base + α_v0 · o_{c*}` | cluster-mean offset (V family, post-VQ residual) — **correction**, not baseline |
| **W1a** | `ŷ_base + α_w1 · g(t; ĥ, â, σ)` | Gaussian template only (W family additive) |
| **W5** | `ŷ_base + α_v0 · o_{c*} + α_w1 · g(t; ĥ, â, σ)` | V0 + W1a hybrid — v02 main correction |

Gaussian template: `g(t; ĥ, â, σ) = â · exp(-(t-ĥ)² / 2σ²)`, normalised so `g.max(axis=1) == â`.

`o_{c*}` is the per-cluster residual offset fitted on the train side (`03_fit_codebook.py`).

### Operating points (carried over from v01 §4.2 unchanged)

These are the **two recommended operating points** v01 paper §4.2 selected after iter5's σ × α_v0 × α_w1 × M grid search — *not* the iter4 default (σ=1.5, α=2.0/0.3) hard-coded in `experiments/v01_peak_from_latent/09_iter4_mechanisms.py`. The iter4 defaults were exploratory; the §4.2 final values are what v01's headline cold-PAPE numbers (−18.0 % / −32.8 %) report on.

Both points use σ=3.0; cold-side α-tuning is explicitly out of scope (would re-introduce v01 §5.4.1 selection bias).

| Op-point | σ | α_v0 | α_w1 | Intent |
|---|---|---|---|---|
| **HR-preserving** | 3.0 | 1.0 | 0.1 | gentle correction, keep HR@k near baseline |
| **PAPE-aggressive** | 3.0 | 1.5 | 0.5 | larger correction, push PAPE down at small HR cost |

### Routings (`04_coldstart_eval.py`)

| Routing | Cluster assignment for cold window | Cost over backbone forward |
|---|---|---|
| **R0** | `KEY(x) → StandardScaler → 1-NN on train KEY pool → cluster_idx of that train window` (5-d) | 0 fwd (KEY is input-only) |
| **R1** | `argmin_c ‖h_g_cold − codebook[c]‖₂` direct (64-d) | 0 fwd (h_g already produced for the aux head) |

R0 is v01's routing; R1 is v02's new ablation question (does ×12 latent info change routing decisions?).

## Outputs

`outputs/v02_fl_8020_ratio/` — see plan §"Outputs" for tree.

## What is NOT in scope

- K-shot fine-tuning (any of F2a/b/c) → **v03**.
- Whole-backbone fine-tune (F1) → v04.
- Federated KMeans / DP-KMeans → v04+.
- 2nd dataset → v04+.
- New W-mechanisms — method frozen at v01's W5 hybrid.
- **Other forecasting baselines** (DLinear, NHiTS, Crossformer, Chronos, …) — v01 used NBEATSx-family throughout; v02 holds the model frozen so the comparison stays "v01 50:50 vs v02 80:20" on identical method. Cross-model baselines deferred to v04+.
