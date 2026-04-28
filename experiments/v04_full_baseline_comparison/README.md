# v04 — Full baseline comparison: FL × Neural Forecasting × Foundation Models

> Successor to `experiments/v03_kshot_pfl/` (v03 personalization). v04 is the
> *closing* version of the project: it places Peak-Aware VQ on the same axes
> as established FL baselines, non-NBEATSx neural forecasting (NF) baselines,
> and zero-shot foundation models (FM).

## Status

**Implementation + 3-seed sweep complete (2026-04-28).** All 13 methods × 3 seeds = 39 cells finished, plus G6 heterogeneity (one-shot on the 80 train apts) and G7 communication accounting (one-shot from parameter counts). Five figures rendered to `papers/v04_draft/figures/`. Paper draft (`papers/v04_draft/v04_full_baseline_comparison.{md,tex}`) is the next step.

All training scripts run **per-seed** (`--seed S`); the {42, 123, 7} sweep is dispatched in parallel via background tasks.

### v04 headline (3 seeds, sorted by cold PAPE)

| Rank | Method | Cold PAPE (kW) | HR@1 (%) |
|---:|---|---|---|
| 1 | **peakvq_on_fedrep** (G5) | **47.50 ± 1.36** | 23.5 |
| 2 | **peakvq_on_fedavg** (G5) | **48.26 ± 3.74** | 23.9 |
| 3 | NF DLinear | 50.37 ± 0.84 | 26.4 |
| 4 | NF Crossformer | 52.54 ± 1.71 | 26.9 |
| 5 | Local-only NBEATSx | 52.64 ± 2.44 | **28.5** |
| 6 | FM Chronos-Bolt small | 52.69 ± 1.56 | 26.2 |
| 7 | NF NHITS | 52.99 ± 1.64 | 27.1 |
| 8 | FM TimesFM | 54.27 ± 2.15 | 25.0 |
| 9 | FedProx | 56.30 ± 1.55 | 26.0 |
| 10 | FedAvg | 56.34 ± 1.41 | 26.4 |
| 11 | Ditto | 56.38 ± 1.63 | 26.5 |
| 12 | FedRep | 57.18 ± 1.52 | 25.7 |
| 13 | FM Chronos-T5 tiny | 63.13 ± 3.04 | 18.3 |
| **ref** | **v02 method (NBEATSxAux + Peak-VQ + W5, PAPE-aggressive)** | **35.70 ± 0.49** | 26.3 |

**v01-v03 method (last row) outperforms every v04 baseline by 11.8 kW.** G5 cross-cell (Peak-VQ on FL backbones) is the second-best block — Peak-VQ contributes ≈ 8-10 kW PAPE reduction even on FL-trained backbones.

| Goal | Result | Judgement |
|---|---|---|
| G1 — FL baselines vs v02 | All 5 FL baselines underperform v02 method (52.6 → 57.2 vs v02's 35.7) | ✅ FL gap quantified |
| G2 — NF baselines vs v02 | Best NF (DLinear 50.4) still 14.7 kW worse than v02 method | ✅ NF gap quantified |
| G3 — FM zero-shot lower bound | Chronos-Bolt 52.7 / TimesFM 54.3 / Chronos-T5 63.1 — all worse than v02 (35.7) | ✅ FM is not enough |
| G4 — Pareto placement | v02 method dominates the cold-PAPE axis; Local-only dominates HR@1 (overfit upper bound, see §Limitations) | ✅ |
| G5 — Peak-VQ on FL backbones | Δ −2.9 to −11.4 PAPE-kW vs raw FL backbone, all 6 cells (3 seeds × 2 FL backbones) positive | ✅ Peak-VQ is complementary |
| G6 — Heterogeneity | Pairwise W1 mean 0.379 / max 1.439, cos hour-profile min 0.811 / mean 0.970 (80 train apts) | ✅ heatmap; per-apt correlation skipped (train ≠ cold apt set) |
| G7 — Communication cost | peak_vq 4.94 MB × 1 cross vs FedAvg 420 MB × 20 crosses → **85× less data, 20× fewer boundary crosses** | ✅ headline efficiency claim |

## Scope (decided so far)

### FL baselines

**Tier 1 (mandatory).** These three give the v04 paper its core FL axis:
a "no FL" lower bound, canonical FL averaging, and the personalization
baseline most directly comparable to v03's F2a.

| Baseline | Pattern | What is shared | Personalization |
|---|---|---|---|
| **FedAvg** | full-model averaging (McMahan'17) | full backbone weights | none (single global model) |
| **FedRep** | head-only personalization (Collins ICML'21) | shared encoder | per-client head |
| **Local-only NBEATSx** | no FL | nothing | each cold gucha trains its own backbone independently |

**Tier 2 (recommended, add if scope allows).**

| Baseline | Pattern | Relationship to v01–v03 |
|---|---|---|
| **FedProx** (Li et al., MLSys'20) | FedAvg + proximal term for non-IID stability | tests whether the non-IID gap (per-apt z-norm + per-apt distribution) helps Peak-Aware VQ |
| **Ditto** (Li et al., ICML'21) | global + per-client local with regularization between them | a stronger personalization point than FedRep — directly competes with v03 F2c (LoRA) |

**FedHiP excluded from v04 baselines** (decision 2026-04-28). The v04
plan originally listed FedHiP (arxiv:2508.04470) as a Tier 2 baseline
under the framing "frozen foundation pretrain + cold-side head only".
Reading the FedHiP paper itself shows the algorithm is built around
**closed-form analytic solutions (gradient-free)** — fundamentally
different from v01-v03's gradient-based NBEATSx training. Therefore:

- v04 paper §results does not include a "FedHiP" row, to avoid
  misrepresenting the FedHiP-paper algorithm.
- The "v01-v03 already implicitly adopts FedHiP-style framing" claim
  in v02 §2.2 / §5.1 remains a paper-level claim and is flagged as
  **needing a separate analysis** (not a baseline cell). That analysis
  is out of v04's coding scope but tracked as a follow-up writing
  task in `papers/v04_draft/` — the user's papers rest on this
  framing, so the eventual analytical comparison of "v01-v03 method
  vs FedHiP-paper algorithm" is owed.

### Non-NBEATSx neural-forecasting baselines (TBD — brainstorming)

Candidate set to discuss:
- DLinear (Zeng et al., AAAI'23)
- NHiTS (Challu et al., AAAI'23)
- Crossformer (Zhang & Yan, ICLR'23)

(Note: these were already trained in `Peak_Analysis/v10_b1/b3/b4` on the
50:50 split; v04 needs to re-train them on the v02 80:20 split for
fair comparison, **not** re-use the v10 checkpoints.)

### Foundation-model baselines (TBD — brainstorming)

Candidate set to discuss:
- Chronos (Ansari et al., 2024) — pre-trained, zero-shot inference
- TimesFM (Das et al., 2024) — pre-trained, zero-shot inference
- (optional) Lag-Llama / Moirai / Time-MoE

The FM axis answers "is the v01/v02 method actually better than just
asking a foundation model with no UMass-specific training?" — a question
v01/v02/v03 explicitly deferred.

### Motivation and efficiency analysis

Two analyses orthogonal to the model-comparison axes above. Both
strengthen the PFL framing without adding a model arm — they are
bookkeeping over the baselines already trained in this version, plus
one heterogeneity computation on the train data.

| Analysis | What it answers | Where it shows up |
|---|---|---|
| **Heterogeneity quantification** | Pairwise Wasserstein-1 / KL / peak-shape similarity on train households, with correlation against the local-only-vs-shared gap. Defends "personalization is needed" empirically — currently framing-only in v02 §5.1. | Paper §motivation; figure (heatmap + correlation plot) |
| **Communication-cost accounting** | Bytes-per-round and total bytes for v02's 1-shot codebook vs FedAvg / FedRep / FedProx / Ditto. Quantifies v02's "1 boundary cross" efficiency claim relative to iterative FL. | Paper §results; table |

These close PFL design axes that v02 currently asserts only in framing:
heterogeneity (why personalization is needed) and the communication
subaxis (how much federation costs).

## Decisions taken (closed)

1. **Common evaluation pool.** v04 reuses the v02 80:20 split + 3 seeds
   {42, 123, 7}.
2. **Metrics.** Cold PAPE (kW), HR@1, HR@2, MAE per cell, plus G7
   communication accounting (bytes / boundary crosses).
3. **FL training protocol.** Federated round simulation, full
   participation per round (`clients_per_round=0`), default
   `rounds=20, local_epochs=2, lr=1e-3, batch_size=512` (bf16 autocast).
   FL convergence verified on seed=42 — all four FL algorithms hit
   train-loss saturation by round ≈17 (max delta among last 3 rounds <
   1e-3). Tuning grid (FedProx mu, FedRep head_eps, Ditto lam) was
   skipped per user decision (defaults adopted).
4. **FM zero-shot input.** L=96, H=24 (same as v01–v03). No per-apt
   z-norm — Chronos and TimesFM run their own internal scaling.
5. **G5 cross-cell.** Peak-VQ + W5 Hybrid on top of FedAvg and FedRep
   backbones. Self-derived `(â, ĥ)` from the FL forecast (FL backbone
   has no peak_aux head; v01 §4.3 E1 uses the same construction for
   the T0 row).
6. **G7 communication scope.** 5 FL baselines + 1-shot codebook upload
   of v01-v03. Adaptation-time bytes row deferred until v03 results
   land.
7. **FedHiP excluded.** See note in §FL baselines (paper algorithm =
   closed-form analytic, fundamentally different from gradient-based
   v01-v03 training; comparing as a peer baseline would misrepresent).

## Limitations (disclosed in paper §6)

- **Local-only "self-train + self-eval".** v04's `evaluate_cold` follows
  v01-v03's protocol: forward the cold apt's first 70% (train segment)
  with warm-start z-norm. For all FL/NF/FM baselines this is *unseen*
  data — the cold apt was never a client. **For Local-only this same
  segment is the training data**, so its result is an *overfit upper
  bound* on its own data, not a fair "no-FL lower bound". Local-only's
  HR@1 = 28.5 (highest in the table) is therefore not a generalisation
  result. The paper discusses this and treats Local-only as a sanity
  point rather than a competing baseline.
- **G6 correlation block skipped.** The G6 heterogeneity heatmap is
  computed over the 80 *train* apts; the per-apt Local-only PAPE is
  computed over the 20 *cold* apts. The two sets do not overlap, so
  the train-side heterogeneity vs cold-side gap correlation is not
  computable from these data.
- **FL hyperparameter saturation.** Train-loss flattens by round 17;
  rounds=30 / 40 / increasing local_epochs would not change the cold
  PAPE numbers. The four FL algorithms cluster within 1 kW of each
  other (56.30 – 57.18) — this clustering is itself a v04 finding
  (FL-algorithm choice does not move the cold-side number much) rather
  than a tuning gap.
- **Concurrent dispatch GPU contention.** Wall-clock per task on the
  GTX 5070 Ti varied with concurrency: FedAvg s42 (low contention) =
  6.4 min, FedAvg s123 / s7 (high contention) = 15.5–16.9 min. Final
  metrics are bit-identical regardless — only wall-clock is affected.

## Script order

| # | Script | Purpose | State |
|---|---|---|---|
| 00 | `00_tune_hyperparams.py` | per-algorithm grid sweep on train val (cold not seen). | committed for future use; **not run** in v04 ship (defaults adopted per user decision). |
| 01 | `01_fl_train.py` | one seed × one FL algorithm. Writes `result.json` + `final_state_dict.pt`. GPU snapshot at start/end. | **5 algorithms × 3 seeds = 15 cells done**. |
| 02 | `02_nf_train.py` | one seed × one NF model (DLinear / NHITS / Crossformer), centralised pooled training. | **3 × 3 = 9 cells done**. |
| 03 | `03_fm_zero_shot.py` | one seed × one FM model (Chronos-Bolt / Chronos-T5 / TimesFM), no UMass training. | **3 × 3 = 9 cells done**. |
| 04 | `04_peakvq_on_fl.py` | G5: Peak-VQ + W5 Hybrid on top of FedAvg / FedRep backbone. Self-derived (â, ĥ). | **2 × 3 = 6 cells done**. |
| 05 | `05_heterogeneity.py` | G6, seed-independent (one-shot on the 80 train apts). | **done** — heatmap saved. |
| 06 | `06_communication.py` | G7, seed-independent (one-shot from parameter counts + protocol). | **done**. |
| 07 | `07_aggregate.py` | multi-seed aggregator → `multiseed_summary.json`. | **done** (PAPE-aggressive op-point used for G5 cells). |
| 08 | `08_make_v04_figures.py` | F1 Pareto, F2 G5 cross-cell, F3 sorted bar, F4 heterogeneity, F5 communication. | **done** — 5 PNGs in `papers/v04_draft/figures/`. |

## Deliverables

- `outputs/v04_full_baseline_comparison/seed{42,123,7}/{method}/result.json`
  (39 cells) + `multiseed_summary.json` (across-seed aggregate).
- `outputs/v04_full_baseline_comparison/heterogeneity_summary.json` (G6).
- `outputs/v04_full_baseline_comparison/communication_summary.json` (G7).
- `papers/v04_draft/figures/v04_F{1..5}.png` (5 figures).
- `papers/v04_draft/v04_full_baseline_comparison.{md,tex}` — paper draft (next step).

## Dependencies on prior versions

- v02 80:20 split + cold pool (`outputs/v02_fl_8020_ratio/splits/`).
- v02 frozen backbone + codebook (used as the "ours" entry on the
  comparison rows).
- v03 K-shot results (used as the "ours, personalized" entry on rows
  where personalization is meaningful).

## What is NOT in scope

- New v01-method redesign (method frozen at v01's NBEATSx + W5 hybrid +
  post-hoc Peak-VQ).
- A third dataset — v04 keeps UMass; cross-dataset generalization is
  v05+.
