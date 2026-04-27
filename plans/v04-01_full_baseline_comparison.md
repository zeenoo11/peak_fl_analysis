# v04-01 — Full baseline comparison: FL × Neural Forecasting × Foundation Models

> Successor to `v03-01_kshot_pfl.md`. v04 is the **closing** version of
> the project: place Peak-Aware VQ on the same axes as established FL
> baselines, non-NBEATSx neural forecasting (NF) baselines, and zero-shot
> foundation models (FM). Method is **frozen** at v01's design — v04 is
> a comparison study, not a new method.

> **Status (2026-04-28).** Scaffolding only. FL baselines decided by
> user (Tier 1 + Tier 2 below). NF / FM lists pending brainstorming.

## Motivation

v01–v03 disclosed three honest limitations that v04 closes:

1. **No FL baseline comparison** (v01 §1.3, v02 §6) — the v02 PFL
   framing argues that the post-hoc Peak-VQ codebook + frozen backbone is
   defensible, but it has never been evaluated against actual FL training
   schemes (FedAvg, FedRep, FedProx, Ditto, FedHiP).
2. **No non-NBEATSx neural forecasting baseline at the v02 80:20 split**
   (v02 §6) — v01 reused NBEATSx throughout; whether DLinear / NHiTS /
   Crossformer would have outperformed it on cold gucha is unknown for
   the v02 protocol.
3. **No foundation-model lower bound** (v01 §6, v02 §6) — the v02
   numbers are fair only against models *trained on UMass*; whether a
   zero-shot Chronos / TimesFM call would already match or beat them is
   the obvious external question.

v04 answers all three on the **same** v02 80:20 split + 3-seed protocol
so every comparison is direct.

## Goals

**G1.** Reproduce the v02 main result with FL baselines on the same
80:20 split:
- **Tier 1**: FedAvg, FedRep, Local-only NBEATSx.
- **Tier 2 (if scope allows)**: FedProx, Ditto, FedHiP.

**G2.** Reproduce the v02 main result with non-NBEATSx NF baselines
trained on the v02 80:20 split (not the v10 50:50 checkpoints).

**G3.** Add zero-shot FM baselines on cold gucha — same input window,
no UMass training.

**G4.** Place v01/v02/v03 method (frozen NBEATSxAux + post-hoc Peak-VQ +
W5 hybrid) on the resulting Pareto frontier and quantify the cold-PAPE
delta over each baseline group.

**G5.** Optional cross-cell: evaluate "Peak-VQ on top of FedAvg" /
"Peak-VQ on top of FedRep" / "Peak-VQ on top of FedHiP" — show that
Peak-VQ is *complementary* to FL training schemes rather than a
replacement.

**G6.** **Heterogeneity quantification** (motivation analysis, no new
model arm). Compute pairwise Wasserstein-1 / KL / peak-shape similarity
across the 80 train households and correlate the resulting heterogeneity
with the local-only-vs-shared cold-PAPE gap. Defends the
"personalization is needed" claim that v02 §5.1 currently asserts only
on framing grounds. Seed-independent (computed once from train data, no
model dependence).

**G7.** **Communication-cost accounting** (efficiency analysis, no new
model arm). Bytes-per-round and total bytes for v02's 1-shot codebook
upload vs. each FL baseline (FedAvg / FedRep / FedProx / Ditto /
FedHiP). Quantifies v02's "1 boundary cross" efficiency claim against
iterative FL. Seed-independent (depends on parameter counts and
protocol, not on the random seed).

## FL baselines

**Common protocol (all FL baselines).** 80 train apts are clients;
20 cold apts are **held-out** and never enter FL training (decision 1,
matches v01–v03 cold-start framing). FL training uses the apt's own
train segment (first 70% of its series) as the local dataset. Cold
inference is identical to v02: warm-start z-norm + stride=24 + frozen
forward.

### Tier 1 (mandatory)

| Baseline | Algorithm | What is federated | Personalization | Cold-side data used |
|---|---|---|---|---|
| **FedAvg** | McMahan et al., AISTATS'17 | full backbone weights, vanilla averaging | none — single global model | none (forward only) |
| **FedRep** | Collins et al., ICML'21 | shared encoder; per-client personalized head | per-client head, encoder frozen on cold | none in v04 main row (cold = held-out, no per-client head trained for cold). Optional sub-row: cold-side K-shot head adaptation = v03 F2a. |
| **Local-only NBEATSx** | — (no FL) | nothing | each cold gucha trains its own NBEATSx from scratch | **YES — cold apt's own train segment (first 70%, ≈9 months)** |

**Local-only is the only baseline where the cold apt's own data is used
for training** (decision 2). It exists to provide the "no-FL lower
bound": *what cold-PAPE can a cold apt achieve using only its own
data?* A FL or centralised method that beats Local-only justifies the
cost of FL.

### Tier 2 (recommended)

| Baseline | Algorithm | Cold-side data used | Relationship to v01–v03 |
|---|---|---|---|
| **FedProx** | Li et al., MLSys'20 | none (forward only) | FedAvg + proximal term ‖w_local − w_global‖²; addresses non-IID stability — UMass per-apt is non-IID, so this directly tests the "FedAvg might be hurting us" hypothesis |
| **Ditto** | Li et al., ICML'21 | none in v04 main row; optional cold-side personal model = v03 F2c LoRA analogue | global + per-client local, regularised toward global; stronger personalisation than FedRep |
| **FedHiP** | arxiv:2508.04470 | none (forward only) | frozen foundation pretrain + cold-side head adaptation; **v02 §2.2 already implicitly adopts this framing** — making it explicit lets v04 state "v01–v03 method = FedHiP + Peak-VQ" cleanly |

The FedHiP row is especially valuable: it isolates the contribution of
*Peak-VQ + W5 hybrid correction* over the underlying FL pattern v02
already inherits.

## NF baselines

Three non-NBEATSx neural-forecasting baselines, all trained centralised
(pooled) on the 80 train apts and evaluated on the 20 cold apts:

| Model | Reference | Architecture sketch |
|---|---|---|
| **DLinear** | Zeng et al., AAAI'23 | Trend/Seasonal decomposition + per-channel linear projection. Very small (~50 lines). |
| **NHiTS** | Challu et al., AAAI'23 | 3-stack multi-rate sampling (similar 3-stack structure to NBEATSx). |
| **Crossformer** | Zhang & Yan, ICLR'23 | DSW (dimension-segment-wise) embedding + cross-time-cross-variate attention + HED decoder. |

**Implementation decision (decision 3): re-implement directly in
`src/models/{dlinear,nhits,crossformer}.py`** rather than porting from
`Peak_Analysis/experiments/federated/v10_b{0,1,3,4}_*.py`. Reasons:

1. The Peak_Analysis scripts have inline model classes mixed with
   training loops, mlflow logging, and v10 50:50 split assumptions —
   porting cleanly is comparable in effort to writing from scratch.
2. Direct implementation against the published architectures avoids
   any silent v10-only assumption being inherited.
3. Each model class is self-contained and verifiable against its
   reference paper.

The NF baselines therefore go through the same v02 80:20 split + 3
seeds + per-apt z-norm + L=96 / H=24 / stride=24 protocol as everything
else.

## FM baselines (TBD — brainstorming)

Candidate set:
- **Chronos** (Ansari et al., 2024) — pretrained, zero-shot inference
- **TimesFM** (Das et al., 2024) — pretrained, zero-shot inference
- (optional) **Lag-Llama** / **Moirai** / **Time-MoE**

All run zero-shot on the cold gucha test segment with L=96, H=24
(same as v01–v03). No UMass training. Forecasts are denormalized to
kW and scored on the same PAPE / HR@1 / HR@2.

Open question: at least one each from the *autoregressive token*
family (Chronos) and the *patch-based regressor* family (TimesFM) seems
the minimum useful sweep.

## Motivation and efficiency analysis (G6 + G7)

Two analyses orthogonal to the model-comparison axes (G1–G5). Both
strengthen the PFL framing without adding a model arm — they are
bookkeeping over the baselines already trained in this version, plus
one heterogeneity computation on the train data.

| Analysis | Question answered | Inputs | Output |
|---|---|---|---|
| **G6 Heterogeneity** | Are the 80 train households heterogeneous enough that personalization is *needed*? Is heterogeneity correlated with the (Local-only minus Shared) cold-PAPE gap? | per-apt time series statistics (Wasserstein-1, KL, peak-shape similarity over train segments); cold-PAPE per-cell from §G1/G2 | heatmap (apt × apt similarity); scatter of heterogeneity quartile vs. local-only-vs-shared gap |
| **G7 Communication cost** | Bytes-per-round and total bytes for v02's 1-shot codebook upload vs. iterative FL (FedAvg / FedRep / FedProx / Ditto / FedHiP). | parameter counts of each FL backbone; FL simulation protocol (rounds × local_steps); v02 codebook size (32 × 64 + offsets + KEY pool) | comparison table (rows: methods; cols: bytes/round, n_rounds, total bytes, boundary crosses) |

Both are **seed-independent** — G6 depends on train-data statistics
only, G7 depends on parameter counts and the FL simulation protocol.
Neither requires re-running the 3-seed sweep, so they sit outside the
per-seed pipeline below and are computed once.

## Method axes (orthogonal)

```
                         Tier 1 FL    Tier 2 FL    NF       FM       v01–v03 method
                         ----------   ----------   ------   ------   --------------
trains on UMass?         yes          yes          yes      no       yes (centralised pretrain)
federated?               yes          yes          no       no       no (codebook is post-hoc)
backbone family          NBEATSx*     NBEATSx*     varies   varies   NBEATSx (peak-aware)
correction               none         none         none     none     CMO + GST + Hybrid
```

\* same NBEATSx backbone as v01–v03 to isolate the "what is federated"
axis cleanly.

## Experimental plan

### Splits and seeds

Reuse v02's 80:20 split YAMLs unchanged
(`outputs/v02_fl_8020_ratio/splits/v02_8020_seed{42,123,7}.yaml`). No new
stratification.

### Per-baseline pipeline (per seed)

For each FL baseline:
1. Train per the FL algorithm's protocol on the 80 train apts. (Local
   FL simulation rounds; user decides round/local-step counts during
   brainstorming.)
2. Cold inference identical to v02: warm-start z-norm, stride=24,
   forward through final model, denormalise to kW.
3. Save `outputs/v04_full_baseline_comparison/seed{S}/{baseline}/...`.

For each NF baseline: same as Tier 1 FL except training is centralised
on the 80 pooled train apts (the FL framing does not apply to NF
comparisons themselves).

For each FM baseline: skip training; run zero-shot on cold apts
directly.

### Cross-cell (G5)

For FedAvg / FedRep / FedHiP, run the same pipeline twice:
- (a) raw forecast (no Peak-VQ correction) — the published FL baseline
- (b) raw forecast + Peak-VQ codebook + W5 Hybrid correction (the v01
  recipe, on top of the FL backbone)

Compare (b)−(a) on each FL baseline → "Peak-VQ adds X kW of cold-PAPE
reduction independent of which FL pattern produced the backbone".

## Outputs (target paths)

```
outputs/v04_full_baseline_comparison/
├── seed{42,123,7}/
│   ├── fedavg/best.pt + cold_metrics.json
│   ├── fedrep/best.pt + cold_metrics.json
│   ├── local_only/{apt_i}/best.pt + cold_metrics.json
│   ├── fedprox/best.pt + cold_metrics.json
│   ├── ditto/best.pt + cold_metrics.json
│   ├── fedhip/best.pt + cold_metrics.json
│   ├── nf_dlinear/best.pt + cold_metrics.json
│   ├── nf_nhits/best.pt + cold_metrics.json
│   ├── nf_crossformer/best.pt + cold_metrics.json
│   ├── fm_chronos/cold_metrics.json
│   ├── fm_timesfm/cold_metrics.json
│   └── peakvq_on_{fedavg,fedrep,fedhip}/cold_metrics.json    # G5 cross-cell
├── heterogeneity_summary.json                                # G6 (seed-independent)
├── communication_summary.json                                # G7 (seed-independent)
├── multiseed_summary.json
└── FINAL_v04_report.md
```

## Deliverables

1. **`papers/v04_draft/v04_full_baseline_comparison.md`** + IEEE `.tex`.
2. **Pareto plots**: `papers/v04_draft/figures/v04_F*.png`
   - F1: PAPE × HR@1 Pareto across all baselines + ours (G1–G4).
   - F2: G5 cross-cell — Peak-VQ delta on FedAvg / FedRep / FedHiP.
   - F3: NF / FM zero-shot vs trained comparison.
   - F4: **G6 heterogeneity heatmap** + scatter of heterogeneity vs.
     (Local-only − Shared) cold-PAPE gap — appears in paper §motivation.
3. **Communication cost table** (G7) — embedded in paper §results
   alongside the cold-PAPE comparison; backed by
   `outputs/v04_full_baseline_comparison/communication_summary.json`.

## Open questions

### Closed (2026-04-28)

1. **Cold apts in FL training?** *Held-out* (decision 1). 80 train apts
   are FL clients; 20 cold apts are inference-only. Matches v01–v03
   cold-start framing.
2. **Local-only NBEATSx data scope.** *Full self-train* (decision 2).
   Each cold gucha trains its own NBEATSx on its own first 70% (≈9
   months). Provides the "no-FL lower bound" — comparable in data
   volume to what a FL/centralised method sees from each client, just
   without inter-client knowledge sharing.
3. **NF model implementation.** *Direct re-implementation* in
   `src/models/{dlinear,nhits,crossformer}.py` (decision 3). Peak_Analysis
   v10_b{0,1,3,4} **not** ported.
4. **NF baseline list.** All three of DLinear, NHiTS, Crossformer
   (open question previously, now fixed).

### Still open (settle during step 3 of "Detailed plan" below)

A. **FM baseline list.** Minimum Chronos + TimesFM. Add Lag-Llama /
   Moirai / Time-MoE if compute allows. (One model per FM family is
   the v04-ship target; extras → v05.)

B. **FL simulation protocol.** Number of communication rounds, local
   epochs, client-batch size — settle on a single (rounds, local_steps)
   pair shared by all FL baselines for clean comparability. Default
   to FedAvg literature norm (rounds=20, local_epochs=2, batch=256)
   unless dry-run shows convergence issues.

C. **G5 (cross-cell) scope.** All three of FedAvg / FedRep / FedHiP, or
   only FedHiP (cleanest narrative match to v02/v03)? Default to all
   three for completeness; demote to FedHiP-only if scope tightens.

D. **Tier 2 inclusion gate.** FedProx / Ditto / FedHiP add ~3× FL
   work. Default to "ship Tier 1 first, Tier 2 added once Tier 1
   results are stable across 3 seeds".

E. **Communication-cost measurement scope (G7).** Bytes-per-round +
   total bytes for each FL baseline + the 1-shot codebook upload of
   v02. If v03 K-shot results are available by paper deadline, an
   adaptation-time bytes row can be added — by design v03 is fully
   local (0 upload bytes), which would reinforce the "1 boundary
   cross" framing. This row is **conditional** on v03 results and not
   a hard dependency for v04 ship.

## Detailed plan (build order)

| Step | Module | Purpose | Tests / verify |
|---|---|---|---|
| **0** | `src/eval/cold_helpers.py` (new) | Extract `gather_cold`, `gauss_template`, `metrics_z_to_kw`, `route_R0`, `route_R1`, `OPERATING_POINTS` from the duplicated copies in v02 04/05/06. Refactor v02 04/05/06 to import from here. | Re-run v02 04/05/06 after refactor; results must be **bit-identical** to current `outputs/v02_fl_8020_ratio/...`. |
| **1** | `src/models/dlinear.py`, `nhits.py`, `crossformer.py` (new) | Direct re-implementation of three NF architectures against their reference papers. | Per-model 1-window forward smoke test in pytest. |
| **2** | `src/fm/chronos.py`, `timesfm.py` (new) | Zero-shot wrapper exposing the project-uniform interface (`forecast(x_window) -> y_kw[24]`, where `x_window` is denormalised cold input). | One real forecast per FM on a cold apt. |
| **3** | `src/fl/__init__.py` + per-algorithm files (FedAvg / FedRep / FedProx / Ditto / FedHiP / Local-only) | Self-contained FL simulator: client = apt, round = pooled local steps + parameter aggregation. | Convergence dry-run on seed=42 with FedAvg. |
| **4** | `experiments/v04_full_baseline_comparison/` scripts | `01_fl_train.py`, `02_nf_train.py`, `03_fm_zero_shot.py`, `04_peakvq_on_fl.py` (G5), `05_heterogeneity.py` (G6, **seed-independent**, computed once), `06_communication.py` (G7, **seed-independent**, computed once), `07_aggregate.py`, `08_make_v04_figures.py`. Per-seed argparse for 01–04; 05/06 take no `--seed` because they only depend on train data + protocol. v02 split YAMLs reused. | Smoke test seed=42 for 01–04; one-shot run for 05/06. |
| **5** | 3-seed sweep, all baselines | Tier 1 (3 FL + 3 NF + 2 FM = 8) × 3 seeds = 24 cells. + Tier 2 (3 FL = 9 cells) + G5 (3 cross-cell × 3 seeds = 9 cells). Total: 24 (Tier 1) + 9 (Tier 2) + 9 (G5) = **42 cells**. G6/G7 computed once each (not in this count). | `multiseed_summary.json`, `heterogeneity_summary.json`, `communication_summary.json`. |
| **6** | `papers/v04_draft/v04_full_baseline_comparison.{md,tex}` + figures | Pareto plot, cross-cell, NF/FM vs trained, **heterogeneity figure (G6)**, **communication table (G7)**, vs v01–v03. | Final paper draft. |

## Dependencies

- v02 split YAMLs (`outputs/v02_fl_8020_ratio/splits/*.yaml`) + cold
  pool definition.
- v02 frozen artifacts (`outputs/v02_fl_8020_ratio/seed*/T2/best.pt`,
  `codebook.npz`) — used as the "v01–v03 method" entry on every
  comparison row.
- v03 results (when v03 lands) — used as the "v01–v03 method,
  personalised" entry on rows that support personalisation.
- New code per the build-order table above.

## What is NOT in scope

- Any new v01-method redesign. Method frozen at v01's NBEATSx + W5
  Hybrid + post-hoc Peak-VQ.
- A second dataset (LCL, Pecan Street, …) — v05 work. v04 stays on
  UMass for direct comparability with v01–v03.
- Federated KMeans / DP-KMeans — orthogonal axis (privacy of the
  codebook itself), v05+.
- New W-mechanisms — v01 §4.6 iter4 has the final word.
