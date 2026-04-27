# v04 — Full baseline comparison: FL × Neural Forecasting × Foundation Models

> Successor to `experiments/v03_kshot_pfl/` (v03 personalization). v04 is the
> *closing* version of the project: it places Peak-Aware VQ on the same axes
> as established FL baselines, non-NBEATSx neural forecasting (NF) baselines,
> and zero-shot foundation models (FM).

## Status

Scaffolding only. Plan in `plans/v04-01_full_baseline_comparison.md`
(placeholder; brainstorming pending).

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
| **FedHiP** (arxiv:2508.04470) | frozen foundation pretrain + cold-side head only | **this is the framing v02/v03 already implicitly use** — making it explicit lets v04 say "v01–v03 method = FedHiP + Peak-VQ" |

The FedHiP row is especially valuable: v02 §2.2 already recasts the v01
backbone as "centrally pretrained, frozen shared encoder = FedHiP". An
explicit FedHiP baseline column (without the Peak-VQ correction) makes
the v01/v02/v03 contribution a clean delta over a published FL pattern.

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
| **Communication-cost accounting** | Bytes-per-round and total bytes for v02's 1-shot codebook vs FedAvg / FedRep / FedProx / Ditto / FedHiP. Quantifies v02's "1 boundary cross" efficiency claim relative to iterative FL. | Paper §results; table |

These close PFL design axes that v02 currently asserts only in framing:
heterogeneity (why personalization is needed) and the communication
subaxis (how much federation costs).

## Open design decisions (for brainstorming)

1. **Common evaluation pool.** v04 must reuse the v02 80:20 split + the
   3 seeds {42, 123, 7} so all numbers are directly comparable to v01–v03.
2. **What to compare per axis.** Cold PAPE, HR@1/HR@2, and (where
   applicable) parameter count and adaptation cost.
3. **FL baselines under what training protocol?** A federated round
   simulation per seed, or a centralized approximation (FedAvg ≈ pooled
   training is technically equivalent in IID; UMass per-apt is non-IID,
   so FedProx adds value here precisely because of the non-IID gap).
4. **FM zero-shot input window.** Use the same L=96, H=24 as v01–v03 so
   the cold gucha sees identical inputs.
5. **Ablation: Peak-Aware VQ on top of FedAvg / FedRep.** Worth a row?
   It would show that the v01 method is *complementary* to FL training
   schemes, not in competition with them.
6. **Communication-cost measurement scope.** Bytes-per-round + total
   bytes for each FL baseline + the 1-shot codebook upload of v02.
   If v03 K-shot results are available by paper deadline, an
   adaptation-time bytes row can be added — by design v03 is fully
   local (0 upload bytes), which would reinforce the "1 boundary
   cross" framing in the joint paper. This row is conditional and not
   a hard dependency.

## Deliverables (target)

- `outputs/v04_full_baseline_comparison/` per-seed × baseline JSONs +
  multiseed summary.
- `papers/v04_draft/v04_full_baseline_comparison.{md,tex}` — final paper.
- `papers/v04_draft/figures/` — Pareto plots, full comparison bars.

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
