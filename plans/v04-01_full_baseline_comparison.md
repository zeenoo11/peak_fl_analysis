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

## FL baselines

### Tier 1 (mandatory)

| Baseline | Algorithm | What is federated | Personalization |
|---|---|---|---|
| **FedAvg** | McMahan et al., AISTATS'17 | full backbone weights, vanilla averaging | none — single global model |
| **FedRep** | Collins et al., ICML'21 | shared encoder; per-client personalized head | per-client head, encoder frozen on cold |
| **Local-only NBEATSx** | — | nothing | each cold gucha trains its own NBEATSx independently from scratch |

### Tier 2 (recommended)

| Baseline | Algorithm | Relationship to v01–v03 |
|---|---|---|
| **FedProx** | Li et al., MLSys'20 | FedAvg + proximal term ‖w_local − w_global‖²; addresses non-IID stability — UMass per-apt is non-IID, so this directly tests the "FedAvg might be hurting us" hypothesis |
| **Ditto** | Li et al., ICML'21 | global + per-client local, regularised toward global; stronger personalisation than FedRep — directly competes with v03 F2c (LoRA) |
| **FedHiP** | arxiv:2508.04470 | frozen foundation pretrain + cold-side head adaptation; **v02 §2.2 already implicitly adopts this framing** — making it explicit lets v04 state "v01–v03 method = FedHiP + Peak-VQ" cleanly |

The FedHiP row is especially valuable: it isolates the contribution of
*Peak-VQ + W5 hybrid correction* over the underlying FL pattern v02
already inherits.

## NF baselines (TBD — brainstorming)

Candidate set:
- **DLinear** (Zeng et al., AAAI'23) — linear per-channel
- **NHiTS** (Challu et al., AAAI'23) — multi-rate stack
- **Crossformer** (Zhang & Yan, ICLR'23) — cross-time-cross-variate

These are already trained in `Peak_Analysis/v10_b1/b3/b4` on the 50:50
split. v04 must **re-train them on the v02 80:20 split** (same 3 seeds,
same 80 train / 20 cold, same per-apt z-norm) for fair comparison.

Open question: how many of {DLinear, NHiTS, Crossformer} to include.
Default: all three.

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
├── multiseed_summary.json
└── FINAL_v04_report.md
```

## Deliverables

1. **`papers/v04_draft/v04_full_baseline_comparison.md`** + IEEE `.tex`.
2. **Pareto plots**: `papers/v04_draft/figures/v04_F*.png`
   - F1: PAPE × HR@1 Pareto across all baselines + ours.
   - F2: G5 cross-cell — Peak-VQ delta on FedAvg / FedRep / FedHiP.
   - F3: NF / FM zero-shot vs trained comparison.

## Open questions (brainstorming pending)

1. **NF baseline list final.** All three (DLinear / NHiTS / Crossformer)
   or just the strongest two on the v01 50:50 results?
2. **FM baseline count.** At minimum Chronos + TimesFM. Add Lag-Llama /
   Moirai / Time-MoE if compute allows?
3. **FL simulation protocol.** Number of communication rounds, local
   epochs, client-batch size — settle on a single (rounds, local_steps)
   pair shared by all FL baselines for clean comparability.
4. **G5 (cross-cell) scope.** All three of FedAvg / FedRep / FedHiP, or
   only FedHiP (cleanest narrative match to v02/v03)?
5. **Tier 2 inclusion gate.** Time-budget question — Tier 2 = ~3× the
   work of Tier 1. Decide before kickoff whether v04 ships with Tier 1
   only, Tier 1+2, or Tier 2 deferred to a v04-extended.

## Dependencies

- v02 split YAMLs + cold pool definition.
- v02 frozen artifacts (`outputs/v02_fl_8020_ratio/seed*/T2/best.pt`,
  `codebook.npz`) — used as the "v01–v03 method" entry on every
  comparison row.
- v03 results — used as the "v01–v03 method, personalised" entry where
  the comparison row supports personalisation (FedRep / Ditto).
- New code:
  - `src/fl/` — FedAvg / FedRep / FedProx / Ditto / FedHiP simulation
    drivers.
  - `src/models/` — DLinear / NHiTS / Crossformer for the NF axis (or
    re-use Peak_Analysis ports).
  - `src/fm/` — Chronos / TimesFM zero-shot wrappers.

## What is NOT in scope

- Any new v01-method redesign. Method frozen at v01's NBEATSx + W5
  Hybrid + post-hoc Peak-VQ.
- A second dataset (LCL, Pecan Street, …) — v05 work. v04 stays on
  UMass for direct comparability with v01–v03.
- Federated KMeans / DP-KMeans — orthogonal axis (privacy of the
  codebook itself), v05+.
- New W-mechanisms — v01 §4.6 iter4 has the final word.
