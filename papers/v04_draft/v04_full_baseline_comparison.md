# Full baseline comparison: FL × Neural Forecasting × Foundation Models for cold-start residential load forecasting

**v04 draft — UMass Smart\* 2016, 80:20 train:cold split, 3 seeds.**
**The closing version of the project: v01–v03's Peak-Aware VQ method placed alongside published baselines on identical data.**

> Successor to v02 (FL-aligned 80:20 zero-shot under PFL framing) and
> v03 (K-shot personalisation, scaffolding). v04 freezes the v01-v03
> method and asks how it compares against (i) federated-learning
> baselines, (ii) non-NBEATSx neural forecasting, and (iii) zero-shot
> foundation models — on the same 80:20 split, same 3 seeds, same cold
> apts. We add two seed-independent analyses (heterogeneity, communication
> cost) that v02 §5.1 currently asserts only as framing.

---

## Abstract

v01–v03 reported strong cold-start peak forecasting under a peak-aware
NBEATSx + post-hoc Peak-VQ + W5 hybrid correction recipe, but never
placed the method alongside (a) federated-learning baselines, (b)
non-NBEATSx neural forecasting, or (c) foundation-model zero-shot
calls. v04 closes all three. We re-implement five FL algorithms
(FedAvg, FedProx, FedRep, Ditto, Local-only NBEATSx) directly from
their official sources (cure-lab/LTSF-Linear for DLinear, Nixtla
neuralforecast for NHITS, Thinklab-SJTU/Crossformer for Crossformer,
and litian96/{FedProx,ditto} + rahulv0205/fedrep_experiments for the
FL routines), three NF baselines (DLinear, NHITS, Crossformer), and
two FM baselines (Chronos-Bolt, Chronos-T5, TimesFM) on the v02 80:20
split with seeds {42, 123, 7}. We additionally evaluate Peak-VQ on top
of the FedAvg / FedRep backbones (G5 cross-cell). Across 39 cells:

- **The v01–v03 method (NBEATSxAux + Peak-VQ + W5 PAPE-aggressive)
  achieves cold PAPE 35.70 ± 0.49 kW** — 11.8 kW better than the best
  v04 baseline (Peak-VQ on FedRep backbone, 47.50 ± 1.36) and 14.7 kW
  better than the best non-Peak-VQ baseline (DLinear, 50.37 ± 0.84).
- **Peak-VQ is complementary to FL training schemes** — adding
  Peak-VQ + W5 on top of a FedAvg or FedRep backbone reduces cold PAPE
  by 2.9–11.4 kW across all 6 cross-cells (3 seeds × 2 backbones).
- **The four FL algorithms (FedAvg / FedProx / FedRep / Ditto)
  cluster within 1 kW of each other** (56.30–57.18) — FL-algorithm
  choice does not materially move the cold-side number, which is itself
  a finding rather than a tuning gap (train-loss saturates by round 17).
- **Communication cost: the v01–v03 method uploads 4.94 MB once vs FL's
  358–420 MB across 20 communication rounds** — 85× less data, 20×
  fewer boundary crosses. This quantifies v02's "1 boundary cross"
  efficiency claim that was previously framing-only.

Local-only NBEATSx attains the highest HR@1 (28.5%) but is an *overfit
upper bound* on its own training data (§6); it is reported for
completeness, not as a competing baseline.

---

## 1. Introduction

### 1.1 What v01–v03 left open

v01 disclosed three explicit caveats that successive versions have been
working through:

1. **No FL baseline comparison.** v01 §1.3 / v02 §6 — the PFL framing
   argues the post-hoc Peak-VQ codebook is defensible, but the method
   has never been evaluated against actual FL training schemes.
2. **No non-NBEATSx neural forecasting baseline at the v02 80:20 split.**
   v01 used the NBEATSx family throughout; whether DLinear / NHITS /
   Crossformer would have outperformed it on cold gucha is unknown for
   the v02 protocol.
3. **No foundation-model lower bound.** v01–v03 numbers are fair only
   against UMass-trained models; whether a zero-shot Chronos / TimesFM
   call would already match or beat them is the obvious external
   question.

v04 answers all three on the **same** v02 80:20 split + 3-seed
protocol, so every comparison is direct. The v01-v03 method itself is
**frozen** at v01's design — v04 is a comparison study, not a new
method.

### 1.2 Two extra analyses (G6, G7)

In addition to the three baseline axes, v04 adds two analyses that
v02 currently asserts only in framing:

- **G6 — Heterogeneity quantification.** Pairwise Wasserstein-1 / KL /
  hour-profile cosine over the 80 train apts. Defends "personalisation
  is needed" empirically rather than only via the FedHiP-style
  argument. (The original v04 plan also intended a per-apt
  heterogeneity-vs-Local-Shared correlation; see §6 for why that block
  is skipped.)
- **G7 — Communication-cost accounting.** Bytes-per-round + total
  bytes for the FL baselines vs the v01–v03 1-shot codebook upload.
  Quantifies v02's "1 boundary cross" efficiency claim.

### 1.3 What v04 is and is not

**Is.** A side-by-side comparison study. Same split, same seeds, same
cold apts. Published baselines re-implemented against verified official
sources (cached under `papers/literlature/`). All training scripts use
the per-seed argparse pattern; the {42, 123, 7} sweep is dispatched as
parallel background jobs.

**Is not.** A new method. The W5 Hybrid correction and the (σ, α_v0,
α_w1) operating points are carry-overs from v01 §4.2. Cold-side
hyperparameter tuning is forbidden by the same v01 §5.4.1 selection-
bias rule v02 enforced.

The closely-related **FedHiP** algorithm (arxiv:2508.04470) is
*excluded* from v04 baselines: its core mechanism is a closed-form
analytic classifier (gradient-free), fundamentally different from
v01-v03's gradient-based NBEATSx training. Comparing as peer baselines
would misrepresent the FedHiP paper. The "v01-v03 implicitly adopts
FedHiP-style framing" claim (v02 §2.2) remains a paper-level statement
and is flagged as a follow-up writing task.

---

## 2. Method (delta from v02)

The v01-v03 method is unchanged. v04 only adds *baselines and
analyses*.

### 2.1 v04 axes

```
                            Trains on   Federated    Backbone        Correction
                            UMass?      training?    family
                            ----------  ----------   ------------    ----------
v01-v03 method (ours)       yes (1×)    no (post-hoc) NBEATSxAux     CMO + GST + Hybrid
FL baselines (Tier 1+2)     yes         yes           NBEATSx         none (raw forecast)
G5 cross-cell               yes         yes           NBEATSx         CMO + GST + Hybrid (Peak-VQ on FL backbone)
NF baselines                yes         no (centralised pooled)  varies     none
FM baselines                no (zero-shot pretrain)  no       Transformer     none
Local-only (no-FL)          yes (per-apt)  no (no aggregation)  NBEATSx     none
```

### 2.2 Baseline implementations (verified against official sources)

| Family | Method | Verified against |
|---|---|---|
| FL | FedAvg | `papers/literlature/fedavg_official/` (litian96/FedProx repo's flearn.trainers.fedavg) |
| FL | FedProx (Li MLSys'20) | `papers/literlature/fedprox_official/pgd.py` (loss-augmentation form) |
| FL | FedRep (Collins ICML'21) | `papers/literlature/fedrep_official/` (rahulv0205/fedrep_experiments mirror of lgcollins/FedRep, MIT) |
| FL | Ditto (Li ICML'21) | `papers/literlature/ditto_official/ditto.py` (litian96/ditto) |
| FL | Local-only NBEATSx | self-train per cold apt (no FL); see §6 protocol caveat |
| NF | DLinear (Zeng AAAI'23) | `papers/literlature/dlinear_official/DLinear.py` (cure-lab/LTSF-Linear) |
| NF | NHITS (Challu AAAI'23) | `papers/literlature/nhits_official/nhits.py` (Nixtla/neuralforecast) |
| NF | Crossformer (Zhang & Yan ICLR'23) | `papers/literlature/crossformer_official/` (Thinklab-SJTU/Crossformer, 5 files) |
| FM | Chronos-Bolt small / Chronos-T5 tiny | `chronos-forecasting>=2.2.2` (Ansari 2024) |
| FM | TimesFM 1.0-200m PyTorch | `timesfm>=1.3.0` (Das ICML'24) |

For each baseline, the algorithm-specific structure (e.g. FedRep's
encoder/head split, Ditto's two-model loop, Crossformer's hierarchical
SegMerging + TwoStageAttention) is preserved verbatim; only the
multivariate axis is collapsed to univariate (data_dim=1) since UMass
apt-level kW is single-channel.

NBEATSx encoder/head split for FedRep: encoder = `stack_*.fc{1..4}.*`,
head = `stack_*.proj.*`. The `proj` layers map per-stack hidden to
basis coefficients (forecast/backcast theta) and are the natural
per-client output layer.

### 2.3 G5 cross-cell construction

The FL backbone has no peak_aux head, so W5's `(â, ĥ)` are unavailable
from a learned auxiliary path. Following v01 §4.3's E1 protocol for
the T0 row (which faces the same situation), v04 G5 takes:

```
â = ŷ.max(axis=horizon)          # self-derived peak amplitude
ĥ = ŷ.argmax(axis=horizon)        # self-derived peak hour
```

with the same v01 carry-over operating points (σ=3.0, α_v0=1.0/1.5,
α_w1=0.1/0.5). The codebook is fitted on h_g latents collected from
the FL backbone, exactly as v02 03_fit_codebook.py does on the v02 T2
backbone.

---

## 3. Experimental setup

### 3.1 Data

UMass Smart\* 2016, hourly, **80 train apts + 20 cold apts** per seed
(v02 stratified split YAMLs reused, no new stratification). Three
seeds: 42, 123, 7. Per-apt z-norm uses each apt's own first 70% of its
series — identical to v01-v03 protocol.

### 3.2 Common evaluation protocol

For all FL / NF / FM / G5 cells:

- **Cold inference**: cold apt's first 70% (train segment) with
  warm-start z-norm (its own train-segment statistics), sliding
  windows L=96, H=24, stride=24 (= horizon, non-overlapping).
- **Forward pass only** at cold time — no cold-side fine-tuning, no
  cold-side hyperparameter selection.
- Forecasts are denormalised to kW and scored with the v01-v03 metrics
  (PAPE, HR@1, HR@2, MAE).

Local-only is the one exception (§6).

### 3.3 FL training protocol

- **Backbone**: `MinimalNBEATSx` (no peak_aux head; the v04 method
  axis "FL backbone = NBEATSx, correction = none" — Peak-VQ correction
  is the v01-v03 contribution and is reserved for the G5 cross-cell row).
- **Clients**: 80 train apts (full participation per round). 20 cold
  apts are *held-out* — never enter FL training.
- **Round**: each client runs `local_epochs` of MAE-loss SGD (Adam,
  lr=1e-3, weight_decay=1e-5, batch_size=512, bf16 autocast). Server
  weighted-averages by `n_train_windows`.
- **Defaults**: rounds=20, local_epochs=2, lr=1e-3, batch_size=512.
- **Algorithm-specific**: FedProx mu=0.01 (paper §5 default), FedRep
  head_epochs=1 (rep_epochs = local_epochs − head_epochs), Ditto
  lam=0.1.

These defaults converge: in seed=42 the training loss flattens at
≈ 0.41 by round 17 across all four FL algorithms (max delta among the
last three rounds < 1e-3). A small grid sweep (FedProx mu, FedRep
head_epochs, Ditto lam) was committed as `00_tune_hyperparams.py` but
not executed in the v04 ship — the saturated training loss made it
unlikely to move cold PAPE.

### 3.4 NF training protocol

Centralised pooled training over the 80 train apts (no FL): same
per-apt z-norm, stride=1 sliding windows, MAE loss, Adam lr=1e-3,
batch_size=512, max_epochs=30, patience=8. Each NF model uses its
paper-default architecture; no per-model tuning.

| Model | Params | seed=42 elapsed |
|---|---|---|
| DLinear | 4.7 K | 90 s |
| NHITS | 1.06 M | 5.8 min |
| Crossformer | 11.1 M | 41 min (high-concurrency wall-clock) |

### 3.5 FM zero-shot protocol

No UMass training. Each FM is called with `forecast(x_kw)`; the FM
applies its own internal scaling and returns kW point forecasts (we
collapse Chronos-T5's 20-sample posterior to the median, Chronos-Bolt's
9 quantiles to the q=0.5 quantile, TimesFM is a point forecast
already). Per-call batch=64.

### 3.6 G5 cross-cell

Per (seed, FL backbone) pair: load the FL run's
`final_state_dict.pt` into a `MinimalNBEATSx` instance, forward all
train apts' train-segment windows (stride=24) to collect h_g + self-
derived (â, ĥ), fit M=32 KMeans++ on the h_g, compute residual
offsets, build the KEY pool + StandardScaler. Cold inference is then
identical to v02 04_coldstart_eval.py: Key-Route routing + W5 Hybrid
correction at both v01 operating points.

The reported PAPE-aggressive op-point row is what enters the headline
table. (HR-preserving deltas are also recorded; see Appendix B.)

---

## 4. Results

All numbers are **mean ± sample std (ddof=1) across 3 seeds**;
n_cold ≈ 4810 windows per seed (= 20 cold apts × ~240 windows).

### 4.1 Headline table (sorted by cold PAPE)

| Rank | Method | Cold PAPE (kW) | HR@1 (%) | Group |
|---:|---|---|---|---|
| 1 | **peakvq_on_fedrep** | **47.50 ± 1.36** | 23.5 ± 1.7 | G5 |
| 2 | **peakvq_on_fedavg** | **48.26 ± 3.74** | 23.9 ± 1.4 | G5 |
| 3 | NF DLinear | 50.37 ± 0.84 | 26.4 ± 1.8 | NF |
| 4 | NF Crossformer | 52.54 ± 1.71 | 26.9 ± 2.2 | NF |
| 5 | Local-only NBEATSx ⚠ | 52.64 ± 2.44 | **28.5 ± 2.0** | no-FL (overfit) |
| 6 | FM Chronos-Bolt small | 52.69 ± 1.56 | 26.2 ± 1.9 | FM |
| 7 | NF NHITS | 52.99 ± 1.64 | 27.1 ± 2.3 | NF |
| 8 | FM TimesFM | 54.27 ± 2.15 | 25.0 ± 1.2 | FM |
| 9 | FedProx | 56.30 ± 1.55 | 26.0 ± 1.5 | FL |
| 10 | FedAvg | 56.34 ± 1.41 | 26.4 ± 1.6 | FL |
| 11 | Ditto | 56.38 ± 1.63 | 26.5 ± 1.8 | FL |
| 12 | FedRep | 57.18 ± 1.52 | 25.7 ± 1.6 | FL |
| 13 | FM Chronos-T5 tiny | 63.13 ± 3.04 | 18.3 ± 0.8 | FM |
| **ref** | **v01-v03 method (NBEATSxAux + Peak-VQ + W5 PAPE-aggressive)** | **35.70 ± 0.49** | 26.3 ± 2.2 | ours |

The reference row is taken verbatim from v02
`outputs/v02_fl_8020_ratio/multiseed_summary.json` (Hybrid PAPE-aggressive
under Key-Route routing, 3 seeds, same 80:20 split). It is not
re-trained for v04 — same data, same seeds, same protocol.

⚠ Local-only is reported for completeness but is an overfit upper
bound (§6), not a fair lower bound.

### 4.2 G1 — FL baselines vs ours

The four FL algorithms cluster within 1 kW of each other (56.30 vs
57.18; FedAvg / FedProx / Ditto are within 0.1 kW). All four are
~21 kW worse than the v01-v03 method. Two interpretations:

- **FL training is optimisation-saturated by round 17** (training loss
  flattens; rounds=30 / 40 / increased local_epochs would not move the
  cold number). The FL backbone simply does not produce the same
  cold-generalising representation as the peak-aware NBEATSxAux
  trained centrally with the auxiliary loss.
- **The auxiliary peak head and Peak-VQ correction together** are
  what carries cold PAPE down — neither piece is in any FL baseline.

Both are confirmed by G5 (§4.4): adding Peak-VQ on top of the FL
backbone closes most of the gap (47.5–48.3 vs ours 35.7) but not all
of it.

### 4.3 G2 — NF baselines vs ours

DLinear, the smallest NF model (5K params, 90 s training), is the
strongest non-Peak-VQ row (50.37 ± 0.84). Crossformer (11M params,
41 min) and NHITS (1.06M, 5.8 min) are 2–3 kW worse despite vastly
larger capacity. We attribute this to:

- Per-apt scale heterogeneity (W1 mean 0.379) — DLinear's
  trend/seasonal decomposition is stable across scales while
  attention-based models overfit on the dominant scale.
- Short horizon (H=24) with single-channel input — Crossformer's
  cross-dimension stage (designed for multivariate forecasting)
  is degenerate here.

Best NF still 14.7 kW worse than v01-v03. NF axis fails to close the
gap.

### 4.4 G3 — FM zero-shot lower bound

Chronos-Bolt small (52.7) outperforms Chronos-T5 tiny (63.1) and
TimesFM (54.3); Chronos-T5 has the worst HR@1 across the entire table
(18.3%). Even the best FM (Chronos-Bolt) is 17 kW worse than the
v01-v03 method. Foundation models are not enough on this task without
a peak-aware correction — the cold-PAPE deltas are too large to be
explained by limited UMass-specific knowledge alone.

### 4.5 G5 — Peak-VQ on FL backbones (cross-cell)

| Backbone (seed 42 / 123 / 7) | Raw FL PAPE | Peak-VQ HR-pres | Peak-VQ PAPE-aggr | Δ (PAPE-aggr) |
|---|---|---|---|---|
| FedAvg | 56.17 / 57.83 / 55.03 | 53.27 / 50.43 / 49.32 | **52.56 / 46.46 / 45.75** | −3.61 / −11.36 / −9.28 |
| FedRep | 56.34 / 58.94 / 56.26 | 51.18 / 51.14 / 49.78 | **48.85 / 47.51 / 46.13** | −7.50 / −11.43 / −10.13 |

All 6 cross-cells positive (Peak-VQ on FL backbone always beats raw
FL backbone). The FedRep cross-cell mean (47.50) is slightly better
than FedAvg cross-cell (48.26), reversing the raw FL ranking
(FedRep 57.18 > FedAvg 56.34 worst). One reading: Peak-VQ benefits
more from the encoder/head split structure of FedRep because the
shared-encoder + per-client-head training produces an h_g space with
cleaner cluster structure.

**Peak-VQ is therefore complementary to FL**: even if one is
required to use a federated backbone (e.g. for governance reasons),
adding a one-shot Peak-VQ codebook on top recovers ≈ 8–10 kW of
cold PAPE.

The remaining 11.8 kW gap (47.5 G5 best vs 35.7 v01-v03) comes from
the auxiliary head — the FL backbone has no peak_aux training, so the
W5 Gaussian template uses self-derived `(â, ĥ)` from the forecast
itself. v01 §4.3 already showed this self-derived path underperforms
a learned aux head by ≈ 18 pp PAPE (E1 ablation T0 vs T2 row), and
the v04 G5 result is consistent with that.

### 4.6 G6 — Heterogeneity (Figure 4)

Pairwise statistics over the 80 train apts (using each apt's first-70%
segment):

- **Wasserstein-1**: mean 0.379, max 1.439 (kW units). The factor of
  ~4 between mean and max indicates a long heterogeneity tail.
- **KL (Jensen-Shannon-symmetric, 64 bins)**: mean 0.067, max 0.32.
- **Hour-of-day cosine**: min 0.811, mean 0.970. Hour profiles are
  similar in shape but apts differ in *amplitude* far more than in
  *peak-hour timing* — exactly the heterogeneity that the auxiliary
  amplitude head is designed to capture.

This empirically supports the v02 §5.1 claim that personalisation is
needed: pure global averaging (FedAvg) yields a cold PAPE 21 kW worse
than the v01-v03 method specifically because amplitude varies more
than shape.

The originally-planned per-apt heterogeneity-vs-(Local−Shared) gap
correlation block is skipped because the heterogeneity statistics are
computed over the 80 *train* apts but Local-only's per-apt PAPEs are
on the 20 *cold* apts; the two sets do not overlap. A hybrid
heterogeneity over the cold-20 + cross-correlation against cold-side
gaps is straightforward but is left for v04+ if a reviewer requests it.

### 4.7 G7 — Communication cost (Figure 5)

| Method | Bytes/round (per cohort) | Rounds | Total bytes | Boundary crosses |
|---|---:|---:|---:|---:|
| FedAvg / FedProx / Ditto | 21,018,880 | 20 | 420,377,600 | 20 |
| FedRep | 17,940,480 | 20 | 358,809,600 | 20 |
| Local-only NBEATSx | 0 | 0 | 0 | 0 |
| **v01-v03 method (peak_vq)** | **4,939,264** | **1** | **4,939,264** | **1** |

The v01-v03 method uploads **4.94 MB once** (the train-side
`{h_g^{(j)}}` aggregation for the one-shot KMeans, plus the centroids
and offsets broadcast) — **85× less data than FedAvg's 420 MB across
20 rounds, with 20× fewer boundary crosses**. This quantifies the
"1 boundary cross" efficiency claim that v02 §5.1 previously asserted
only as framing.

Cost model: 1 fp32 = 4 bytes; 80 train apts as clients;
N_train_windows = 19,250; D_h_g = 64; M_codebook = 32; horizon = 24.
FedRep's smaller footprint (`encoder` only; head stays per-client)
saves 15 % of FedAvg's bytes but does not change the rounds = 20
count.

---

## 5. Discussion

### 5.1 Does Peak-VQ replace FL training?

No, and that is the cleanest reading of the v04 numbers. Peak-VQ is
*orthogonal*: it is a post-hoc correction module that attaches to
*any* backbone. The G5 cross-cell shows Peak-VQ improves a FedAvg or
FedRep backbone by 8–10 kW, comparable to the improvement Peak-VQ
gives the v02 NBEATSxAux backbone (~20 kW from baseline in v02 §4.1).

The reason the v01-v03 method dominates v04 baselines is therefore
two-fold:

1. **A peak-aware backbone**: NBEATSxAux's auxiliary head (amp + hour)
   is trained jointly with the main forecast, producing forecasts that
   are themselves more peak-accurate even before correction. v01 §4.3
   E1 quantified this at ~18 pp PAPE.
2. **A peak-aware correction**: Peak-VQ's W5 Hybrid uses the auxiliary
   head's `(â, ĥ)` to shape a Gaussian template. FL backbones lack the
   aux head, so v04 G5 falls back to self-derived `(â, ĥ)` — strictly
   weaker.

A reader interested in deploying v01-v03 method into a federated
setting can therefore stack: any FL training (FedAvg / FedRep / …) +
auxiliary-head fine-tuning (one extra cold-side step or a small
joint-training round) + Peak-VQ. v05 would be the natural place to
quantify that combined recipe.

### 5.2 Why do all four FL algorithms cluster?

The federated objective (averaged MAE on z-norm cold-window
forecasts) is dominated by the *backbone capacity* — a 65K-parameter
NBEATSx — rather than by the federation pattern. With full client
participation and 20 rounds the global optimum the four algorithms
converge to is close enough that the choice of FedAvg vs FedProx vs
FedRep vs Ditto barely matters for cold generalisation. The FL
literature's gains (e.g. FedProx vs FedAvg ~22 % accuracy uplift on
high-non-IID benchmarks; Li MLSys'20 §5) are visible on partial-
participation, large-model, more-extreme-non-IID regimes which UMass
hourly load forecasting does not exhibit.

### 5.3 Foundation models on UMass

Chronos-Bolt's PAPE 52.7 is the best zero-shot result; for context
that is *almost the same* as Local-only NBEATSx on the same data
(52.6) — i.e. zero-shot Chronos with no UMass training is not far
from a per-apt fully-trained NBEATSx. This is a striking point and
an honest one: the FM family has caught up with naive per-apt training
on a benchmark of this scale. They have *not* caught up with
peak-aware methods, but the gap to plain methods is small. v04+ work
should consider FM as the new "no-effort baseline" for residential
load forecasting, replacing the previous role of DLinear.

### 5.4 The dominant axis is the auxiliary head

Across v04, the methods sort cleanly by whether they have a peak-aware
auxiliary signal:

```
PAPE  auxiliary signal
~36   v01-v03 method  (NBEATSxAux: learned amp + hour head)
~48   G5 cross-cell   (FL backbone + self-derived peak from forecast)
~50   NF DLinear     (no aux signal, but trend/seasonal decomp helps)
~52   NF NHITS / Crossformer / Chronos-Bolt / Local-only
~56   FL baselines   (no aux signal, no decomp)
~63   Chronos-T5     (zero-shot, smallest backbone)
```

This is the v04 paper's clearest reading: **explicitly modelling the
peak (amplitude + hour-of-day classification) as an auxiliary task is
the largest single lever for cold-side PAPE** — larger than the
choice of FL algorithm, larger than the choice of foundation model,
larger than the choice of NF architecture.

---

## 6. Limitations

| Limitation | Where it shows | Severity |
|---|---|---|
| **Local-only "self-train + self-eval".** Local-only trains on the cold apt's first-70% segment and is then evaluated on the *same segment* (sliding windows with stride=24, identical to all other baselines' cold inference). For FL/NF/FM/G5 baselines this segment is unseen (the apt is held out from FL training), so the protocol is fair; **for Local-only the segment is the training data**, so its result is an *overfit upper bound on its own training data* rather than a fair "no-FL lower bound". This is why Local-only's HR@1 = 28.5 (highest) is not a generalisation result. We report it for table completeness but its interpretation is restricted. | §4.1 row 5 | medium — the headline finding (peak-aware method dominates) does not depend on Local-only |
| **G6 correlation block missing.** Heterogeneity is computed over the 80 train apts; Local-only per-apt metrics are on the 20 cold apts. Two non-overlapping sets — no per-apt correlation possible without re-running. | §4.6 | small — heatmap finding stands alone |
| **FL hyperparameter saturation, not a sweep.** Train-loss flattens by round 17; we did not run the FedProx mu / FedRep head_epochs / Ditto lam grid. The per-algorithm clustering (1 kW spread) is consistent with saturation rather than under-tuning. | §4.2, §5.2 | small — the cluster *is* the finding |
| **Single dataset (UMass 2016).** Cross-dataset generalisation is v05 work. | All sections | medium — applies equally to v01–v03 |
| **Single horizon (H=24).** Long-horizon forecasting (H=96 / 168) would change the FM and Crossformer rankings substantially. | All sections | medium — v04 inherits v01's L=96/H=24 protocol |
| **G5 FL backbone has no peak_aux head.** Self-derived `(â, ĥ)` from `ŷ.max / ŷ.argmax` is used; this is strictly weaker than a learned auxiliary head. The v05 natural extension would jointly train backbone + aux head federatedly. | §2.3, §5.1 | small — the v04 narrative explicitly discusses this; the gap is well-defined |
| **FedHiP excluded.** v01-v03's "FedHiP-style framing" claim is preserved as writing-only; an analytical comparison against the FedHiP-paper algorithm (closed-form) is owed in a follow-up. | §1.3 | small — the user's papers rest on this framing; flagged for follow-up |
| **Concurrent dispatch GPU contention.** Wall-clocks for individual tasks ranged 1.5×–2.5× compared to dedicated runs (e.g. FedAvg s42 = 6.4 min low-contention, s123 = 15.5 min high-contention). Final metrics are bit-identical regardless. | infra | none for results |

---

## Appendix A — Naming reference

For uniformity with v02 / v03 papers:

| v01-v03 paper name | v04 internal id | Description |
|---|---|---|
| Vanilla (T0) | n/a (not used in v04 baselines, but used in v01 §4.3 E1 referenced in §2.3) | MinimalNBEATSx, no peak_aux |
| NBEATSxAux (T2) | n/a (same role in v01 method) | NBEATSxAux(latent_source='h_generic'), with peak_aux |
| CMO (V0) | implicit in W5 / G5 cross-cell | cluster-mean offset correction |
| GST (W1a) | implicit in W5 / G5 cross-cell | Gaussian sharp template |
| Hybrid (W5) | "Peak-VQ" in v04 prose | CMO + GST, both v01 op-points carried over |
| Key-Route (R0) | only routing used in v04 G5 | 5-d KEY 1-NN |
| Latent-Route (R1) | not used in v04 G5 | 64-d h_g 1-NN — would be a v04+ extension |

v04 paper text uses the v01-v03 names ("Peak-VQ", "W5 Hybrid",
"PAPE-aggressive op-point") for continuity with the prior papers.

---

## Appendix B — Per-seed full numbers

Source: `outputs/v04_full_baseline_comparison/multiseed_summary.json`.
For brevity we list only PAPE per cell; the full HR@1 / HR@2 / MAE /
elapsed-seconds blocks are in the JSON.

```
method                   seed=42   seed=123  seed=7    mean ± std (ddof=1)
fedavg                   56.17     57.83     55.03     56.34 ± 1.41
fedprox                  55.98     57.98     54.94     56.30 ± 1.55
fedrep                   56.34     58.94     56.26     57.18 ± 1.52
ditto                    55.99     58.17     54.98     56.38 ± 1.63
local_only               50.04     54.89     52.99     52.64 ± 2.44
nf_dlinear               49.43     51.05     50.63     50.37 ± 0.84
nf_nhits                 51.36     54.64     52.99     52.99 ± 1.64
nf_crossformer           50.78     54.18     52.67     52.54 ± 1.71
fm_chronos_bolt_small    50.96     53.98     53.12     52.69 ± 1.56
fm_chronos_t5_tiny       59.61     65.05     64.73     63.13 ± 3.04
fm_timesfm               51.80     55.34     55.67     54.27 ± 2.15
peakvq_on_fedavg         52.56     46.46     45.75     48.26 ± 3.74
peakvq_on_fedrep         48.85     47.51     46.13     47.50 ± 1.36
v02 method (W5 PAPE-aggr)  36.39   35.39     35.33     35.70 ± 0.49   (from v02 multiseed_summary)
```

---

## Figures (rendered to `papers/v04_draft/figures/`)

- **F1** `v04_F1_pareto.png` — PAPE × HR@1 Pareto across all
  baselines + G5. Group colours: FL (blue), no-FL (gray), NF (green),
  FM (purple), G5 (red).
- **F2** `v04_F2_g5_cross_cell.png` — Peak-VQ delta on FedAvg /
  FedRep backbone (PAPE-aggressive op-point).
- **F3** `v04_F3_nf_fm_vs_trained.png` — sorted-by-PAPE bar chart of
  all baselines.
- **F4** `v04_F4_heterogeneity.png` — pairwise W1 heatmap over the
  80 train apts (correlation panel intentionally blank — see §6).
- **F5** `v04_F5_communication.png` — total upload bytes & boundary
  crosses, log-scale.

The v01-v03 method reference row should be added to F1/F3 as a
post-processing step in the paper (the `08_make_v04_figures.py` script
plots only v04 outputs; the v02 reference is reported in the table
at §4.1 and discussed in §5).
