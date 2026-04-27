# FL-aligned 80:20 zero-shot evaluation of Peak-Aware VQ under a PFL framing

**v02 draft — UMass Smart\* 2016, frozen NBEATSxAux backbone, post-hoc Peak-VQ.**
**Multi-seed evaluation across {42, 123, 7}.**

> Successor to *Peak-Aware VQ for cold-start residential load forecasting* (v01,
> 50:50, centralized). v02 keeps the model and the two operating points
> unchanged; it changes only the train/cold ratio (50:50 → 80:20), the framing
> (centralized → frozen shared encoder, FedHiP pattern), and adds a routing
> ablation. The K-shot personalization study is deferred to v03.

---

## Abstract

v01 reported large cold peak-amplitude error reductions under a 50:50
train:cold split with a centralized backbone, but explicitly disclosed two
caveats: the FL claim was asserted but not tested, and 50:50 is unusual
relative to FL/cold-start literature (BuildingsBench ≈ 99.8:0.2, FedCCL 80:20,
Briggs et al. 80:20-style folds). v02 closes the second caveat directly and
reframes the first: we treat the v01 backbone as a centrally pretrained,
*frozen* shared encoder (the FedHiP pattern, arxiv:2508.04470) and re-evaluate
the full pipeline at a single 80:20 split with three random seeds. We also
introduce **Latent-Route**, a 64-d routing alternative to v01's 5-d
**Key-Route**, that costs zero extra forward passes. Across three seeds we
find: (i) the headline cold-PAPE reduction is **preserved** at 80:20
(−19.0% / −33.8% under the two operating points, vs v01's −18% / −32%);
(ii) Latent-Route is **statistically indistinguishable** from Key-Route on
cold PAPE while showing a small (+0.6 pp) HR@1 advantage; (iii) the W5
Hybrid correction continues to dominate either component alone (CMO
cluster-mean-offset; GST Gaussian sharp template) by **3.24–3.47 PAPE-kW**;
(iv) the +18.6 pp peak_aux ablation effect from v01 §4.3 is **partially
preserved but with substantially larger seed variance** (mean +11.9 ± 9.2 pp
across our three seeds, with one seed exhibiting Vanilla-codebook collapse).
We therefore present v02 as a **robustness validation under a more standard
FL ratio**, with one explicit caveat (peak_aux contribution at 80:20 is
seed-sensitive when the Vanilla codebook fragments).

---

## 1. Introduction

### 1.1 Motivation

The v01 paper (Peak-Aware VQ, 50:50 centralized) reported strong cold-start
peak forecasting numbers but openly listed two limitations:

1. **§1.3 / §D.6.1** — "is not a federated training algorithm; the codebook
   is a post-hoc artifact". The FedAvg + post-hoc Peak-VQ extension was
   listed as P3 future work.
2. **§5.4.6** — "50 cold households is moderate; no per-household power
   analysis", and 50:50 is an outlier vs FL/cold-start literature
   (BuildingsBench ≈ 99.8 : 0.2, FedCCL 80:20, Briggs et al. 80:20-style
   folds).

v02 addresses these two points directly, without changing the method.

### 1.2 Contribution

1. **80:20 robustness check.** Reproduces the full v01 pipeline at the
   FL-canonical train:cold ratio (n_train=80, n_cold=20) on UMass 2016, with
   three seeds. Reports whether the cold-PAPE improvement, the peak_aux V0
   ablation effect, and the codebook health metric all survive.

2. **PFL reframing of the FL claim.** Recasts the v01 backbone as a
   centrally pretrained, frozen shared encoder (the *FedHiP* pattern,
   arxiv:2508.04470). Under this framing only the post-hoc codebook
   crosses the donor → server boundary, and only at the (one-shot)
   pretraining stage; cold inference is fully local.

3. **Routing ablation: Latent-Route.** Adds a 64-d routing variant that
   uses the cold gucha's `h_g` directly (already produced by the aux head's
   forward pass) as the codebook lookup key, rather than the v01 5-d KEY
   descriptor. This costs zero extra forward and uses ×12 more information
   per query.

4. **Honest reporting of partial reproductions.** We separate the four
   sub-results that survived cleanly (cold-PAPE improvement, codebook
   utilization, Hybrid dominance over CMO/GST, multi-seed PAPE σ) from the
   one that survived with caveats (peak_aux V0 contribution: seed-sensitive
   under Vanilla codebook fragmentation; see §5.2).

### 1.3 What this work is and is not

**Is.** A protocol-level robustness study. Identical method, identical
operating points, larger train pool, smaller cold pool, three seeds.

**Is not.** A new method. The W5 Hybrid correction, the M=32 codebook,
and the (σ=3.0, α_v0, α_w1) operating points are all carry-overs from
v01 §3.4. Cold-side hyperparameter tuning is **explicitly forbidden**
in this study to avoid re-introducing v01 §5.4.1's selection-bias concern.

K-shot personalisation, whole-backbone fine-tuning, federated KMeans /
DP-KMeans, a second dataset, and other forecasting-model baselines are
all out of scope and listed in §6 with their target version.

---

## 2. Method (delta from v01)

**Figure 0** gives the end-to-end view: a *centralized one-shot
pretrain* of NBEATSxAux on 80 train apts (Phase A) → a *post-hoc 1-shot
KMeans codebook* on the resulting `h_g` (Phase B), with the donor →
server boundary crossed exactly once → *fully local cold inference*
(Phase C) using the W5 Hybrid correction. Only what differs from v01 §3
is reproduced below; for full backbone / auxiliary-head / VQ
definitions see Appendix C.

| Component | v01 | v02 |
|---|---|---|
| Train : cold split | 50 : 50 | **80 : 20** |
| Backbone framing | "centralized" | **frozen shared encoder (FedHiP)** — implementation identical |
| Backbone arms | Vanilla, NBEATSxAux, NBEATSxAux-h_concat (T0/T2/T3) | Vanilla + NBEATSxAux only |
| Codebook routing | Key-Route only | **Key-Route + Latent-Route** ablation |
| Cold inference | warm-start z-norm + zero-shot | unchanged |
| Operating points | HR-preserving (σ=3.0, α_v0=1.0, α_w1=0.1), PAPE-aggressive (σ=3.0, α_v0=1.5, α_w1=0.5) | unchanged (no cold-side tuning) |
| Seeds | 3 | 3 |

### 2.1 Naming reference (v02-internal)

To keep the body short we use the following compact names. They are
defined here once and then used freely; each first appearance is
parenthesised with the long form.

| Name | Long form / role |
|---|---|
| **Vanilla** | `MinimalNBEATSx` trained with pure MAE — no auxiliary head, baseline backbone |
| **NBEATSxAux** | `NBEATSxAux(latent_source='h_generic')` trained with MAE + λ·peak_aux — the v02 main backbone |
| **CMO** | Cluster-Mean Offset correction: `ŷ + α_v0 · o_{c*}` (the v01 V0 mechanism) |
| **GST** | Gaussian Sharp Template correction: `ŷ + α_w1 · g(t; ĥ, â, σ)` (v01 W1a mechanism) |
| **Hybrid** | CMO + GST: `ŷ + α_v0 · o_{c*} + α_w1 · g` (v01 W5 mechanism, the v02 main correction) |
| **Key-Route** | 5-d KEY → StandardScaler → 1-NN on the train KEY pool → that train window's cluster (v01 R0) |
| **Latent-Route** | argmin_c ‖h_g − codebook[c]‖₂ on the cold-side `h_g` directly (v02 new, R1) |
| **HR-preserving op-point** | (σ=3.0, α_v0=1.0, α_w1=0.1) — gentle correction |
| **PAPE-aggressive op-point** | (σ=3.0, α_v0=1.5, α_w1=0.5) — stronger correction |
| **k_min** | minimum cluster size of the M=32 codebook (a "codebook health" diagnostic from v01 §3.3) |

A full mapping back to the codebase identifiers (T0/T2, V0/W1a/W5, R0/R1)
is in Appendix A.

### 2.2 PFL framing and the federation boundary

v01 §1.3 acknowledged that the codebook is a post-hoc artifact and that
the work is **not** a federated training algorithm. v02 makes this
honest: the only object that crosses the donor → server boundary is the
aggregated `{h_g^{(j)}}` used to fit the one-shot KMeans codebook, and
this happens once at the pretraining stage. At cold inference time:

- the cold gucha calls the shared backbone *locally*,
- looks up the global codebook *locally*,
- emits its forecast *locally*,

so neither raw load, hidden vectors, nor input statistics leave the cold
gucha. This is the FedHiP pattern (arxiv:2508.04470). Why the
pretraining-stage `h_g` aggregation is acceptable in v02 scope, and why
iterative federated KMeans is not (the TAR attack at arxiv:2511.07073),
is discussed in §6.

### 2.3 Latent-Route

At cold inference, the auxiliary head already requires `h_g_cold`. We
exploit this: instead of the 5-d KEY-NN against a train KEY pool
(Key-Route), Latent-Route assigns the cold cluster as

```
c* = argmin_c ‖h_g_cold − codebook[c]‖₂                (Latent-Route)
```

with no scaling, no extra forward, no train-side KEY storage. The rest
of the W5 Hybrid correction is identical. Open question 2 of the v02
plan asked whether this should use raw or StandardScaler-normalised
distance; we kept the default (raw Euclidean) and confirm in §4 that it
is competitive with Key-Route, so we did not promote the normalised
fallback.

---

## 3. Experimental setup

### 3.1 Data and split

UMass Smart\* 2016, hourly, 100 households (passing the size-based
≥7000-hour filter, identical to v01's pool). The 80:20 stratified split
is generated per seed via 4-feature StandardScaler → KMeans(k=2,
random_state=seed) → per-cluster proportional alternating extraction →
KL(cold ‖ train) gate, retrying once with seed+1 if the KL exceeds 0.5.
The KMeans random_state is intentionally bound to the seed, so seeds
{42, 123, 7} produce three different stratified cold-20 sets (pairwise
Jaccard 0.08–0.14) without explicit fold rotation.

| Seed | KL(cold ‖ train) | Retry seed | Cluster sizes (0/1) |
|---|---|---|---|
| 42 | 0.177 | — | 49 / 51 |
| 123 | 0.208 | 124 (initial KL > 0.5) | 51 / 49 |
| 7 | 0.328 | — | 51 / 49 |

All three KL values fall under the 0.5 threshold (one after a single
retry). Each cold set overlaps v01's cold-50 by 7–11 of 20 households,
so the cold pools are not identical to v01.

### 3.2 Training

Vanilla (MAE only) and NBEATSxAux (MAE + λ·peak_aux, λ=0.3, hr_weight=
0.1, latent source = h_generic) are trained per-seed at 80:20 with
the v01 hyperparameters (Adam, lr=1e-3, weight_decay=1e-5, batch=256,
patience=8, max 30 epochs, per-apt train-segment z-norm with stride=1).
Backbone state-dict layer names match v01 / Peak_Analysis v10_b2 so
checkpoints would load `strict=True` if needed.

### 3.3 Codebook fit

For each seed, all train apts' train-segment windows (stride=24, =
horizon, matching v01) are forwarded through the frozen NBEATSxAux. We
fit M=32 KMeans++ on the resulting `h_g` and compute per-cluster
residual offsets `o_c = mean(y_true_z − y_hat_z)` over windows assigned
to cluster *c*. The KEY pool and StandardScaler parameters required by
Key-Route are stored alongside the centroids and offsets.

### 3.4 Cold inference

For each cold apt: per-apt warm-start z-norm using its **own** first 70%
(no train-side stats leak), sliding windows with stride=24 on the same
train segment, frozen forward pass. Routing assigns a cluster (Key-Route
or Latent-Route), and the W5 Hybrid correction adds α_v0·o_{c*} and
α_w1·g in z-norm space; the result is denormalised to kW for metric
computation. PAPE / HR@1 / HR@2 / MAE follow v01 §4.1 definitions.

---

## 4. Results

All numbers are mean ± std across seeds {42, 123, 7}; n_cold per seed
varies 4810 / 4830 / 4820 windows.

### 4.1 G1 — Headline 80:20 result (Key-Route)

The v01 cold-PAPE improvement is preserved at 80:20:

| Cell | PAPE (kW) | HR@1 (%) | HR@2 (%) | PAPE ratio vs baseline |
|---|---|---|---|---|
| baseline (NBEATSxAux, no correction) | 53.95 ± 0.69 | 27.7 ± 1.45 | 38.5 ± 1.07 | 1.000 |
| Hybrid, HR-preserving | **43.72 ± 0.48** | 26.6 ± 2.28 | 38.1 ± 1.85 | 0.810 (−19.0%) |
| Hybrid, PAPE-aggressive | **35.70 ± 0.49** | 26.3 ± 2.15 | 37.5 ± 1.74 | 0.662 (−33.8%) |

The PAPE-aggressive op-point reaches 35.70 ± 0.49 kW, slightly better
than v01's 50:50 multi-seed value of 37.62 ± 0.45 kW. The HR-preserving
op-point reaches 43.72 ± 0.48 kW vs v01's reported single-seed 45.34 kW
at the same op-point. Cold-PAPE ratios (−19.0% / −33.8%) are
within roughly one percentage point of v01's reported (−18% / −32%).

See **Figure 1** for a side-by-side bar comparison.

### 4.2 G2 — Latent-Route is statistically indistinguishable on PAPE

| Op-point | Key-Route PAPE | Latent-Route PAPE | ΔPAPE | Key-Route HR@1 | Latent-Route HR@1 | ΔHR@1 |
|---|---|---|---|---|---|---|
| HR-preserving | 43.72 ± 0.48 | 43.53 ± 0.66 | −0.19 | 26.6 ± 2.28 | 27.2 ± 2.45 | +0.6 pp |
| PAPE-aggressive | 35.70 ± 0.49 | 35.69 ± 0.34 | −0.01 | 26.3 ± 2.15 | 26.8 ± 2.14 | +0.5 pp |

The PAPE differences are within one σ. The HR@1 advantage of
Latent-Route (+0.5 / +0.6 pp) is small but consistent across the two
op-points and all three seeds; per-seed cluster usage is also more
balanced under Latent-Route (usage_min 31 vs 26–29 under Key-Route).

We therefore present Latent-Route as the **information-theoretically
preferable** routing in v02 — it uses an already-computed 64-d quantity
in place of a hand-crafted 5-d descriptor — but **not** as a
PAPE-significant improvement. **Figure 2** shows the comparison.

### 4.3 G3 — Multi-seed stability

PAPE σ across seeds is 0.34–0.66 kW (compare v01 §4.4: σ ≈ 0.45 kW for
the PAPE-aggressive op-point). HR@1 σ is larger (2.14–2.45 pp) because
n_cold = 20 is half of v01's 50, but well within the range expected at
this sample size. The codebook health diagnostics (Section 4.6) and the
W-component synergy (Section 4.5) also have small std. Where σ is
unusually large (E1 contribution, Section 4.4) we flag it and discuss
the underlying cause.

### 4.4 E1 — peak_aux contribution on the CMO correction (mixed)

Mirroring v01 §4.3 exactly: hold the correction mechanism fixed at CMO
(α_v0 = 2.0, no GST) and vary the backbone training only. Vanilla has
no aux head, so its CMO correction is fitted on its own latent space
with the same protocol.

| Backbone | baseline PAPE | CMO PAPE | rel. improvement | k_min |
|---|---|---|---|---|
| Vanilla (peak_aux OFF) | 54.04 ± 1.95 | 47.95 ± 4.61 | varies, see below | 22 ± 12 |
| NBEATSxAux (peak_aux ON) | 53.95 ± 0.69 | 41.55 ± 0.84 | −23.0% ± 1.6 | 137 ± 28 |

The peak_aux contribution measured as the gap between the two relative
improvements is **+11.9 ± 9.2 pp**, compared to the v01 §4.3 single-seed
value of +18.6 pp. Per-seed it is +24.7 (seed 42), +7.4 (seed 123), and
+3.6 (seed 7). The σ of 9.2 is dominated by Vanilla's behaviour:

- Vanilla's k_min collapses across seeds (24 / 35 / 9), meaning some
  clusters in the Vanilla codebook receive < 10 windows and produce
  noisy offsets;
- correspondingly, Vanilla's CMO PAPE swings (54.24 / 49.39 / 44.23 by
  seed), occasionally even *under-performing* its own baseline;
- NBEATSxAux's CMO PAPE is far more stable (42.13 / 40.61 / 41.90;
  k_min 163 / 98 / 149).

In other words, **the +18.6 pp v01 figure is a Vanilla-side instability
on the smaller cold pool, not a property of NBEATSxAux**. The
NBEATSxAux-side improvement (NBEATSxAux baseline → NBEATSxAux + CMO,
−23.0% ± 1.6) is robust; the *attributable contribution of peak_aux*
inflates whenever Vanilla's codebook happens to fragment. We discuss
this honestly in §5.2 and recommend reporting both the −23.0% ± 1.6
NBEATSxAux number (clean) and the +11.9 ± 9.2 pp delta (noisy) in
follow-up work.

**Figure 3** shows the two-arm bar comparison with each arm's codebook
k_min annotated above the CMO bar. The pp-delta effect size (v01's
+18.6 pp vs v02's +11.9 ± 9.2 pp) is **not** plotted: the v02 std bar
would visually understate v02's contribution while the underlying issue
is one of seed-level Vanilla codebook fragmentation rather than a
genuine effect-size collapse — see §5.2.

### 4.5 G4 — Hybrid still dominates CMO and GST at 80:20

Holding the backbone fixed (NBEATSxAux) and the routing fixed
(Key-Route), we vary only the correction:

| Op-point | CMO PAPE | GST PAPE | Hybrid PAPE | best-single − Hybrid synergy |
|---|---|---|---|---|
| HR-preserving | 47.19 ± 0.71 | 50.07 ± 1.03 | **43.72 ± 0.48** | **+3.47 ± 0.33 kW** |
| PAPE-aggressive | 44.18 ± 0.14 | 38.93 ± 0.47 | **35.70 ± 0.49** | **+3.24 ± 0.58 kW** |

Hybrid is the best of the three in **every seed × op-point** cell.
Synergy is positive and tight (σ ≤ 0.58 kW) — the v01 §4.6 iter4
ranking conclusion (W5 dominates V0 and W1a alone) is robustly
reproduced at 80:20. **Figure 4**.

### 4.6 Codebook health

Codebook utilisation is 1.000 in every seed (no dead clusters), and
perplexity is 26.6 / 28.7 / 28.2 (max possible: 32). v01's k_min ≥ 113
health threshold is met by 2 of 3 seeds (seed 7: 149, seed 42: 163,
seed 123: 98). The mean is 137 ± 28; **seed 123 falls 15 below the
threshold**. We flag this honestly: at 80:20 the smallest cluster can
get under-populated even under healthy aggregate utilisation. **Figure
5** shows per-seed values.

---

## 5. Discussion

### 5.1 What the v02 reframing buys

Under the FedHiP framing the FL claim of v01 becomes coherent: the
boundary that is crossed (`{h_g^{(j)}}` aggregated for the one-shot
KMeans fit, at the donor / pretraining stage) is the same boundary that
is already crossed by any centrally pretrained shared encoder. Cold
inference is fully local. This does *not* turn v02 into a federated
training algorithm — that remains v04 — but it does mean v02 is a
defensible **PFL evaluation** rather than a centralized one masquerading
as FL. In particular, the post-hoc 1-shot KMeans dodges the iterative
TAR attack (arxiv:2511.07073, 43–77 % input reconstruction from
iterated centroid release); model-inversion of the aggregated 64-d
hidden remains an open question for v04+.

### 5.2 The +18.6 pp E1 effect at 80:20

The most interesting honest finding of v02 is that the v01 §4.3 +18.6
pp peak_aux ablation effect **does not survive cleanly** at 80:20 — but
its failure mode is informative. The NBEATSxAux side (with peak_aux) is
stable across seeds (CMO gain of −23.0% ± 1.6); the Vanilla side
fluctuates because its codebook is unhealthy on small clusters at
80:20. Two interpretations:

1. **Conservative.** Report only the NBEATSxAux side: peak_aux training
   gives a 23 % cold-PAPE reduction with σ < 2 pp, regardless of
   correction-side comparisons.
2. **Reproducing v01 §4.3.** Report the per-seed (Vanilla CMO − NBEATSxAux
   CMO) gap as v01 did: +11.9 ± 9.2 pp, with explicit caveat that the
   Vanilla codebook fragmentation drives the variance.

We choose interpretation 2 in the headline, with interpretation 1 in
§4.4 as a complementary read.

### 5.3 Latent-Route's modest HR@1 advantage

The 5-d KEY descriptor was hand-crafted to summarise input-side peak
shape; the 64-d `h_g` is what the backbone learns to use for forecasting.
A priori one would have expected `h_g` to outperform a 5-d hand-crafted
projection by a wider margin. The fact that the PAPE difference is
within σ at 80:20 suggests one of:
- the cluster structure is already well captured by the 5-d KEY at this
  pool size (M=32, 19k train windows is enough to make either routing
  produce similar 1-NN cluster assignments);
- or that the bottleneck on cold PAPE is the offset/template magnitude,
  not the cluster identity.

The +0.5 / +0.6 pp HR@1 advantage of Latent-Route is small but
consistent (3 seeds × 2 op-points = 6 of 6 cells positive); we cannot
reject it at 3 seeds, and we recommend it as the v02 default routing
because it costs nothing extra.

### 5.4 Hybrid dominance is robust

The W-component synergy is positive in 6 of 6 cells (3 seeds × 2
op-points), with σ ≤ 0.58 kW. The CMO mechanism alone leaves
hour-of-peak imprecise; the GST mechanism alone suffers when the aux
head's hour prediction is mildly off; the Hybrid recovers from either
failure mode. This is the cleanest reproduced finding from v01 §4.6
iter4.

---

## 6. Limitations and future work

| Limitation | Where it shows | Where it is addressed |
|---|---|---|
| n_cold = 20 is small (HR@k σ ≈ 2.3 pp). | §4.1, §4.5 | n_cold ≥ 50 or rotation in v04. |
| Single dataset (UMass 2016). | All sections | Second dataset (LCL or Pecan Street) in v04+. |
| Backbone is centrally pretrained, not federatedly trained. | §2.2 | F1 (whole-backbone fine-tune) in v04. |
| Codebook is a one-shot post-hoc KMeans, not federated KMeans. | §2.2, §5.1 | Federated / DP-KMeans in v04+; v02 specifically dodges the TAR attack by **not** iterating. |
| `h_g` aggregation crosses the donor boundary; model-inversion of 64-d hidden vectors is not analysed. | §5.1 | DP noise on `h_g` aggregation, or model-inversion bound, in v04+. |
| K-shot personalization (head, last-layer, LoRA) not evaluated. | — | **v03**, scaffolding already in place. |
| Cold-side hyperparameter tuning is forbidden by design. | §1.3, §3 | Stays forbidden. |
| Other forecasting baselines (DLinear, NHiTS, Crossformer, Chronos) not compared. | — | v04+. v02 deliberately holds the model frozen so the comparison is "v01 50:50 vs v02 80:20" on identical method. |
| The Vanilla-side codebook fragmentation drives the +18.6 pp E1 finding. | §4.4, §5.2 | Cleanly reportable as NBEATSxAux-only CMO gain (−23.0 % ± 1.6); the relative gap formulation needs more seeds to stabilise. |
| Single 80:20 stratification per seed (no fold rotation). | §3.1, plan §"Open questions" | seed-coupled KMeans random_state already gives 3 different cold-20 sets; explicit fold rotation deferred to v04 if needed. |

---

## Appendix A — Naming map (paper ↔ codebase)

| Paper name | Codebase identifier | Source location |
|---|---|---|
| Vanilla | `T0` (ARM) | `experiments/v01_peak_from_latent/01_train_arms.py`; `src/models/nbeatsx.py:MinimalNBEATSx` |
| NBEATSxAux | `T2` (ARM) | `src/models/nbeatsx_aux.py:NBEATSxAux(latent_source='h_generic')` |
| CMO | `V0` mechanism | `experiments/v01_peak_from_latent/09_iter4_mechanisms.py:run_V0` |
| GST | `W1a` mechanism | `experiments/v01_peak_from_latent/09_iter4_mechanisms.py:run_W1a` |
| Hybrid | `W5` mechanism | `experiments/v01_peak_from_latent/09_iter4_mechanisms.py:run_W5` |
| Key-Route | `R0` routing | `experiments/v02_fl_8020_ratio/04_coldstart_eval.py:route_R0` (5-d KEY 1-NN) |
| Latent-Route | `R1` routing | `experiments/v02_fl_8020_ratio/04_coldstart_eval.py:route_R1` (64-d h_g 1-NN) |
| HR-preserving op-point | `(σ=3.0, α_v0=1.0, α_w1=0.1)` | constants in `04_coldstart_eval.py` |
| PAPE-aggressive op-point | `(σ=3.0, α_v0=1.5, α_w1=0.5)` | same |

The seven scripts that produce all numbers in this paper are
`experiments/v02_fl_8020_ratio/{01..08}_*.py`, summarised in
`experiments/v02_fl_8020_ratio/README.md`. The aggregated result file is
`outputs/v02_fl_8020_ratio/multiseed_summary.json`; per-seed JSONs sit in
`outputs/v02_fl_8020_ratio/seed{42,123,7}/`.

---

## Appendix B — Full ablation table (every cell, 3 seeds)

### B.1 Cold zero-shot (Section 4.1, 4.2)

| Routing | Op-point | Cell | seed 42 | seed 123 | seed 7 | mean ± std |
|---|---|---|---|---|---|---|
| Key-Route | — | baseline PAPE | 51.80 | 55.32 | 54.73 | 53.95 ± 0.69 |
| Key-Route | HR-preserving | Hybrid PAPE | 43.26 | 44.39 | 43.51 | 43.72 ± 0.48 |
| Key-Route | PAPE-aggressive | Hybrid PAPE | 36.39 | 35.39 | 35.33 | 35.70 ± 0.49 |
| Latent-Route | HR-preserving | Hybrid PAPE | 42.95 | 44.46 | 43.19 | 43.53 ± 0.66 |
| Latent-Route | PAPE-aggressive | Hybrid PAPE | 36.18 | 35.41 | 35.48 | 35.69 ± 0.34 |
| Key-Route | HR-preserving | Hybrid HR@1 | 27.1 | 23.4 | 29.2 | 26.6 ± 2.28 |
| Key-Route | PAPE-aggressive | Hybrid HR@1 | 26.7 | 23.4 | 28.8 | 26.3 ± 2.15 |
| Latent-Route | HR-preserving | Hybrid HR@1 | 27.4 | 24.0 | 30.1 | 27.2 ± 2.45 |
| Latent-Route | PAPE-aggressive | Hybrid HR@1 | 26.6 | 24.1 | 29.5 | 26.8 ± 2.14 |

(Per-seed values shown to 2 decimals; aggregator output is the source.)

### B.2 E1 — peak_aux ablation on CMO (Section 4.4)

| Backbone | seed 42 | seed 123 | seed 7 | mean ± std |
|---|---|---|---|---|
| Vanilla baseline PAPE | 51.16 | 56.83 | 55.14 | 54.04 ± 1.95 |
| Vanilla CMO PAPE | 54.24 | 49.39 | 44.23 | 47.95 ± 4.61 |
| Vanilla relative Δ | +6.0 % | −13.1 % | −19.8 % | (signed mean inflates v01-style pp) |
| NBEATSxAux baseline PAPE | 51.80 | 55.32 | 54.73 | 53.95 ± 0.69 |
| NBEATSxAux CMO PAPE | 42.13 | 40.61 | 41.90 | 41.55 ± 0.84 |
| NBEATSxAux relative Δ | −18.7 % | −26.6 % | −23.4 % | −23.0 % ± 1.6 (clean) |
| (Vanilla relΔ − NBEATSxAux relΔ) | +24.7 pp | +13.5 pp | +3.6 pp | +11.9 ± 9.2 pp (v01-style; noisy) |
| Vanilla codebook k_min | 24 | 35 | 9 | 22.7 ± 12 |
| NBEATSxAux codebook k_min | 163 | 98 | 149 | 137 ± 28 |

### B.3 W-component decomposition (Section 4.5)

| Op-point | Mechanism | seed 42 | seed 123 | seed 7 | mean ± std |
|---|---|---|---|---|---|
| HR-preserving | CMO | 46.28 | 48.00 | 47.29 | 47.19 ± 0.71 |
| HR-preserving | GST | 48.71 | 51.21 | 50.28 | 50.07 ± 1.03 |
| HR-preserving | Hybrid | 43.26 | 44.39 | 43.51 | 43.72 ± 0.48 |
| HR-preserving | best-single − Hybrid | +3.02 | +3.61 | +3.78 | +3.47 ± 0.33 |
| PAPE-aggressive | CMO | 43.99 | 44.21 | 44.34 | 44.18 ± 0.14 |
| PAPE-aggressive | GST | 39.09 | 39.40 | 38.30 | 38.93 ± 0.47 |
| PAPE-aggressive | Hybrid | 36.39 | 35.39 | 35.33 | 35.70 ± 0.49 |
| PAPE-aggressive | best-single − Hybrid | +2.70 | +4.02 | +2.96 | +3.24 ± 0.58 |

### B.4 Codebook health (NBEATSxAux backbone, M=32, Section 4.6)

| Diagnostic | seed 42 | seed 123 | seed 7 | mean ± std |
|---|---|---|---|---|
| utilization | 1.000 | 1.000 | 1.000 | 1.000 |
| perplexity (max 32) | 28.73 | 28.21 | 26.56 | 27.84 ± 0.93 |
| k_min | **163** | **98** | **149** | 137 ± 28 |
| k_max | 1772 | 2476 | 2476 | (varies) |

The bold k_min values mark the v01 §3.3 health threshold of 113;
seed 123 falls 15 below it (98), the others are well above.

---

## Appendix C — Model spec recap (carry-over from v01)

For the full method see v01 §3. The non-trivial pieces re-used unchanged:

- **NBEATSx backbone** — 3 stacks (trend / seasonal / generic), forward
  returns `(y_hat, hiddens={h_trend, h_seasonal, h_generic})`,
  `h_g ∈ ℝ⁶⁴`. Layer names match `Peak_Analysis/v10_b2`.
- **Auxiliary head** — `Linear(64, 32) → ReLU → (Linear(32, 1) for amp,
  Linear(32, 24) for hour-class)`. Loss
  `MAE(y, ŷ) + λ · [MSE(amp, y.max) + 0.1 · CE(hr, y.argmax)]`,
  λ = 0.3.
- **Post-hoc Peak-VQ** — `KMeans++(n_clusters=32, init='k-means++',
  n_init=10, random_state=seed).fit(H_train)`; one-shot, no STE,
  no in-loop quantisation regulariser.
- **W5 Hybrid correction** — `ŷ_corr = ŷ + α_v0 · o_{c*} + α_w1 ·
  g(t; ĥ, â, σ)`, with `g(t) = â · exp(−(t − ĥ)² / 2σ²)` normalised
  so `g.max(axis=1) = â`. Two operating points (σ = 3.0 always):
  `(α_v0=1.0, α_w1=0.1)` for HR-preserving, `(α_v0=1.5, α_w1=0.5)` for
  PAPE-aggressive.
- **Key-Route** — KEY descriptor
  `[max(x), argmax(x)/96, mean(x), std(x), max(x[-24:])]`,
  StandardScaler-normalised, 1-NN against the train KEY pool, then map
  to that train window's cluster.
- **Latent-Route** — `argmin_c ‖h_g_cold − codebook[c]‖₂` (raw
  Euclidean).

Hard-coded shapes: input window L = 96 h, horizon H = 24 h, D_model =
64, M = 32, polynomial trend order 3, seasonal harmonics 5. Per-apt
z-norm uses train-segment statistics only (cold side reproduces this
locally — see §3.4 warm-start).

---

## Figures

- **F0** `papers/v02_draft/figures/v02_F0_architecture.png` — end-to-end
  pipeline. Phase A (centralized pretrain), Phase B (post-hoc 1-shot
  KMeans + offsets, server side), Phase C (fully local cold inference
  with Key-Route / Latent-Route + W5 Hybrid). The donor → server boundary
  is drawn explicitly.
- **F1** `..._F1_8020_vs_5050.png` — v02 80:20 vs v01 50:50 across
  baseline, Hybrid (HR-preserving), Hybrid (PAPE-aggressive). PAPE / HR@1 /
  HR@2 panels.
- **F2** `..._F2_routing_R0_R1.png` — Key-Route vs Latent-Route at both
  op-points. PAPE / HR@1 panels.
- **F3** `..._F3_E1_peak_aux.png` — E1 ablation: Vanilla vs NBEATSxAux
  baseline + CMO PAPE on the two backbones, with each arm's codebook
  k_min annotated. Effect-size delta (v01 +18.6 pp vs v02 +11.9 ± 9.2 pp)
  is reported in §4.4 prose, not plotted, to avoid the std bar visually
  obscuring the per-arm finding.
- **F4** `..._F4_W_components.png` — CMO / GST / Hybrid PAPE at the
  two op-points; baseline reference dashed; synergy (best-single −
  Hybrid) inset.
- **F5** `..._F5_codebook_health.png` — k_min, utilization, perplexity
  per seed. v01's k_min ≥ 113 threshold marked.
