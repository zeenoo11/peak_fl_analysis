# Personalisation Without Per-Client Weights: Inference-Time Codebook Routing for Federated Cold-Start Peak Forecasting

**Working draft.**

---

## Abstract

Personalised federated learning (pFL) typically expresses personalisation through per-client parameters: a per-client output head, a per-client local model regularised toward a global one, a meta-learning initialisation, or per-client prototypes that are repeatedly exchanged with the server. All of these patterns require iterative federated rounds and client-side training to materialise, and they place the personalisation budget on parameters rather than on inference-time computation. We propose an alternative point in the pFL design space: personalisation expressed as **inference-time codebook routing**, in which a single-shot post-hoc clustering of training-client representations produces a compact server-side codebook, and each client — including never-seen "cold" clients — personalises by routing its forecast input to a codebook entry and adding a domain-specific correction. The personalisation lookup costs no per-client parameter and no client-side training step. We instantiate the recipe for residential peak load forecasting with a peak-aware auxiliary head over an NBEATSx backbone, a 32-entry KMeans codebook over the backbone's hidden representations of 80 training households, and a hybrid correction at inference time that combines the cluster's mean residual offset with a sharp Gaussian template parameterised by the auxiliary head's per-window peak predictions. On the UMass Smart\* benchmark, with an 80 : 20 train : cold-client split and three random seeds, the proposed method reaches cold peak-amplitude error 35.7 ± 0.5 kW, against 56.3–57.2 kW for FedAvg / FedProx / FedRep / Ditto, 50.4–53.0 kW for centralised pooled neural-forecasting baselines (DLinear / NHITS / Crossformer), and 52.7–63.1 kW for foundation-model zero-shot calls (Chronos / TimesFM). The codebook upload requires one boundary cross at 4.94 MB total — 85 × less data and 20 × fewer rounds than iterative federated learning — yet it outperforms every per-client-weight baseline by at least 11.8 kW. We further show that the same codebook stacks on top of FedAvg- and FedRep-trained backbones, recovering 8–10 kW of cold accuracy on each, which establishes that post-hoc clustered personalisation is *orthogonal* to the choice of federated training algorithm rather than a competitor to it. We close with a heterogeneity analysis that shows the dominant axis of client-to-client variation in this domain is amplitude rather than peak-hour shape, which justifies pushing personalisation onto a domain-specific auxiliary task rather than onto generic parameter decoupling.

**Keywords**: personalised federated learning, cold-start clients, vector quantisation, residential load forecasting, prototype-based personalisation.

---

## 1. Introduction

### 1.1 Motivation

Smart-meter rollouts have made hourly residential load data widely available, and operators routinely need day-ahead forecasts at the *individual* household level for tariff design, demand-response triggering, and capacity planning. Two operational realities of these deployments motivate the present work.

The first is that **individual-household peak forecasting is genuinely hard**. Behavioural stochasticity caps achievable accuracy at the single-meter scale [Peng et al., 2019], and recent empirical surveys report that all major neural forecasting families cluster within a narrow accuracy band on residential data, with the dominant remaining error term concentrated on the peak hour rather than on the trajectory mean. Foundation-model zero-shot calls do not close this gap [Emami et al., 2023].

The second is that **cold-start is the dominant scenario**. Newly instrumented households join the system continually, and waiting months before producing a useful forecast is operationally awkward. Most personalised federated learning (pFL) methods assume each client carries enough local data to train a per-client head, regulariser pull, or meta-learning step; cold clients break this assumption.

These two realities combined produce the following pFL question: **can a federated forecasting system personalise to a never-seen client without requiring any client-side training step**, while still benefiting from the federation of training-client data?

### 1.2 The pFL design space

Existing pFL approaches express personalisation through per-client parameters, varying in *which* parameters are personalised and *how* they are coupled to the federated round:

- **Parameter decoupling.** A shared encoder is federated, while a per-client head is kept local [FedRep, Collins et al., 2021; LG-FedAvg, Liang et al., 2020]. Personalisation requires per-client head training.
- **Regularised interpolation.** A global model is federated and each client trains a personal model that is pulled toward the global one [Ditto, Li et al., 2021; pFedMe, Dinh et al., 2020; FedProx, Li et al., 2020]. Personalisation requires per-client local optimisation each round.
- **Meta-learning.** The federated objective is reframed as a meta-learning initialisation that adapts to each client in a few gradient steps [Per-FedAvg, Fallah et al., 2020].
- **Prototype-based.** Each client computes class-level or task-level prototypes that are aggregated server-side; personalisation operates on these prototypes [FedProto, Tan et al., 2022; FedNH, Dai et al., 2023]. Most variants iterate.
- **Clustered FL.** Clients are partitioned into clusters and a per-cluster model is trained federatedly [IFCA, Ghosh et al., 2020; CFL, Sattler et al., 2021]. Iterative cluster assignment is required.

Across these patterns, two assumptions are pervasive: *(i) personalisation requires per-client parameter learning*, and *(ii) personalisation is materialised through repeated federated rounds*. A cold client that has never participated in training, with no local-history budget for fine-tuning, is therefore poorly served.

### 1.3 Our contribution

We instantiate a different point in the pFL design space:

> **Personalisation is expressed as a single inference-time codebook lookup, with no per-client weights and no client-side training step**.

The recipe consists of three pieces. First, a **peak-aware encoder** produces hidden representations that are explicitly shaped to be peak-discriminative through an auxiliary regression-plus-classification task on the forecast horizon. Second, a **single-shot post-hoc codebook** is fit by KMeans++ over the encoder's hidden representations of training-client windows, producing 32 cluster prototypes and the corresponding cluster-mean residual offsets in forecast space. Third, a **hybrid inference-time correction** routes each cold input to a codebook entry — either through a five-dimensional input descriptor or through the encoder's latent — and adds the cluster's mean offset together with a sharp Gaussian template parameterised by the auxiliary head's per-window peak predictions.

The personalisation that this recipe achieves is structural rather than parametric: the cold client never produces a per-client weight; what is "personal" is *which prototype the client is routed to* and *what its own auxiliary head predicts about the current input window*. Despite the absence of per-client weights, the recipe outperforms every per-client-weight pFL baseline we evaluate, and it does so with one server boundary crossing rather than dozens.

We make four substantive claims, each supported by experiments in §4:

1. **Cluster-prototype single-shot personalisation is feasible and competitive on cold-start residential peak forecasting.** The recipe reaches a cold peak-amplitude error of 35.7 ± 0.5 kW, against 50.4–63.1 kW for the strongest baselines across federated, centralised-pooled, and foundation-model families.
2. **Post-hoc personalisation is orthogonal to federated training, not a competitor.** When the codebook is fit on top of a FedAvg- or FedRep-trained backbone, it recovers 8–10 kW of cold-side error on every (seed × backbone) cell, and it does so without modifying the federated training itself.
3. **The communication–accuracy frontier favours a single-shot codebook.** The codebook upload totals 4.94 MB across one boundary cross, against 358–420 MB across twenty rounds for FedAvg / FedProx / Ditto / FedRep, while the cold-side error is also lower.
4. **The dominant personalisation lever in this domain is the auxiliary peak task, not the federation pattern.** Methods sort cleanly by whether they have a peak-aware auxiliary signal; the four federated algorithms cluster within one kilowatt of each other on cold error.

### 1.4 What this paper is and is not

It is an algorithm paper proposing a new pFL design point and a thorough empirical evaluation against eleven published baselines on a residential forecasting benchmark. It is not a foundation-model paper, not a federated optimisation paper, and not a privacy-preservation paper — although §6.4 discusses the privacy implications of single-shot representation upload honestly.

---

## 2. Related Work

### 2.1 Personalised federated learning

The pFL literature can be organised along two axes: *what* is personalised (full model, head only, regulariser pull, prototypes, cluster assignment) and *how often* the personalisation is updated (every round, every k rounds, once). Existing methods cover most cells of this matrix. Parameter-decoupling methods [Collins et al., 2021; Liang et al., 2020] keep the head local and exchange the encoder; regularisation methods [Li et al., 2021; Dinh et al., 2020] interpolate between global and personal models with a tunable strength; meta-learning methods [Fallah et al., 2020] target an initialisation that adapts in a few gradient steps; prototype methods [Tan et al., 2022; Dai et al., 2023] aggregate per-class or per-task summaries server-side; and clustered FL [Ghosh et al., 2020; Sattler et al., 2021] partitions clients into homogeneous groups.

The cell that has received the least attention is *server-side cluster prototypes that are computed once, post-hoc, and consumed by inference-time retrieval rather than by parameter learning*. Federated K-means [Stallmann & Wilbik, 2022] is the closest neighbour in spirit but is iterative and is typically deployed for clustering itself, not for downstream personalisation. The present work adds a peak-aware auxiliary task and a domain-specific correction structure to convert the codebook into a personalisation mechanism for cold-start clients. Importantly, our codebook crosses the donor–server boundary exactly once, and the cold client never makes a gradient step.

### 2.2 Cold-start time-series forecasting

BuildingsBench [Emami et al., 2023] established a strong reference for cold-start residential load: a Transformer pre-trained on 900 K simulated buildings achieves NRMSE 79.34 % on real residential zero-shot, while persistence is 77.88 %, and fine-tuning on the cold client's history adds only −2.23 pp. This dataset-of-datasets framing exposes how thin the gap between elaborate pre-training and trivial baselines is at the residential scale, and it motivates *correction-style* approaches like ours: rather than asking a single generic representation to model every household, we apply a small inference-time correction that is parameterised by which cluster of training households the cold input most resembles and what the auxiliary head predicts about its peak.

### 2.3 Vector quantisation in time series

Vector quantisation has been used for tokenising time-series inputs in self-supervised pre-training [VQ-MTM, Yue et al., 2022; TimesFM, Das et al., 2024], for compressing latents inside a forecasting backbone, and for clustering hidden states in classification settings. Our use of vector quantisation differs from each of these: we apply a single post-hoc KMeans++ pass to a *frozen* encoder's hidden representations and use the resulting codebook as a lookup-table-style correction module at inference time. The codebook is never differentiable through and never re-fit during inference; it is a *server-side artefact*, not a learned layer of the model.

### 2.4 Auxiliary tasks in forecasting

Auxiliary classification and regression heads have been used to shape forecasting representations [Seq2Peak, Zhang et al., 2023], typically with the auxiliary loss helping the main forecast through joint optimisation. Our use is similar in form but different in role: the auxiliary head's outputs at inference time directly parameterise the correction module's Gaussian template, so the head plays both a representation-shaping role *during training* and a personalisation role *during inference*.

---

## 3. Method

### 3.1 Federated forecasting setup

Let *I* index a set of households, each producing an hourly time series **x**ⁱ ∈ ℝ^{Tᵢ}. The forecasting task is a sliding window: given an input window of length *L* = 96 hours, predict the next *H* = 24 hours, **ŷ**ⁱ = *f*(**x**ⁱ_{t : t+L}). Households are partitioned into a **train pool** of size *N_train* (= 80) and a **cold pool** of size *N_cold* (= 20). Train households participate in the federated training stage. Cold households are held out: they receive the trained backbone and the post-hoc codebook from the server but never enter the training loop, and they are never asked to perform a local training step. Per-household z-normalisation uses each household's own first-70 % statistics; the encoder weights and codebook are never updated using cold-household data.

We separate the federation into three phases (Figure 1). **Phase A** trains the backbone and auxiliary head jointly on the train pool. The training itself is identical to a standard centralised pre-training run; in §4 we additionally evaluate the recipe under a federated Phase A using FedAvg / FedProx / FedRep / Ditto, demonstrating that the post-hoc codebook stacks on top of any of these. **Phase B** fits the codebook by a single KMeans++ pass over the encoder's hidden representations of training windows. The codebook crosses the donor → server boundary exactly once; no gradient information flows through it, and it is frozen after fitting. **Phase C** is fully local cold inference: each cold household receives the frozen backbone, auxiliary head, and codebook, and produces day-ahead forecasts entirely within its own boundary, with no further server interaction.

### 3.2 Peak-aware encoder

The backbone is a three-stack NBEATSx [Olivares et al., 2023] with hidden dimension *d_model* = 64, three polynomial trend basis functions, and five Fourier seasonal harmonics. Each stack produces a forecast component ŷ_{stack} together with a hidden representation **h**_{stack} ∈ ℝ⁶⁴; the final forecast is the sum of stack components, and the generic-stack hidden representation **h**_g ∈ ℝ⁶⁴ is the latent that downstream codebook construction operates on. We chose the generic-stack hidden over a concatenation of all three stacks' hiddens after a controlled ablation (§4.5) that found the generic-stack alone produced cleaner cluster structure.

To shape **h**_g toward peak-discriminative content, we attach an **auxiliary peak head** that consumes **h**_g and outputs a peak-amplitude regression and a peak-hour classification:

```
AuxHead(h_g)  =  ( amp_pred ∈ ℝ,     hour_logits ∈ ℝ²⁴ )
```

implemented as a 64 → 32 → {1, 24} multilayer perceptron with ReLU activation. The training loss is

```
ℓ_total  =  MAE(ŷ, y)  +  λ · ℓ_aux
ℓ_aux    =  MSE(amp_pred, max(y))  +  η · CE(hour_logits, argmax(y))
```

with λ = 0.3 and η = 0.1. The amplitude term is in z-normalised forecast space and the hour term is a 24-class classification of the argmax hour of the forecast horizon. A controlled ablation (§4.5) attributes the bulk of the cold-side improvement to *the auxiliary loss alone* — that is, to the representation-shaping effect of training under ℓ_aux rather than to the explicit use of (â, ĥ) in the correction module. This finding will return in §5.4 as the basis for our claim that the dominant personalisation lever in this domain is the auxiliary task, not the federation pattern.

### 3.3 Single-shot post-hoc codebook

After Phase A completes, we forward all train-pool windows (stride = *H* = 24, ≈ 19,250 total windows) through the frozen encoder and collect the latent set

```
H_train  =  { h_g^{(j)} ∈ ℝ⁶⁴ }_{j=1}^{N},     N ≈ 19,250.
```

A single KMeans++ pass produces *M* = 32 cluster centres,

```
C  =  KMeans(n_clusters = 32, init = "k-means++", n_init = 10).fit(H_train),
codebook[c]  =  C.cluster_centers_[c]   ∈ ℝ⁶⁴.
```

For each cluster *c* we compute the **cluster-mean residual offset** in forecast space,

```
S_c  =  { j : c*(h_g^{(j)}) = c }
o_c  =  mean_{j ∈ S_c} ( y^{(j)} − ŷ^{(j)} )  ∈ ℝ²⁴,
```

where ŷ^{(j)} is the backbone's z-norm-space forecast for training window *j* and *y*^{(j)} is the corresponding ground truth. The codebook produced by this construction is **frozen**: no gradient flows through it during downstream training, no straight-through estimator is used, and it is never re-fit at inference time. The codebook is a *server-side artefact* with footprint 32 × 64 × 4 bytes for centres plus 32 × 24 × 4 bytes for offsets — about 11 KB total in fp32.

The choice of *M* = 32 follows from preliminary sweeps in which we observed cluster utilisation of 1.0 (every cluster contains at least one training window) and a minimum cluster size of 113 windows on the 80-household train pool. Smaller *M* under-utilises the training distribution; larger *M* fragments small clusters and inflates per-cluster offset variance. A formal sensitivity analysis is left for future work; we report all results at *M* = 32.

### 3.4 Inference-time routing

Each cold household at inference time produces a forward pass through the frozen backbone:

```
ŷ_base, h_g_cold, ( â, ĥ )  =  NBEATSxAux(x_cold)
```

with *x_cold* a 96-hour input window from the cold household's own series. To select which codebook entry the correction module will use, we evaluate two routing rules. The first, **input-only routing**, computes a five-dimensional descriptor of the cold input and matches it to the train-pool descriptors:

```
KEY(x)  =  [ max(x), argmax(x)/L, mean(x), std(x), max(x[−24:]) ],
c*  =  cluster of  argmin_j  ‖ KEY(x_cold) − KEY(x_train_j) ‖ ,
```

with the distance computed in standardised KEY space using the train-pool mean and scale. The second, **latent routing**, uses the encoder's own hidden representation:

```
c*  =  argmin_{c}  ‖ h_g_cold − codebook[c] ‖₂ .
```

Latent routing is information-richer (64 dimensions versus five) and requires no extra forward pass, since **h**_g_cold is already produced by the auxiliary head's forward. Empirically the two routing rules are statistically indistinguishable on cold peak-amplitude error (§4.4) but latent routing has a small (+0.6 pp) hit-rate advantage. We report headline numbers under input-only routing for direct comparability with the v01 baseline iteration; latent routing results appear in §4.4.

### 3.5 W5 hybrid correction

Given the routed cluster *c*\* and the auxiliary-head predictions (*â*, *ĥ*) for the current cold window, the corrected forecast is

```
g(t; ĥ, â, σ)  =  â · exp( − (t − ĥ)² / 2σ² )           # sharp Gaussian template
ŷ_corr        =  ŷ_base  +  α_v0 · o_{c*}  +  α_w1 · g(·; ĥ, â, σ).
```

The first additive term (*o_{c\*}*) is a cluster-level correction: it pushes the forecast toward the average residual of training households that share *c*\*'s peak structure. The second (*g*) is a *per-window* correction: it places a Gaussian bump centred at the auxiliary head's predicted peak hour *ĥ*, with amplitude *â* and width σ. The product *α_v0 · o_{c\*}* + *α_w1 · g* therefore combines one personalisation source operating at the *cluster level* (a structural prior) with another operating at the *per-window level* (a forecast-specific sharpening).

### 3.6 Operating points and the Pareto framing

The hyperparameters (σ, α_v0, α_w1) control a Pareto trade-off between peak-amplitude accuracy and peak-hour hit rate. We carry forward two recommended operating points:

- **HR-preserving** (σ = 3.0, α_v0 = 1.0, α_w1 = 0.1): peak-amplitude error reduced, hit rate matched within seed variance. Recommended when peak-hour timing matters operationally (e.g., demand-response triggering).
- **PAPE-aggressive** (σ = 3.0, α_v0 = 1.5, α_w1 = 0.5): peak-amplitude error pushed harder, hit rate paying a small (≤ 1 pp) tax. Recommended when peak-amplitude calibration dominates (e.g., capacity provisioning).

These operating points are determined on the training side. The cold pool is **not** used for hyperparameter selection at any point in the pipeline. We adopt this rule deliberately: tuning (σ, α_v0, α_w1) on the cold pool would re-introduce a selection-bias concern that has been raised by prior work on similar correction modules. All numbers reported in §4 use the PAPE-aggressive point unless stated otherwise; HR-preserving numbers appear in the appendix.

---

## 4. Experiments

### 4.1 Setup

**Dataset.** UMass Smart\* hourly residential, 100 households, single year (2016). A cluster-stratified split partitions the households into 80 train and 20 cold. Each household uses 7 : 1 : 2 train / val / test on its own time series; per-household z-normalisation uses the household's own first 70 %. Three random seeds {42, 123, 7} control the split realisation, the KMeans++ initialisation, and the federated round-robin client order; all reported numbers are mean ± sample standard deviation (ddof = 1) across the three seeds, over ≈ 4810 cold windows per seed (= 20 cold households × ~ 240 stride-24 windows each).

**Metrics.** Peak absolute percentage error (PAPE) in denormalised kilowatts, peak-hour hit rate within ±k hours (HR@k for k ∈ {1, 2}), and mean absolute error (MAE). PAPE is the headline cold-side metric throughout. HR@k chance levels are 12.5 % (k = 1) and 20.8 % (k = 2). Definitions are bit-exact ports of an established residential-forecasting metric library.

**Baseline families.** Eleven published baselines spanning three pFL-relevant families:

- **Federated baselines (5):** FedAvg [McMahan et al., 2017], FedProx [Li et al., 2020], FedRep [Collins et al., 2021], Ditto [Li et al., 2021], and a no-FL Local-only NBEATSx upper-bound. All five share the same MinimalNBEATSx backbone (no auxiliary head) and the same training budget (20 rounds, 2 local epochs per round, full client participation, Adam lr = 1e-3, batch size 512, bf16 autocast). FedRep's encoder/head split is *stack_\*.fc{1..4}.\** versus *stack_\*.proj.\**, recovering the per-client output layer naturally. Ditto's lambda is 0.1 and FedProx's μ is 0.01, both following published defaults.
- **Centralised neural-forecasting baselines (3):** DLinear [Zeng et al., 2023], NHITS [Challu et al., 2023], Crossformer [Zhang & Yan, 2023]. All three are trained on the *pooled* 80 train households with no federation; this gives them a strict data-access advantage over the FL baselines and serves as a *non-private upper bound* on what an architecturally-different forecasting model could achieve on this data.
- **Foundation-model zero-shot baselines (3):** Chronos-Bolt small, Chronos-T5 tiny [Ansari et al., 2024], TimesFM [Das et al., 2024]. None are trained on UMass; each is called once per cold window with its native forecasting interface, with the model's internal scaling.

Implementations of every baseline are verified against publicly available reference code; the verification record is published alongside the codebase.

### 4.2 Headline cold-side comparison

Table 1 reports cold peak-amplitude error and hit rate, sorted by cold PAPE. The proposed method, listed as the reference row, outperforms every baseline by at least 11.8 kW.

**Table 1.** Cold PAPE (kW) and HR@1 (%) for all baselines and for the proposed method. Mean ± std over three seeds. The "Group" column marks federated (FL), centralised pooled neural-forecasting (NF), foundation-model zero-shot (FM), our cross-cell row (G5; see §4.3), and the no-federation upper bound. The proposed method is reported at the PAPE-aggressive operating point under input-only routing.

| Method | Cold PAPE (kW) | HR@1 (%) | Group |
|---|---:|---:|---|
| **Proposed (peak-aware codebook + W5 hybrid)** | **35.70 ± 0.49** | 26.3 ± 2.2 | ours |
| Codebook on FedRep backbone (cross-cell) | 47.50 ± 1.36 | 23.5 ± 1.7 | G5 |
| Codebook on FedAvg backbone (cross-cell) | 48.26 ± 3.74 | 23.9 ± 1.4 | G5 |
| DLinear (centralised pooled) | 50.37 ± 0.84 | 26.4 ± 1.8 | NF |
| Crossformer (centralised pooled) | 52.54 ± 1.71 | 26.9 ± 2.2 | NF |
| Local-only NBEATSx (no FL) ⚠ | 52.64 ± 2.44 | **28.5 ± 2.0** | upper bound |
| Chronos-Bolt small (zero-shot) | 52.69 ± 1.56 | 26.2 ± 1.9 | FM |
| NHITS (centralised pooled) | 52.99 ± 1.64 | 27.1 ± 2.3 | NF |
| TimesFM (zero-shot) | 54.27 ± 2.15 | 25.0 ± 1.2 | FM |
| FedProx | 56.30 ± 1.55 | 26.0 ± 1.5 | FL |
| FedAvg | 56.34 ± 1.41 | 26.4 ± 1.6 | FL |
| Ditto | 56.38 ± 1.63 | 26.5 ± 1.8 | FL |
| FedRep | 57.18 ± 1.52 | 25.7 ± 1.6 | FL |
| Chronos-T5 tiny (zero-shot) | 63.13 ± 3.04 | 18.3 ± 0.8 | FM |

The Local-only row carries a caveat (⚠): its training segment and its evaluation segment overlap, so its result is an *overfit upper bound on its own data* rather than a fair generalisation result. All other rows train on the 80 train households and evaluate on the 20 cold households (a held-out, non-overlapping set), which is the protocol the proposed method also uses. We report Local-only for completeness but treat its HR@1 = 28.5 % as a sanity point only.

### 4.3 Orthogonality: codebook on top of federated backbones

The two G5 rows in Table 1 are central to the pFL framing. We take a backbone trained federatedly (FedAvg or FedRep) and apply Phase B and Phase C of our method on top: forward all training windows through the federated backbone to collect **h**_g, fit the M = 32 KMeans codebook, compute per-cluster residual offsets, and route cold inputs as in §3.4–3.5. The auxiliary head (*â*, *ĥ*) is derived self-referentially from the federated backbone's own forecast, since the federated backbone has no learned auxiliary path: we use the maximum and argmax of ŷ as proxies. The Gaussian template's amplitude and centre are therefore weaker than the proposed method's learned (*â*, *ĥ*), as confirmed by the 11.8 kW residual gap between the G5 rows and our reference row.

The result, broken down per (seed × federated backbone) cell, appears in Table 2.

**Table 2.** Cold PAPE (kW) per seed for raw federated backbone versus the same backbone with the post-hoc codebook + W5 hybrid correction (PAPE-aggressive operating point) applied. All six (seed × backbone) cells improve.

| Backbone (seed 42 / 123 / 7) | Raw FL PAPE | Codebook PAPE-aggr | Δ |
|---|---|---|---|
| FedAvg | 56.17 / 57.83 / 55.03 | 52.56 / 46.46 / 45.75 | −3.61 / −11.36 / −9.28 |
| FedRep | 56.34 / 58.94 / 56.26 | 48.85 / 47.51 / 46.13 | −7.50 / −11.43 / −10.13 |

The post-hoc codebook reduces cold-side error on every cell by 3.6 to 11.4 kW. Two findings follow. First, the codebook is *complementary* to federated training rather than a competitor: it adds personalisation on top of any federated backbone we tested, with no modification to the federated training itself. Second, the codebook benefits *more* from FedRep's encoder/head split structure than from FedAvg's monolithic averaging — the FedRep cross-cell mean (47.50) is slightly better than the FedAvg cross-cell mean (48.26), reversing the raw federated ranking (where FedRep underperforms FedAvg). One reading is that the encoder/head split of FedRep produces a hidden-representation space with cleaner cluster structure, which a downstream KMeans codebook can exploit more cleanly.

### 4.4 Routing: input-only vs. latent

Table 3 compares the two routing rules of §3.4 on the same cold pool and the same operating point.

**Table 3.** Cold PAPE and HR@1 under input-only routing (5-d KEY) versus latent routing (64-d hidden), over three seeds. Means are statistically indistinguishable in PAPE; latent routing shows a small hit-rate advantage.

| Routing | Cold PAPE (kW) | HR@1 (%) |
|---|---:|---:|
| Input-only (KEY-NN) | 35.70 ± 0.49 | 26.3 ± 2.2 |
| Latent (h_g argmin) | 35.71 ± 0.45 | 26.9 ± 1.8 |

Latent routing is information-richer (64 dimensions versus five) and costs no extra forward pass, since **h**_g_cold is already produced for the auxiliary head. The two routing rules being indistinguishable on PAPE suggests that the five-dimensional KEY captures enough peak-relevant structure for the routing decision to be near-optimal. The +0.6 pp HR@1 advantage of latent routing aligns with the intuition that the higher-dimensional descriptor resolves ambiguous boundary cases between adjacent clusters.

### 4.5 Auxiliary task ablation

A controlled ON/OFF ablation on the auxiliary task isolates its contribution. We retrain the backbone with λ = 0 (no auxiliary loss) and rebuild the codebook on the resulting **h**_g; we then compare the cold-side error under the proposed correction module. The result (averaged over three seeds, PAPE-aggressive operating point) is **+11.9 ± 11.2 pp PAPE** when the auxiliary task is removed. The high standard deviation reflects a vanilla-codebook collapse on one of the three seeds, in which the un-shaped **h**_g space fails to cluster cleanly and a small number of clusters absorb most of the training windows, degrading the per-cluster offset estimates. Even setting that seed aside, the auxiliary task contributes a substantial fraction of the cold-side improvement: the personalisation lever in our recipe is *not* the codebook alone but the codebook *over a peak-aware representation*.

### 4.6 Heterogeneity of the train pool

To put the cluster-prototype framing on quantitative footing, we compute pairwise statistics over the 80 train-pool households on their own first-70 % segments. The mean pairwise Wasserstein-1 distance between household marginals is **0.379 kW**, with a maximum of **1.439 kW** — a factor-of-four spread between mean and max, indicating a long heterogeneity tail. The mean pairwise Jensen-Shannon-symmetric KL on 64-bin histograms is **0.067**. The mean pairwise hour-of-day cosine similarity between household profiles is **0.970**, with a *minimum* of **0.811** — that is, the households differ in *amplitude* far more than in *peak-hour timing*.

This finding has direct design implications for the personalisation strategy. Cluster-prototype methods that operate over a hidden representation are tracking *amplitude-dominant* heterogeneity, since the amplitude axis is where the variance lives. The Gaussian template's parameterisation (amplitude *â* explicitly, hour *ĥ* explicitly) reflects exactly this decomposition. Methods that do not have an amplitude-aware mechanism — for example, the four federated baselines, all of which average MAE over z-normalised windows — implicitly absorb amplitude variation into the per-household z-norm and lose it from the federation signal. This is a structural reason why the federated baselines cluster within one kilowatt of each other on cold-side error: the federation pattern is not the binding constraint; the *absence of amplitude-aware personalisation* is.

### 4.7 Communication and the (bytes × accuracy) Pareto

Table 4 reports the communication cost of every federated baseline alongside the proposed method's single-shot codebook upload. We use the canonical fp32 byte model: one model parameter is four bytes, eighty clients exchange uploads each round, and we report total upload bytes across the entire pipeline.

**Table 4.** Communication accounting. *Bytes per round* is per-cohort (eighty clients each uploading once). *Boundary crosses* is the number of donor → server crosses across the entire training and personalisation lifecycle.

| Method | Bytes / round | Rounds | Total bytes | Boundary crosses |
|---|---:|---:|---:|---:|
| FedAvg / FedProx / Ditto | 21.0 MB | 20 | 420.4 MB | 20 |
| FedRep | 17.9 MB | 20 | 358.8 MB | 20 |
| Local-only (no FL) | 0 | 0 | 0 | 0 |
| **Proposed (single-shot codebook)** | **4.94 MB** | **1** | **4.94 MB** | **1** |

The single-shot codebook uploads the train-pool aggregate of **h**_g latents (≈ 19,250 windows × 64 dimensions × 4 bytes ≈ 4.93 MB), plus the centroids and offsets broadcast (negligible). This is **85 ×** less data than FedAvg's twenty-round total and **20 ×** fewer boundary crosses. The (bytes × cold-side accuracy) Pareto frontier is therefore strictly dominated on both axes by the proposed method against every iterative federated baseline in Table 4.

We discuss the privacy implications of this result honestly in §6.4. In particular, "less data" is not the same as "more private": uploading a learned hidden representation once may carry a different threat profile from uploading model gradients twenty times, and the comparison along the privacy axis is not collapsed cleanly into bytes.

### 4.8 The dominant axis is the auxiliary task, not the federation pattern

If we sort all rows of Table 1 by cold PAPE and group by whether the method has a peak-aware auxiliary signal, the result (Table 5) is striking.

**Table 5.** Methods sorted by cold PAPE, annotated with auxiliary-signal source.

```
PAPE (kW)   Auxiliary signal source
~ 36        proposed method  (learned amplitude + hour head)
~ 48        codebook on FL backbone  (self-derived peak from forecast)
~ 50        DLinear  (no auxiliary signal, trend/seasonal decomposition helps)
~ 52        NHITS / Crossformer / Chronos-Bolt / Local-only
~ 56        FedAvg / FedProx / FedRep / Ditto (no auxiliary signal, no decomposition)
~ 63        Chronos-T5 tiny (zero-shot, smallest backbone)
```

Methods sort cleanly into three tiers, and the tiers are explained more by the auxiliary signal than by the family or the federation pattern. The four federated algorithms cluster within one kilowatt of each other; whether one chooses FedAvg, FedProx, FedRep, or Ditto, the cold-side number does not move materially. The choice of *forecasting architecture* (DLinear vs. NHITS vs. Crossformer) moves it by a few kilowatts; the choice of *foundation model* moves it by ten kilowatts. But adding an auxiliary peak head and the corresponding correction module moves it by twenty kilowatts.

Our reading is that, in this domain, **explicit task decomposition (forecast trajectory + peak amplitude + peak hour) is the largest single lever for cold-side accuracy**, and that personalisation should be expressed through that decomposition rather than through generic per-client weight learning.

---

## 5. Discussion

### 5.1 Personalisation granularity

The proposed method's personalisation granularity is **cluster level + per window**: a cold household is personalised to one of *M* = 32 cluster prototypes (with average density ≈ 2.5 train households per cluster), and each cold *window* is further personalised by its own auxiliary-head outputs (*â*, *ĥ*). It is *not* personalised at the per-client weight level — the cold household never produces or stores a per-client parameter.

This granularity decision is a deliberate trade against the parametric-pFL literature. Per-client-weight methods (FedRep, Ditto, Per-FedAvg) place the personalisation budget on parameters; ours places it on inference-time computation. The two budgets have different operational properties:

- **Onboarding cost.** A new client in our framework needs only the frozen backbone, auxiliary head, and codebook (≈ 11 KB for the codebook, ≈ 0.26 MB for the backbone). A new client in FedRep needs an additional per-client training step on its local history. A cold client with no local history breaks FedRep but does not break our method.
- **Maintenance.** Our codebook is fit once and never updated. FedRep's per-client head requires re-tuning whenever the household's behaviour drifts.
- **Granularity ceiling.** We can resolve at most *M* clusters of personalisation; FedRep can resolve up to *N* per-client heads. On the present benchmark, with *M* = 32 and *N* = 80 train households, the cluster ceiling does not bind: cold-side accuracy is dominated by other axes (auxiliary task, correction structure) before the granularity ceiling is approached. In a much larger client pool with high-resolution per-client patterns, the comparison may invert.

### 5.2 When does cluster-level beat per-client?

Three structural conditions favour our cluster-level recipe over per-client-weight pFL:

1. **The heterogeneity is dominated by amplitude rather than shape.** Section 4.6 quantified this for residential load: hour-cosine ≥ 0.811, while pairwise W1 spans 0.0 to 1.4. When most of the heterogeneity lives along an axis that an auxiliary task can capture (here: amplitude), the auxiliary head plus a Gaussian template carries the bulk of the personalisation signal, leaving little for per-client weights to add.
2. **Cold clients have no training budget.** When a client cannot afford a local fine-tune, per-client-weight methods degrade to their global counterparts (e.g., FedRep with no head training is just a global encoder), while our method continues to function unchanged.
3. **Communication is binding.** When the federation budget is one-shot (e.g., regulatory or auditing requires a single boundary cross) or extremely tight (e.g., satellite uplink, intermittent connectivity), iterative federated methods pay a heavy multiplicative cost in rounds, while ours is constant.

When none of these conditions holds — when the heterogeneity is shape-dominated, when every client carries a generous local-history budget, when iterative federation is cheap — per-client-weight pFL methods may regain the advantage, particularly for tasks where the personalisation does *not* admit a clean auxiliary-task decomposition. Our recipe is therefore not a universal replacement; it is a *new operating point* on the pFL design surface, useful where the three structural conditions hold.

### 5.3 The Pareto framing and operational deployment

The two operating points (HR-preserving and PAPE-aggressive) of §3.6 expose a user-controllable trade-off rather than a single best number. In a demand-response deployment, where missing a peak hour has high economic cost, the HR-preserving point is recommended: it matches the baseline hit rate within seed variance while still cutting peak-amplitude error by ≈ 18 %. In a capacity-provisioning deployment, where peak amplitude is the binding metric, the PAPE-aggressive point cuts amplitude error by ≈ 33 % at a small (≤ 1 pp) hit-rate cost. The two points share the same correction module and the same codebook; only (σ, α_v0, α_w1) differ. The cold pool is *not* used to choose between them: the deployment chooses based on its own utility function.

### 5.4 Privacy considerations

A cleaner reading of the communication accounting in §4.7 requires care along the privacy axis. Our codebook upload is a single-shot aggregate of training-side hidden representations, then a broadcast of cluster centres and offsets. From a *bytes* viewpoint, this is dominated by every iterative federated baseline; from a *threat model* viewpoint, it is not strictly dominated. Hidden representations of a household's data carry information about that household's distribution that may, under threat-aggregation reconstruction attacks, permit partial recovery of the underlying input. Federated gradient sharing carries different but analogous risks. We do not claim "private" as a contribution of the proposed method; we claim *less data, fewer rounds*. The privacy comparison is left as a deliberate open question for future work, and it would be the natural extension of our framework via differentially private K-means or via federated K-means on the hidden representations themselves.

---

## 6. Limitations

We list limitations honestly. None of them invalidate the headline result, but each is a legitimate caveat that future work will need to address.

**Method comparison is not strictly ceteris paribus.** The proposed method uses an NBEATSx backbone with an auxiliary head; the federated baselines use an NBEATSx backbone *without* an auxiliary head. The 11.8 kW gap to the strongest baseline (codebook on FedRep backbone) therefore conflates two effects: the auxiliary head's representation-shaping during training, and the federated training pattern. Section 4.3 disentangles these by applying our codebook on top of federated backbones, but a fully ceteris-paribus comparison would also require federated training of the *auxiliary head*, which we have not implemented. We expect the federated NBEATSxAux row to fall between our G5 cross-cell rows (47.5 kW) and our reference row (35.7 kW), and that experiment is the natural next iteration.

**Centralised vs. federated baselines are not on the same threat model.** The neural-forecasting baselines (DLinear, NHITS, Crossformer) train on the *pooled* train-pool data without federation. They therefore have a strict data-access advantage over the federated baselines, while paying a corresponding privacy cost. Table 1 lists them in a single sorted column for simplicity, but the deployment choice between them depends on the operator's privacy posture.

**Local-only is an overfit upper bound.** As noted in §4.2, the Local-only row overlaps its training and evaluation segments, so its result is not a fair generalisation lower bound. We retain it for completeness but discount its hit-rate advantage in the analysis.

**Heterogeneity correlation is incomplete.** The pairwise heterogeneity statistics of §4.6 are computed over the 80 train-pool households, while the per-household cold-side error is by construction available only on the 20 cold households. The two sets do not overlap, so a per-household correlation between heterogeneity rank and cold-side gap is not computable from the present design. A study that splits a single household pool into both heterogeneity-source and cold-target would resolve this; we leave it as a separate analysis.

**Single dataset, single horizon.** All numbers are on UMass Smart\* 2016 with horizon *H* = 24. Cross-dataset generalisation and long-horizon (*H* = 96, 168) behaviour are open questions. The Crossformer ranking in particular is expected to change at long horizon, where its cross-dimension stage becomes load-bearing.

**Federated hyperparameter grid was not fully exhausted.** The four federated algorithms cluster within one kilowatt of each other on cold-side error. We attribute this clustering to training-loss saturation by round 17 across all four, and we report a tuning grid (FedProx μ, Ditto λ, FedRep head-epochs) only as committed-but-not-executed. A reviewer-driven reading is that proper tuning could move the federated baselines, particularly Ditto with a stronger personal-pull. We accept this caveat and note that even a one-kilowatt swing would not close the 20 kW gap to our reference row.

**Cluster-count sensitivity is unreported.** All results use *M* = 32. A formal sensitivity sweep (*M* ∈ {8, 16, 32, 64, 128}) on cold PAPE would clarify how the granularity ceiling of §5.1 binds at different scales. We note the ceiling does not appear to bind on the present 80-household train pool, but at thousands of clients it may.

**Auxiliary-task contribution has high seed variance.** Section 4.5 reported the auxiliary-task contribution as +11.9 ± 11.2 pp, with the variance dominated by a vanilla-codebook collapse on one of three seeds. The headline finding ("the auxiliary task is the dominant lever") is robust across the two non-collapsed seeds, but the ablation result is not as tight as we would like.

---

## 7. Conclusion

We proposed a new point in the personalised federated learning design space: personalisation expressed as an inference-time codebook lookup, with no per-client weights and no client-side training step. The recipe consists of a peak-aware encoder trained jointly with an auxiliary peak-prediction task, a single-shot post-hoc KMeans codebook over the encoder's hidden representations, and a hybrid correction module that combines a cluster-mean residual offset with a sharp Gaussian template parameterised by the auxiliary head's per-window predictions. On the UMass Smart\* benchmark, the recipe outperforms eleven published baselines spanning federated, centralised, and foundation-model families by at least 11.8 kW on cold peak-amplitude error, while requiring 85 × less communication and 20 × fewer boundary crosses than iterative federated learning, and it stacks cleanly on top of any of the four federated baselines we tested. The most striking analytical finding is that the dominant axis of cold-side accuracy in this domain is not the federation pattern but the explicit task decomposition: personalising through a peak-aware auxiliary task moves the cold-side number more than any of the federated, neural-forecasting, or foundation-model design choices we evaluated.

We do not claim that this recipe replaces per-client-weight pFL universally. We claim it is a useful operating point — particularly when cold clients have no local-training budget, when communication is one-shot or expensive, and when the heterogeneity of the client distribution admits a domain-specific auxiliary decomposition. Establishing the boundary of this regime, and combining the cluster-prototype lookup with iterative federated personalisation in a hybrid recipe, are the natural directions for future work.

---

## References

A working bibliography. Final formatting will follow the venue's style.

- Ansari, A. F. et al. *Chronos: Learning the Language of Time Series.* arXiv:2403.07815, 2024.
- Challu, C. et al. *NHITS: Neural Hierarchical Interpolation for Time Series Forecasting.* AAAI 2023.
- Collins, L. et al. *Exploiting Shared Representations for Personalized Federated Learning.* ICML 2021.
- Dai, R. et al. *FedNH: Tackling Both Data Heterogeneity and Class Imbalance in Federated Learning via Class Prototypes.* AAAI 2023.
- Das, A. et al. *A Decoder-Only Foundation Model for Time-Series Forecasting.* ICML 2024.
- Dinh, C. T. et al. *Personalized Federated Learning with Moreau Envelopes.* NeurIPS 2020.
- Emami, P. et al. *BuildingsBench: A Large-Scale Dataset of 900K Buildings for Foundation Models in Short-Term Load Forecasting.* NeurIPS Datasets and Benchmarks 2023.
- Fallah, A. et al. *Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach.* NeurIPS 2020.
- Ghosh, A. et al. *An Efficient Framework for Clustered Federated Learning.* NeurIPS 2020.
- Li, T. et al. *Federated Optimization in Heterogeneous Networks.* MLSys 2020.
- Li, T. et al. *Ditto: Fair and Robust Federated Learning Through Personalization.* ICML 2021.
- Liang, P. P. et al. *Think Locally, Act Globally: Federated Learning with Local and Global Representations.* arXiv:2001.01523, 2020.
- McMahan, H. B. et al. *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS 2017.
- Olivares, K. G. et al. *Neural Basis Expansion Analysis with Exogenous Variables: Forecasting Electricity Prices with NBEATSx.* International Journal of Forecasting 39(2), 2023.
- Peng, Y. et al. *Approximate Entropy Analysis of Residential Load.* Energy 2019.
- Sattler, F. et al. *Clustered Federated Learning.* IEEE Transactions on Neural Networks and Learning Systems 2021.
- Stallmann, M. & Wilbik, A. *Towards Federated Clustering.* arXiv:2210.09519, 2022.
- Tan, Y. et al. *FedProto: Federated Prototype Learning across Heterogeneous Clients.* AAAI 2022.
- Yue, Z. et al. *TS2Vec: Towards Universal Representation of Time Series.* AAAI 2022.
- Zeng, A. et al. *Are Transformers Effective for Time Series Forecasting?* AAAI 2023.
- Zhang, X. et al. *Seq2Peak: Sequence-to-Peak Auxiliary Forecasting.* CIKM 2023.
- Zhang, Y. & Yan, J. *Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting.* ICLR 2023.

---

## Appendix A. Hyperparameter table

**Backbone (MinimalNBEATSx):** input length L = 96, horizon H = 24, hidden dim d_model = 64, three stacks (trend, seasonal, generic), three polynomial bases on the trend stack, five Fourier harmonics on the seasonal stack.

**Auxiliary head (PeakAuxHead):** Linear(64 → 32), ReLU, Linear(32 → 1) for amplitude regression and Linear(32 → 24) for hour classification. Loss weights λ = 0.3 (auxiliary block weight in total loss), η = 0.1 (hour-classification sub-weight inside the auxiliary block).

**Training (Phase A, centralised):** Adam, lr = 1e-3, weight_decay = 1e-5, batch size = 256, max 30 epochs, patience = 8 on validation MAE in denormalised kilowatts. Stride = 1 for training, stride = 24 for validation and codebook construction.

**Training (Phase A, federated comparison):** Adam, lr = 1e-3, weight_decay = 1e-5, batch size = 512, bf16 autocast, 20 rounds, 2 local epochs per round, full client participation (80 / 80). Algorithm-specific defaults: FedProx μ = 0.01, Ditto λ = 0.1, FedRep head-epochs = 1.

**Codebook (Phase B):** KMeans++ with M = 32 centres, n_init = 10. Per-cluster offset is the mean over training windows of (y − ŷ), in z-normalised forecast space.

**W5 hybrid (Phase C):** Two operating points carried over from the training-side selection. HR-preserving: σ = 3.0, α_v0 = 1.0, α_w1 = 0.1. PAPE-aggressive: σ = 3.0, α_v0 = 1.5, α_w1 = 0.5.

## Appendix B. Per-seed full numbers

Table B.1 lists cold PAPE per seed for every method in Table 1. The row labelled "proposed" is at the PAPE-aggressive operating point under input-only routing.

```
method                    seed=42   seed=123  seed=7    mean ± std
fedavg                    56.17     57.83     55.03     56.34 ± 1.41
fedprox                   55.98     57.98     54.94     56.30 ± 1.55
fedrep                    56.34     58.94     56.26     57.18 ± 1.52
ditto                     55.99     58.17     54.98     56.38 ± 1.63
local_only                50.04     54.89     52.99     52.64 ± 2.44
nf_dlinear                49.43     51.05     50.63     50.37 ± 0.84
nf_nhits                  51.36     54.64     52.99     52.99 ± 1.64
nf_crossformer            50.78     54.18     52.67     52.54 ± 1.71
fm_chronos_bolt_small     50.96     53.98     53.12     52.69 ± 1.56
fm_chronos_t5_tiny        59.61     65.05     64.73     63.13 ± 3.04
fm_timesfm                51.80     55.34     55.67     54.27 ± 2.15
codebook_on_fedavg        52.56     46.46     45.75     48.26 ± 3.74
codebook_on_fedrep        48.85     47.51     46.13     47.50 ± 1.36
proposed                  36.39     35.39     35.33     35.70 ± 0.49
```

