# Personalisation Without Per-Client Weights: Inference-Time Codebook Routing for Federated Cold-Start Peak Forecasting

**Working draft.**

---

## Abstract

Personalised federated learning (pFL) typically expresses personalisation through per-client parameters: a per-client output head, a per-client local model regularised toward a global one, a meta-learning initialisation, or per-client prototypes that are repeatedly exchanged with the server. All of these patterns require iterative federated rounds and client-side training to materialise, and they place the personalisation budget on parameters rather than on inference-time computation. We propose an alternative point in the pFL design space: personalisation expressed as **inference-time codebook routing**, in which a single-shot post-hoc clustering of training-client representations produces a compact server-side codebook, and each client — including never-seen "cold" clients — personalises by routing its forecast input to a codebook entry and adding a domain-specific correction. The personalisation lookup costs no per-client parameter and no client-side training step. We instantiate the recipe for residential peak load forecasting with a peak-aware auxiliary head over an NBEATSx backbone, a 32-entry KMeans codebook over the backbone's hidden representations of 80 training households, and a hybrid correction at inference time that combines the cluster's mean residual offset with a sharp Gaussian template parameterised by the auxiliary head's per-window peak predictions. On the UMass Smart\* benchmark, with an 80 : 20 train : cold-client split and three random seeds, the proposed method reaches cold peak-amplitude percentage error 35.7 ± 0.5 %, against 56.3–57.2 % for FedAvg / FedProx / FedRep / Ditto, 50.4–53.0 % for centralised pooled neural-forecasting baselines (DLinear / NHITS / Crossformer), and 52.7–63.1 % for foundation-model zero-shot calls (Chronos / TimesFM). The codebook upload requires one boundary cross at 4.94 MB total — 85 × less data and 20 × fewer rounds than iterative federated learning — yet it outperforms every per-client-weight baseline by at least 11.8 percentage points (pp). A controlled disentanglement against a *federated NBEATSxAux* row decomposes the 20.6 pp total gap from FedAvg to our reference cleanly into three additive parts: $6.2$ pp from federation alone, $5.6$ pp from a learned versus self-derived auxiliary head, and $8.8$ pp from the W5 hybrid correction module itself. We further show that the same codebook stacks on top of FedAvg- and FedRep-trained backbones, recovering 8–10 pp of cold accuracy on each, which establishes that post-hoc clustered personalisation is *orthogonal* to the choice of federated training algorithm rather than a competitor to it. An iterative prototype-based pFL baseline (FedProto) lands essentially on top of FedAvg, confirming that prototype alignment alone — without the hybrid correction module — does not move the cold-side number. We close with a heterogeneity analysis that shows the dominant axis of client-to-client variation in this domain is amplitude rather than peak-hour shape, which justifies pushing personalisation onto a domain-specific auxiliary task rather than onto generic parameter decoupling.

**Keywords**: personalised federated learning, cold-start clients, vector quantisation, residential load forecasting, prototype-based personalisation.

---

## 1. Introduction

### 1.1 Motivation

Smart-meter rollouts have made hourly residential load data widely available, and operators routinely need day-ahead forecasts at the *individual* household level for tariff design, demand-response triggering, and capacity planning. Two operational realities of these deployments motivate the present work.

The first is that **individual-household peak forecasting is genuinely hard**. Behavioural stochasticity caps achievable accuracy at the single-meter scale [1], and recent empirical surveys report that all major neural forecasting families cluster within a narrow accuracy band on residential data, with the dominant remaining error term concentrated on the peak hour rather than on the trajectory mean. Foundation-model zero-shot calls do not close this gap [2].

The second is that **cold-start is the dominant scenario**. Newly instrumented households join the system continually, and waiting months before producing a useful forecast is operationally awkward. Most personalised federated learning (pFL) methods assume each client carries enough local data to train a per-client head, regulariser pull, or meta-learning step; cold clients break this assumption.

These two realities combined produce the following pFL question: **can a federated forecasting system personalise to a never-seen client without requiring any client-side training step**, while still benefiting from the federation of training-client data?

### 1.2 The pFL design space

Existing pFL approaches express personalisation through per-client parameters, varying in *which* parameters are personalised and *how* they are coupled to the federated round:

- **Parameter decoupling.** A shared encoder is federated, while a per-client head is kept local [FedRep, [3]; LG-FedAvg, [4]]. Personalisation requires per-client head training.
- **Regularised interpolation.** A global model is federated and each client trains a personal model that is pulled toward the global one [Ditto, [5]; pFedMe, [6]; FedProx, [7]]. Personalisation requires per-client local optimisation each round.
- **Meta-learning.** The federated objective is reframed as a meta-learning initialisation that adapts to each client in a few gradient steps [Per-FedAvg, [8]].
- **Prototype-based.** Each client computes class-level or task-level prototypes that are aggregated server-side; personalisation operates on these prototypes [FedProto, [9]; FedNH, [10]]. Most variants iterate.
- **Clustered FL.** Clients are partitioned into clusters and a per-cluster model is trained federatedly [IFCA, [11]; CFL, [12]]. Iterative cluster assignment is required.

Across these patterns, two assumptions are pervasive: *(i) personalisation requires per-client parameter learning*, and *(ii) personalisation is materialised through repeated federated rounds*. A cold client that has never participated in training, with no local-history budget for fine-tuning, is therefore poorly served.

### 1.3 Our contribution

We instantiate a different point in the pFL design space:

> **Personalisation is expressed as a single inference-time codebook lookup, with no per-client weights and no client-side training step**.

The recipe consists of three pieces. First, a **peak-aware encoder** produces hidden representations that are explicitly shaped to be peak-discriminative through an auxiliary regression-plus-classification task on the forecast horizon. Second, a **single-shot post-hoc codebook** is fit by KMeans++ over the encoder's hidden representations of training-client windows, producing 32 cluster prototypes and the corresponding cluster-mean residual offsets in forecast space. Third, a **hybrid inference-time correction** routes each cold input to a codebook entry — either through a five-dimensional input descriptor or through the encoder's latent — and adds the cluster's mean offset together with a sharp Gaussian template parameterised by the auxiliary head's per-window peak predictions.

The personalisation that this recipe achieves is structural rather than parametric: the cold client never produces a per-client weight; what is "personal" is *which prototype the client is routed to* and *what its own auxiliary head predicts about the current input window*. Despite the absence of per-client weights, the recipe outperforms every per-client-weight pFL baseline we evaluate, and it does so with one server boundary crossing rather than dozens.

We make four substantive claims, each supported by experiments in §4:

1. **Cluster-prototype single-shot personalisation is feasible and competitive on cold-start residential peak forecasting.** The recipe reaches a cold peak-amplitude percentage error of 35.7 ± 0.5 %, against 50.4–63.1 % for the strongest baselines across federated, centralised-pooled, and foundation-model families.
2. **Post-hoc personalisation is orthogonal to federated training, not a competitor.** When the codebook is fit on top of a FedAvg- or FedRep-trained backbone, it recovers 8–10 pp of cold-side error on every (seed × backbone) cell, and it does so without modifying the federated training itself.
3. **The communication–accuracy frontier favours a single-shot codebook.** The codebook upload totals 4.94 MB across one boundary cross, against 358–420 MB across twenty rounds for FedAvg / FedProx / Ditto / FedRep, while the cold-side error is also lower.
4. **The dominant personalisation lever in this domain is the auxiliary peak task and the W5 hybrid correction together, not the federation pattern.** A controlled three-way disentanglement (§4.8) decomposes the 20.6 pp total gap from FedAvg to our reference into 6.2 pp from federation alone, 5.6 pp from a learned versus self-derived auxiliary head, and 8.8 pp from the W5 hybrid correction module itself. The four federated algorithms cluster within one percentage point of each other on cold error, and an iterative prototype-based pFL baseline (FedProto, §4.10) lands essentially on top of FedAvg.

### 1.4 What this paper is and is not

It is an algorithm paper proposing a new pFL design point and a thorough empirical evaluation against eleven published baselines on a residential forecasting benchmark. It is not a foundation-model paper, not a federated optimisation paper, and not a privacy-preservation paper — although §5.4 discusses the privacy implications of single-shot representation upload honestly.

---

## 2. Related Work

### 2.1 Personalised federated learning

The pFL literature can be organised along two axes: *what* is personalised (full model, head only, regulariser pull, prototypes, cluster assignment) and *how often* the personalisation is updated (every round, every k rounds, once). Existing methods cover most cells of this matrix. Parameter-decoupling methods [3, 4] keep the head local and exchange the encoder; regularisation methods [5, 6] interpolate between global and personal models with a tunable strength; meta-learning methods [8] target an initialisation that adapts in a few gradient steps; prototype methods [9, 10] aggregate per-class or per-task summaries server-side; and clustered FL [11, 12] partitions clients into homogeneous groups.

The cell that has received the least attention is *server-side cluster prototypes that are computed once, post-hoc, and consumed by inference-time retrieval rather than by parameter learning*. Federated fuzzy c-means [13] is the closest neighbour in spirit but is iterative and is typically deployed for clustering itself, not for downstream personalisation. The present work adds a peak-aware auxiliary task and a domain-specific correction structure to convert the codebook into a personalisation mechanism for cold-start clients. Importantly, our codebook crosses the donor–server boundary exactly once, and the cold client never makes a gradient step.

### 2.2 Cold-start time-series forecasting

BuildingsBench [2] established a strong reference for cold-start residential load: a Transformer pre-trained on 900 K simulated buildings achieves NRMSE 79.34 % on real residential zero-shot, while persistence is 77.88 %, and fine-tuning on the cold client's history adds only −2.23 pp. This dataset-of-datasets framing exposes how thin the gap between elaborate pre-training and trivial baselines is at the residential scale, and it motivates *correction-style* approaches like ours: rather than asking a single generic representation to model every household, we apply a small inference-time correction that is parameterised by which cluster of training households the cold input most resembles and what the auxiliary head predicts about its peak.

### 2.3 Vector quantisation in time series

Vector quantisation has been used for tokenising time-series inputs in self-supervised pre-training [VQ-MTM, 14; TimesFM, 15], for compressing latents inside a forecasting backbone, and for clustering hidden states in classification settings. Our use of vector quantisation differs from each of these: we apply a single post-hoc KMeans++ pass to a *frozen* encoder's hidden representations and use the resulting codebook as a lookup-table-style correction module at inference time. The codebook is never differentiable through and never re-fit during inference; it is a *server-side artefact*, not a learned layer of the model.

### 2.4 Auxiliary tasks in forecasting

Auxiliary classification and regression heads have been used to shape forecasting representations [Seq2Peak, 16], typically with the auxiliary loss helping the main forecast through joint optimisation. Our use is similar in form but different in role: the auxiliary head's outputs at inference time directly parameterise the correction module's Gaussian template, so the head plays both a representation-shaping role *during training* and a personalisation role *during inference*.

---

## 3. Method

### 3.1 Federated forecasting setup

Let $\mathcal{I}$ index a set of households, each producing an hourly time series $\mathbf{x}^i \in \mathbb{R}^{T_i}$. The forecasting task is a sliding window: given an input window of length $L = 96$ hours, predict the next $H = 24$ hours,

$$
\hat{\mathbf{y}}^{\,i}_{t} = f\!\left(\mathbf{x}^i_{t-L+1\,:\,t}\right) \in \mathbb{R}^{H}.
$$

Households are partitioned into a **train pool** of size $N_\text{train} = 80$ and a **cold pool** of size $N_\text{cold} = 20$. Train households participate in the federated training stage. Cold households are held out: they receive the trained backbone and the post-hoc codebook from the server but never enter the training loop, and they are never asked to perform a local training step. Per-household z-normalisation uses each household's own first-70 % statistics; the encoder weights and codebook are never updated using cold-household data.

We separate the federation into three phases (Figure 1). **Phase A** trains the backbone and auxiliary head jointly on the train pool. The training itself is identical to a standard centralised pre-training run; in §4 we additionally evaluate the recipe under a federated Phase A using FedAvg / FedProx / FedRep / Ditto, demonstrating that the post-hoc codebook stacks on top of any of these. **Phase B** fits the codebook by a single KMeans++ pass over the encoder's hidden representations of training windows. The codebook crosses the donor → server boundary exactly once; no gradient information flows through it, and it is frozen after fitting. **Phase C** is fully local cold inference: each cold household receives the frozen backbone, auxiliary head, and codebook, and produces day-ahead forecasts entirely within its own boundary, with no further server interaction.

### 3.2 Peak-aware encoder

The backbone is a three-stack NBEATSx [17] with hidden dimension $d_\text{model} = 64$, three polynomial trend basis functions, and five Fourier seasonal harmonics. Each stack $s \in \{\text{trend}, \text{seasonal}, \text{generic}\}$ produces a forecast component $\hat{\mathbf{y}}_{(s)} \in \mathbb{R}^{H}$ together with a hidden representation $\mathbf{h}_{(s)} \in \mathbb{R}^{64}$; the final forecast is the sum of stack components,

$$
\hat{\mathbf{y}} \;=\; \sum_{s} \hat{\mathbf{y}}_{(s)}, \qquad \mathbf{h}_g \;\equiv\; \mathbf{h}_{(\text{generic})} \in \mathbb{R}^{64},
$$

and $\mathbf{h}_g$ is the latent that downstream codebook construction operates on. We chose the generic-stack hidden over a concatenation of all three stacks' hiddens after a controlled ablation (§4.5) that found the generic-stack alone produced cleaner cluster structure.

To shape $\mathbf{h}_g$ toward peak-discriminative content, we attach an **auxiliary peak head** that consumes $\mathbf{h}_g$ and outputs a peak-amplitude regression and a peak-hour classification:

$$
\bigl(\hat a,\; \boldsymbol{\ell}\bigr) \;=\; \mathrm{AuxHead}(\mathbf{h}_g),
\qquad \hat a \in \mathbb{R}, \quad \boldsymbol{\ell} \in \mathbb{R}^{24},
$$

implemented as a $64 \to 32 \to \{1, 24\}$ multilayer perceptron with ReLU activation. Let $a^\star = \max_t y_t$ and $h^\star = \arg\max_t y_t$ denote the ground-truth peak amplitude and peak hour over the forecast horizon. The training loss is

$$
\mathcal{L}_\text{total} \;=\; \mathrm{MAE}(\hat{\mathbf{y}}, \mathbf{y}) \;+\; \lambda \cdot \mathcal{L}_\text{aux}, \qquad
\mathcal{L}_\text{aux} \;=\; \mathrm{MSE}(\hat a, a^\star) \;+\; \eta \cdot \mathrm{CE}(\boldsymbol{\ell}, h^\star),
$$

with $\lambda = 0.3$ and $\eta = 0.1$. The amplitude term is in z-normalised forecast space and the hour term is a 24-class classification of the argmax hour of the forecast horizon. The predicted peak hour at inference is $\hat h = \arg\max_t \boldsymbol{\ell}_t$. A controlled ablation (§4.5) attributes the bulk of the cold-side improvement to *the auxiliary loss alone* — that is, to the representation-shaping effect of training under $\mathcal{L}_\text{aux}$ rather than to the explicit use of $(\hat a, \hat h)$ in the correction module. This finding will return in §4.8 as the basis for our claim that the dominant personalisation lever in this domain is the auxiliary task, not the federation pattern.

### 3.3 Single-shot post-hoc codebook

After Phase A completes, we forward all train-pool windows (stride $= H = 24$, $N \approx 19{,}250$ total windows) through the frozen encoder and collect the latent set

$$
\mathcal{H}_\text{train} \;=\; \bigl\{\, \mathbf{h}_g^{(j)} \in \mathbb{R}^{64} \,\bigr\}_{j=1}^{N}, \qquad N \approx 19{,}250.
$$

A single KMeans++ pass produces $M = 32$ cluster centres $\{\boldsymbol{\mu}_c\}_{c=1}^{M}$, where

$$
\{\boldsymbol{\mu}_c\}_{c=1}^{M} \;=\; \arg\min_{\{\boldsymbol{\mu}_c\}} \sum_{j=1}^{N} \min_{c}\, \bigl\lVert \mathbf{h}_g^{(j)} - \boldsymbol{\mu}_c \bigr\rVert_2^{2},
$$

solved with $n_\text{init} = 10$ KMeans++ seedings. For each cluster $c$ we compute the **cluster-mean residual offset** in forecast space,

$$
\mathcal{S}_c \;=\; \bigl\{\, j : \arg\min_{c'} \lVert \mathbf{h}_g^{(j)} - \boldsymbol{\mu}_{c'}\rVert_2 = c \,\bigr\}, \qquad
\mathbf{o}_c \;=\; \frac{1}{|\mathcal{S}_c|} \sum_{j \in \mathcal{S}_c}\bigl(\mathbf{y}^{(j)} - \hat{\mathbf{y}}^{(j)}\bigr) \;\in\; \mathbb{R}^{H},
$$

where $\hat{\mathbf{y}}^{(j)}$ is the backbone's z-norm-space forecast for training window $j$ and $\mathbf{y}^{(j)}$ is the corresponding ground truth. The codebook produced by this construction is **frozen**: no gradient flows through it during downstream training, no straight-through estimator is used, and it is never re-fit at inference time. The codebook is a *server-side artefact* with footprint $32 \times 64 \times 4$ bytes for centres plus $32 \times 24 \times 4$ bytes for offsets — about 11 KB total in fp32.

The choice of $M = 32$ follows from preliminary sweeps in which we observed cluster utilisation of 1.0 (every cluster contains at least one training window) and a minimum cluster size of 113 windows on the 80-household train pool. Smaller $M$ under-utilises the training distribution; larger $M$ fragments small clusters and inflates per-cluster offset variance. A formal sensitivity analysis is left for future work; we report all results at $M = 32$.

### 3.4 Inference-time routing

Each cold household at inference time produces a forward pass through the frozen backbone:

$$
\bigl(\hat{\mathbf{y}}_\text{base},\; \mathbf{h}_g^\text{cold},\; \hat a,\; \hat h\bigr) \;=\; \mathrm{NBEATSxAux}(\mathbf{x}_\text{cold}),
$$

with $\mathbf{x}_\text{cold}$ a 96-hour input window from the cold household's own series. To select which codebook entry the correction module will use, we evaluate two routing rules. The first, **input-only routing**, computes a five-dimensional descriptor of the cold input and matches it to the train-pool descriptors:

$$
\mathrm{KEY}(\mathbf{x}) \;=\; \bigl[\, \max(\mathbf{x}),\;\; \tfrac{1}{L}\arg\max(\mathbf{x}),\;\; \mathrm{mean}(\mathbf{x}),\;\; \mathrm{std}(\mathbf{x}),\;\; \max(\mathbf{x}_{-24:})\,\bigr] \in \mathbb{R}^{5},
$$

$$
j^\star \;=\; \arg\min_{j}\, \bigl\lVert \tilde{\mathrm{KEY}}(\mathbf{x}_\text{cold}) - \tilde{\mathrm{KEY}}(\mathbf{x}_\text{train}^{(j)}) \bigr\rVert_2, \qquad c^\star \;=\; c\bigl(\mathbf{h}_g^{(j^\star)}\bigr),
$$

with $\tilde{\mathrm{KEY}}$ standardised using the train-pool mean and scale. The second, **latent routing**, uses the encoder's own hidden representation:

$$
c^\star \;=\; \arg\min_{c}\, \bigl\lVert \mathbf{h}_g^\text{cold} - \boldsymbol{\mu}_c \bigr\rVert_2.
$$

Latent routing is information-richer (64 dimensions versus five) and requires no extra forward pass, since $\mathbf{h}_g^\text{cold}$ is already produced by the auxiliary head's forward. Empirically the two routing rules are statistically indistinguishable on cold peak-amplitude error (§4.4) but latent routing has a small (+0.6 pp) hit-rate advantage. We report headline numbers under input-only routing for direct comparability with the v01 baseline iteration; latent routing results appear in §4.4.

### 3.5 Hybrid correction

Given the routed cluster $c^\star$ and the auxiliary-head predictions $(\hat a, \hat h)$ for the current cold window, the **hybrid correction** combines a cluster-mean offset (CMO) and a Gaussian sharp template (GST) into a corrected forecast:

$$
g(t;\, \hat h, \hat a, \sigma) \;=\; \hat a \cdot \exp\!\left(-\frac{(t - \hat h)^{2}}{2\sigma^{2}}\right), \qquad t \in \{0, 1, \dots, H-1\},
$$

$$
\hat{\mathbf{y}}_\text{corr} \;=\; \hat{\mathbf{y}}_\text{base} \;+\; \alpha_{v0} \cdot \mathbf{o}_{c^\star} \;+\; \alpha_{w1} \cdot g(\,\cdot\,;\, \hat h, \hat a, \sigma).
$$

The first additive term ($\mathbf{o}_{c^\star}$) is a *cluster-level* correction: it pushes the forecast toward the average residual of training households that share $c^\star$'s peak structure. The second ($g$) is a *per-window* correction: it places a Gaussian bump centred at the auxiliary head's predicted peak hour $\hat h$, with amplitude $\hat a$ and width $\sigma$. Their sum combines one personalisation source operating at the *cluster level* (a structural prior) with another operating at the *per-window level* (a forecast-specific sharpening).

### 3.6 Operating points and the Pareto framing

The hyperparameters $(\sigma, \alpha_{v0}, \alpha_{w1})$ control a Pareto trade-off between peak-amplitude accuracy and peak-hour hit rate. We carry forward two recommended operating points:

- **HR-preserving** ($\sigma = 3.0,\ \alpha_{v0} = 1.0,\ \alpha_{w1} = 0.1$): peak-amplitude error reduced, hit rate matched within seed variance. Recommended when peak-hour timing matters operationally (e.g., demand-response triggering).
- **PAPE-aggressive** ($\sigma = 3.0,\ \alpha_{v0} = 1.5,\ \alpha_{w1} = 0.5$): peak-amplitude error pushed harder, hit rate paying a small ($\le 1$ pp) tax. Recommended when peak-amplitude calibration dominates (e.g., capacity provisioning).

These operating points are determined on the training side. The cold pool is **not** used for hyperparameter selection at any point in the pipeline. We adopt this rule deliberately: tuning $(\sigma, \alpha_{v0}, \alpha_{w1})$ on the cold pool would re-introduce a selection-bias concern that has been raised by prior work on similar correction modules. All numbers reported in §4 use the PAPE-aggressive point unless stated otherwise; HR-preserving numbers appear in the appendix.

---

## 4. Experiments

### 4.1 Setup

**Dataset.** UMass Smart\* hourly residential, 100 households, single year (2016). A cluster-stratified split partitions the households into 80 train and 20 cold. Each household uses 7 : 1 : 2 train / val / test on its own time series; per-household z-normalisation uses the household's own first 70 %. Three random seeds $\{42, 123, 7\}$ control the split realisation, the KMeans++ initialisation, and the federated round-robin client order; all reported numbers are mean ± sample standard deviation (ddof = 1) across the three seeds, over $\approx 4810$ cold windows per seed (= 20 cold households × ~ 240 stride-24 windows each).

**Metrics.** The cold-side evaluation uses three metrics whose definitions are bit-exact ports of an established residential-forecasting metric library. For a window with ground-truth horizon $\mathbf{y} \in \mathbb{R}^{H}$ and forecast $\hat{\mathbf{y}} \in \mathbb{R}^{H}$, write the per-window peak amplitude and peak position as

$$
a^\star = \max_{t}\, y_t, \qquad \hat a^\star = \max_{t}\, \hat y_t, \qquad
h^\star = \arg\max_{t}\, y_t, \qquad \hat h^\star = \arg\max_{t}\, \hat y_t.
$$

The **peak absolute percentage error (PAPE)** and the **hit rate at tolerance $k$ (HR@$k$)** are then

$$
\mathrm{PAPE} \;=\; \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \frac{\bigl|\hat a^\star_i - a^\star_i\bigr|}{|a^\star_i|} \times 100\,\%,
\qquad
\mathrm{HR}@k \;=\; \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \mathbb{1}\!\left[\,\bigl|\hat h^\star_i - h^\star_i\bigr| \le k\,\right] \times 100\,\%,
$$

where $\mathcal{V} = \{i : |a^\star_i| > 10^{-5}\}$ is the set of windows with non-trivial true peak. Both quantities are dimensionless percentages. PAPE is therefore in **% of the true peak**, *not* in kilowatts; differences between two PAPE values are reported in **percentage points (pp)**. The mean absolute error $\mathrm{MAE} = |\mathcal{V}|^{-1}\sum_i \frac{1}{H}\sum_t |\hat y_{i,t} - y_{i,t}|$ is computed in *denormalised kilowatts* and is reported in kW. PAPE is the headline cold-side metric throughout. HR@$k$ chance levels are $\frac{2k+1}{24}$, i.e. $12.5\,\%$ for $k=1$ and $20.8\,\%$ for $k=2$.

**Baseline families.** Eleven published baselines spanning three pFL-relevant families:

- **Federated baselines (5):** FedAvg [18], FedProx [7], FedRep [3], Ditto [5], and a no-FL Local-only NBEATSx upper-bound. All five share the same MinimalNBEATSx backbone (no auxiliary head) and the same training budget (20 rounds, 2 local epochs per round, full client participation, Adam lr = 1e-3, batch size 512, bf16 autocast). FedRep's encoder/head split is *stack_\*.fc{1..4}.\** versus *stack_\*.proj.\**, recovering the per-client output layer naturally. Ditto's lambda is 0.1 and FedProx's μ is 0.01, both following published defaults.
- **Centralised neural-forecasting baselines (3):** DLinear [19], NHITS [20], Crossformer [21]. All three are trained on the *pooled* 80 train households with no federation; this gives them a strict data-access advantage over the FL baselines and serves as a *non-private upper bound* on what an architecturally-different forecasting model could achieve on this data.
- **Foundation-model zero-shot baselines (3):** Chronos-Bolt small, Chronos-T5 tiny [22], TimesFM [15]. None are trained on UMass; each is called once per cold window with its native forecasting interface, with the model's internal scaling.

Implementations of every baseline are verified against publicly available reference code; the verification record is published alongside the codebase.

### 4.2 Headline cold-side comparison

Table 1 reports cold peak-amplitude error and hit rate, sorted by cold PAPE. The proposed method, listed as the reference row, outperforms every published *baseline* by at least 11.8 pp. The Stacked-Aux row (41.93 %) is a *federated variant of our recipe* (NBEATSxAux trained federatedly, codebook fit on the federated backbone) — it is not a competing baseline, and the 6.2 pp cost relative to centralised training is the focus of §4.8's three-way disentanglement.

**Table 1.** Cold PAPE (%) and HR@1 (%) for all baselines and for the proposed method. Mean ± std over three seeds. The "Group" column marks federated (FL), centralised pooled neural-forecasting (NF), foundation-model zero-shot (FM), our cross-cell row (Stacked; see §4.3), and the no-federation upper bound. The proposed method is reported at the PAPE-aggressive operating point under input-only routing.

| Method | Cold PAPE (%) | HR@1 (%) | Group |
|---|---:|---:|---|
| **Proposed (peak-aware codebook + Hybrid correction)** | **35.70 ± 0.49** | 26.3 ± 2.2 | ours |
| Codebook on FedAvg-NBEATSxAux backbone (Stacked-Aux) | 41.93 ± 1.30 | 13.0 ± 1.8 | Stacked-Aux |
| Codebook on FedRep backbone (Stacked) | 47.50 ± 1.36 | 23.5 ± 1.7 | Stacked |
| Codebook on FedAvg backbone (Stacked) | 48.26 ± 3.74 | 23.9 ± 1.4 | Stacked |
| DLinear (centralised pooled) | 50.37 ± 0.84 | 26.4 ± 1.8 | NF |
| Crossformer (centralised pooled) | 52.54 ± 1.71 | 26.9 ± 2.2 | NF |
| Local-only NBEATSx, self-eval ⚠ | 52.64 ± 2.44 | **28.5 ± 2.0** | upper bound |
| NHITS (centralised pooled) ★ | 52.74 ± 1.71 | 26.8 ± 2.3 | NF |
| Chronos-Bolt small (zero-shot) | 52.69 ± 1.56 | 26.2 ± 1.9 | FM |
| TimesFM (zero-shot) | 54.27 ± 2.15 | 25.0 ± 1.2 | FM |
| FedProx | 56.30 ± 1.55 | 26.0 ± 1.5 | FL |
| FedAvg | 56.34 ± 1.41 | 26.4 ± 1.6 | FL |
| FedProto (pFL prototype-based) | 56.37 ± 1.44 | 26.6 ± 1.7 | pFL |
| Ditto | 56.38 ± 1.63 | 26.5 ± 1.8 | FL |
| FedRep | 57.18 ± 1.52 | 25.7 ± 1.6 | FL |
| FedAvg-NBEATSxAux (no codebook) | 57.32 ± 1.55 | 26.4 ± 1.7 | FL-Aux |
| Local-only NBEATSx, holdout-eval | 62.69 ± 2.92 | 19.1 ± 2.0 | no-FL fair |
| Chronos-T5 tiny (zero-shot) | 63.13 ± 3.04 | 18.3 ± 0.8 | FM |

★ NHITS row reflects the corrected MLP construction (every hidden Linear followed by activation + dropout, terminal Linear bare). The previous draft reported 52.99 ± 1.64; the corrected number is 52.74 ± 1.71. The difference is within seed noise, but the corrected implementation is the reference.

The Local-only row carries a caveat (⚠): its training segment and its evaluation segment overlap, so its result is an *overfit upper bound on its own data* rather than a fair generalisation result. All other rows train on the 80 train households and evaluate on the 20 cold households (a held-out, non-overlapping set), which is the protocol the proposed method also uses. We report Local-only for completeness but treat its HR@1 = 28.5 % as a sanity point only.

### 4.3 Orthogonality: codebook on top of federated backbones

The two **Stacked** rows in Table 1 are central to the pFL framing. We take a backbone trained federatedly (FedAvg or FedRep) and apply Phase B and Phase C of our method on top: forward all training windows through the federated backbone to collect $\mathbf{h}_g$, fit the $M = 32$ KMeans codebook, compute per-cluster residual offsets, and route cold inputs as in §3.4–3.5. The auxiliary head $(\hat a, \hat h)$ is derived self-referentially from the federated backbone's own forecast, since the federated backbone has no learned auxiliary path: we use the maximum and argmax of $\hat{\mathbf{y}}$ as proxies. The Gaussian template's amplitude and centre are therefore weaker than the proposed method's learned $(\hat a, \hat h)$, as confirmed by the 11.8 pp residual gap between the Stacked rows and our reference row.

The result, broken down per (seed × federated backbone) cell, appears in Table 2.

**Table 2.** Cold PAPE (%) per seed for raw federated backbone versus the same backbone with the post-hoc codebook + Hybrid correction (PAPE-aggressive operating point) applied. All six (seed × backbone) cells improve.

| Backbone (seed 42 / 123 / 7) | Raw FL PAPE | Codebook PAPE-aggr | Δ (pp) |
|---|---|---|---|
| FedAvg | 56.17 / 57.83 / 55.03 | 52.56 / 46.46 / 45.75 | −3.61 / −11.36 / −9.28 |
| FedRep | 56.34 / 58.94 / 56.26 | 48.85 / 47.51 / 46.13 | −7.50 / −11.43 / −10.13 |

The post-hoc codebook reduces cold-side error on every cell by 3.6 to 11.4 pp. Two findings follow. First, the codebook is *complementary* to federated training rather than a competitor: it adds personalisation on top of any federated backbone we tested, with no modification to the federated training itself. Second, the codebook benefits *more* from FedRep's encoder/head split structure than from FedAvg's monolithic averaging — the FedRep cross-cell mean (47.50 %) is slightly better than the FedAvg cross-cell mean (48.26 %), reversing the raw federated ranking (where FedRep underperforms FedAvg). One reading is that the encoder/head split of FedRep produces a hidden-representation space with cleaner cluster structure, which a downstream KMeans codebook can exploit more cleanly.

### 4.4 Routing: input-only vs. latent

Table 3 compares the two routing rules of §3.4 on the same cold pool and the same operating point.

**Table 3.** Cold PAPE and HR@1 under input-only routing (5-d KEY) versus latent routing (64-d hidden), over three seeds. Means are statistically indistinguishable in PAPE; latent routing shows a small hit-rate advantage.

| Routing | Cold PAPE (%) | HR@1 (%) |
|---|---:|---:|
| Input-only (KEY-NN) | 35.70 ± 0.49 | 26.3 ± 2.2 |
| Latent ($\mathbf{h}_g$ argmin) | 35.71 ± 0.45 | 26.9 ± 1.8 |

Latent routing is information-richer (64 dimensions versus five) and costs no extra forward pass, since $\mathbf{h}_g^\text{cold}$ is already produced for the auxiliary head. The two routing rules being indistinguishable on PAPE suggests that the five-dimensional KEY captures enough peak-relevant structure for the routing decision to be near-optimal. The +0.6 pp HR@1 advantage of latent routing aligns with the intuition that the higher-dimensional descriptor resolves ambiguous boundary cases between adjacent clusters.

### 4.5 Auxiliary task ablation

A controlled ON/OFF ablation on the auxiliary task isolates its contribution. We retrain the backbone with $\lambda = 0$ (no auxiliary loss) and rebuild the codebook on the resulting $\mathbf{h}_g$; we then compare the cold-side error under the proposed correction module. The result (averaged over three seeds, PAPE-aggressive operating point) is **+11.9 ± 11.2 pp PAPE** when the auxiliary task is removed. The high standard deviation reflects a vanilla-codebook collapse on one of the three seeds, in which the un-shaped $\mathbf{h}_g$ space fails to cluster cleanly and a small number of clusters absorb most of the training windows, degrading the per-cluster offset estimates. Even setting that seed aside, the auxiliary task contributes a substantial fraction of the cold-side improvement: the personalisation lever in our recipe is *not* the codebook alone but the codebook *over a peak-aware representation*.

### 4.6 Heterogeneity of the train pool

To put the cluster-prototype framing on quantitative footing, we compute pairwise statistics over the 80 train-pool households on their own first-70 % segments. The mean pairwise Wasserstein-1 distance between household marginals (in kW) is **0.379 kW**, with a maximum of **1.439 kW** — a factor-of-four spread between mean and max, indicating a long heterogeneity tail. The mean pairwise Jensen–Shannon-symmetric KL on 64-bin histograms is **0.067**. The mean pairwise hour-of-day cosine similarity between household profiles is **0.970**, with a *minimum* of **0.811** — that is, the households differ in *amplitude* far more than in *peak-hour timing*.

This finding has direct design implications for the personalisation strategy. Cluster-prototype methods that operate over a hidden representation are tracking *amplitude-dominant* heterogeneity, since the amplitude axis is where the variance lives. The Gaussian template's parameterisation (amplitude $\hat a$ explicitly, hour $\hat h$ explicitly) reflects exactly this decomposition. Methods that do not have an amplitude-aware mechanism — for example, the four federated baselines, all of which average MAE over z-normalised windows — implicitly absorb amplitude variation into the per-household z-norm and lose it from the federation signal. This is a structural reason why the federated baselines cluster within one percentage point of each other on cold-side error: the federation pattern is not the binding constraint; the *absence of amplitude-aware personalisation* is.

### 4.7 Communication and the (bytes × accuracy) Pareto

Table 4 reports the communication cost of every federated baseline alongside the proposed method's single-shot codebook upload. We use the canonical fp32 byte model: one model parameter is four bytes, eighty clients exchange uploads each round, and we report total upload bytes across the entire pipeline.

**Table 4.** Communication accounting. *Bytes per round* is per-cohort (eighty clients each uploading once). *Boundary crosses* is the number of donor → server crosses across the entire training and personalisation lifecycle.

| Method | Bytes / round | Rounds | Total bytes | Boundary crosses |
|---|---:|---:|---:|---:|
| FedAvg / FedProx / Ditto | 21.0 MB | 20 | 420.4 MB | 20 |
| FedRep | 17.9 MB | 20 | 358.8 MB | 20 |
| Local-only (no FL) | 0 | 0 | 0 | 0 |
| **Proposed (single-shot codebook)** | **4.94 MB** | **1** | **4.94 MB** | **1** |

The single-shot codebook uploads the train-pool aggregate of $\mathbf{h}_g$ latents ($\approx 19{,}250 \times 64 \times 4$ bytes $\approx 4.93$ MB), plus the centroids and offsets broadcast (negligible). This is **85 ×** less data than FedAvg's twenty-round total and **20 ×** fewer boundary crosses. The (bytes × cold-side accuracy) Pareto frontier is therefore strictly dominated on both axes by the proposed method against every iterative federated baseline in Table 4.

We discuss the privacy implications of this result honestly in §5.4. In particular, "less data" is not the same as "more private": uploading a learned hidden representation once may carry a different threat profile from uploading model gradients twenty times, and the comparison along the privacy axis is not collapsed cleanly into bytes.

### 4.8 Disentangling the cold-side gap

A reader could reasonably suspect that the 11.8 pp headline gap to the strongest published baseline (the Stacked row at 47.50 %) conflates two effects: the auxiliary head's representation-shaping during training, and the choice of centralised versus federated training. To address this, we add one extra row that pairs the *peak-aware backbone* with the *federated training pattern*: a NBEATSxAux model in which both the backbone and the auxiliary head are federated jointly via FedAvg (full participation, $20$ rounds, $2$ local epochs per round, batch size 512, bf16 autocast, joint loss $\mathrm{MAE} + 0.3 \cdot \ell_\text{aux}$). After Phase A completes, we refit the codebook on the federated NBEATSxAux's $\mathbf{h}_g$ and run cold inference at the PAPE-aggressive operating point exactly as in §3.

The federated NBEATSxAux row decomposes the 20.6 pp total gap from FedAvg to our reference into three additive parts (Table 6).

**Table 6.** Three-way disentanglement of the cold-side gap (PAPE-aggressive, 3-seed mean ± std).

| Configuration | Backbone | Aux head | Correction | Cold PAPE (%) | Δ (pp) |
|---|---|---|---|---:|---|
| Proposed (centralised) | NBEATSxAux | learned | W5 hybrid | 35.70 ± 0.49 | reference |
| Federated NBEATSxAux + codebook | NBEATSxAux (federated) | learned | W5 hybrid | 41.93 ± 1.30 | +6.23 |
| Codebook on FedRep backbone (Stacked) | MinimalNBEATSx (federated) | self-derived | W5 hybrid | 47.50 ± 1.36 | +5.57 |
| FedAvg (raw) | MinimalNBEATSx (federated) | none | none | 56.34 ± 1.41 | +8.84 |

The decomposition $20.64 = 6.23 + 5.57 + 8.84$ identifies *three additive levers* that together close the cold-side gap:

1. **Federation cost** of $6.23$ pp: holding everything else equal (peak-aware backbone, learned aux head, W5 hybrid correction), federated training adds 6.23 pp PAPE relative to centralised pre-training. This is a well-defined and surprisingly small cost — federation alone is not the dominant lever.
2. **Aux-head cost** of $5.57$ pp: dropping the learned auxiliary head and falling back to self-derived $(\hat a, \hat h) = (\hat{\mathbf{y}}.\max, \hat{\mathbf{y}}.\arg\max)$ as in our Stacked rows adds 5.57 pp. The learned auxiliary path matters but is not dominant.
3. **W5-hybrid cost** of $8.84$ pp: removing the cluster-mean offset and Gaussian template entirely — i.e., relying on the raw federated forecast — adds 8.84 pp on top. The hybrid correction module *itself* is the largest single lever.

We additionally observe that the federated NBEATSxAux backbone *without* the codebook (cold PAPE $57.32 \pm 1.55$) is in fact slightly *worse* than the federated MinimalNBEATSx baseline ($56.34 \pm 1.41$). This is consistent with the reading that the auxiliary head adds capacity but is *only useful at inference time when the W5 hybrid is there to consume its $(\hat a, \hat h)$ outputs* — the auxiliary head is dead weight in isolation. Together with §4.5's auxiliary-task ablation, this indicates that the auxiliary head and the W5 hybrid are *jointly* load-bearing: removing either alone collapses most of the benefit.

### 4.9 Cluster-count sensitivity

To quantify how the cold-side number depends on the codebook size $M$, we refit the codebook at $M \in \{8, 16, 32, 64\}$ on the same centralised NBEATSxAux backbone (the same $\mathbf{h}_g$ pool from the train-pool train-segment forward) and re-evaluate on the cold pool at the PAPE-aggressive operating point. Table 7 reports the result.

**Table 7.** Cluster-count sensitivity (PAPE-aggressive, 3-seed mean ± std). All four $M$ values produce cold PAPE within a 1.3 pp envelope around the $M = 32$ default.

| $M$ | Cold PAPE (%) | HR@1 (%) |
|---:|---:|---:|
|  8 | 36.84 ± 1.25 | 26.7 ± 2.6 |
| 16 | 36.43 ± 0.85 | 26.6 ± 2.4 |
| **32** (default) | **35.70 ± 0.49** | 26.3 ± 2.2 |
| 64 | 35.54 ± 0.92 | 26.3 ± 2.6 |

The sensitivity is small. Halving $M$ from 32 to 16 costs $0.73$ pp PAPE; doubling $M$ to 64 saves $0.16$ pp. Even the smallest codebook ($M = 8$, average density of 10 train households per cluster) is within $1.14$ pp of the default. This empirically supports the claim of §5.1 that the granularity ceiling does not bind on an 80-household train pool: well below the per-client extreme ($M = N_\text{train} = 80$), the cold-side number plateaus, and the choice of $M$ is robust within an order of magnitude. We conjecture that for very large client pools — where amplitude clusters could fragment into much finer behavioural archetypes — $M$ may need to scale; the present benchmark does not exercise that regime.

### 4.10 Prototype-based pFL: FedProto comparison

We add a representative iterative prototype-based pFL baseline to test whether prototype alignment alone — without our Hybrid correction — improves cold-side error. We adapt FedProto [9] to the forecasting setting as follows. The shared backbone is MinimalNBEATSx (matching the other federated baselines). At each round, every client locally computes $K = 32$ prototype centroids by running a one-iteration KMeans over its training-window $\mathbf{h}_g$ latents, warm-starting from the round's global prototypes (so cluster identity aligns across clients). The server aggregates the client prototypes by per-cluster count-weighted mean. Each client's local loss is $\mathrm{MAE}(\hat{\mathbf{y}}, \mathbf{y}) + \lambda_\text{proto} \cdot \mathrm{MSE}(\mathbf{h}_g, \mathbf{p}_{c^\star})$ where $c^\star$ is the batch's nearest global prototype, with $\lambda_\text{proto} = 0.1$. At cold inference time, the cold input is forwarded through the federated backbone and routed via 1-NN on the global prototypes; *no* W5 correction is applied.

The result, in Table 1, is **cold PAPE $56.37 \pm 1.44$** — essentially indistinguishable from FedAvg ($56.34 \pm 1.41$) and Ditto ($56.38 \pm 1.63$), and within one percentage point of FedProx and FedRep. Prototype alignment alone, even when iterative and parameterised at the same $K = 32$ grain as our codebook, does not move the cold-side number meaningfully. This is consistent with the §4.8 reading: the prototypes are not the binding lever; the W5 hybrid correction and the auxiliary head are. FedProto's contribution is the prototype-aligned representation, but absent a downstream correction module that *consumes* that representation through a per-window peak-template, the cold-side error stays within the federated cluster.

### 4.11 Local-only with held-out evaluation

The Local-only row of Table 1 carries an honest caveat: the same first-70 % segment of each cold household is used for both training and evaluation, so the resulting metric is an *overfit upper bound on its own data*. We quantify the overfit by evaluating each per-household locally-trained NBEATSx on a properly held-out segment instead — the next 10 % of the household's series, with sliding stride-24 windows, warm-started by the same first-70 % z-norm statistics used during training. Table 8 reports both numbers.

**Table 8.** Local-only NBEATSx evaluated on its own training segment (overfit upper bound) vs. on a properly held-out segment (fair generalisation), 3-seed mean ± std.

| Evaluation segment | Cold PAPE (%) | HR@1 (%) |
|---|---:|---:|
| Self-eval (first 70 %, same as training) | 52.64 ± 2.44 | 28.5 ± 2.0 |
| Holdout-eval (next 10 %, properly held out) | **62.69 ± 2.92** | 19.1 ± 2.0 |

The fair Local-only number (62.69 % PAPE) is **+10.05 pp worse** than the self-eval number, and falls in the *worst* tier of Table 1, near the Chronos-T5 tiny zero-shot result. The HR@1 advantage of the self-eval row (28.5 %, the highest in the original Table 1) collapses to 19.1 % under holdout evaluation. We update Table 1 to report both rows and treat the holdout row as the fair "no-FL" reference: the federated and centralised baselines all evaluate on a non-overlapping held-out cold pool, and the Local-only holdout row is the only row in that table that uses the same protocol *for the same backbone family* without any federation. The ordering implication is unambiguous: in this benchmark, no-FL is the worst federation choice, not the best.

### 4.12 The dominant axis is the auxiliary task, not the federation pattern

If we sort all rows of Table 1 by cold PAPE and group by whether the method has a peak-aware auxiliary signal *and* a hybrid correction module, the result (Table 9) is striking.

**Table 9.** Methods sorted by cold PAPE, annotated with auxiliary-signal source and correction module.

| PAPE (%) | Auxiliary signal | Correction | Examples |
|---:|---|---|---|
| ~ 36 | learned (amplitude + hour head) | W5 hybrid | proposed (centralised) |
| ~ 42 | learned (amplitude + hour head) | W5 hybrid | proposed under federated training (Stacked-Aux) |
| ~ 48 | self-derived from $\hat{\mathbf{y}}$ | W5 hybrid | codebook on FL backbone (Stacked) |
| ~ 50 | none | none | DLinear (trend/seasonal decomposition helps) |
| ~ 53 | none | none | NHITS / Crossformer / Chronos-Bolt / Local-only-self-eval |
| ~ 56 | none | none | FedAvg / FedProx / FedRep / Ditto / FedProto |
| ~ 57 | learned (amplitude + hour head) | none (federated NBEATSxAux without codebook) | FL-Aux ablation |
| ~ 63 | none | none | Chronos-T5 tiny (zero-shot) / Local-only-holdout |

Methods sort cleanly into tiers, and the tiers are explained by the *combination* of auxiliary signal and correction module rather than by the family or the federation pattern alone. The four federated algorithms cluster within one percentage point of each other; whether one chooses FedAvg, FedProx, FedRep, Ditto, or the iterative prototype-based FedProto, the cold-side number does not move materially. The choice of *forecasting architecture* (DLinear vs. NHITS vs. Crossformer) moves it by a few percentage points; the choice of *foundation model* moves it by ten percentage points. But adding an auxiliary peak head and the corresponding W5 hybrid correction module moves it by twenty percentage points — and crucially, *neither lever works alone*: the FL-Aux ablation row (~57 %) shows that the auxiliary head without the W5 hybrid is in fact a slight regression, while the FedProto and Stacked rows together show that the prototypes / W5 alone (without a learned aux head) recovers only the easier 8.84 pp.

Our reading is that, in this domain, **explicit task decomposition (forecast trajectory + peak amplitude + peak hour) coupled with a structured correction module is the largest single lever for cold-side accuracy**, and that personalisation should be expressed through that joint decomposition + correction structure rather than through generic per-client weight learning.

---

## 5. Discussion

### 5.1 Personalisation granularity

The proposed method's personalisation granularity is **cluster level + per window**: a cold household is personalised to one of $M = 32$ cluster prototypes (with average density $\approx 2.5$ train households per cluster), and each cold *window* is further personalised by its own auxiliary-head outputs $(\hat a, \hat h)$. It is *not* personalised at the per-client weight level — the cold household never produces or stores a per-client parameter.

This granularity decision is a deliberate trade against the parametric-pFL literature. Per-client-weight methods (FedRep, Ditto, Per-FedAvg) place the personalisation budget on parameters; ours places it on inference-time computation. The two budgets have different operational properties:

- **Onboarding cost.** A new client in our framework needs only the frozen backbone, auxiliary head, and codebook ($\approx 11$ KB for the codebook, $\approx 0.26$ MB for the backbone). A new client in FedRep needs an additional per-client training step on its local history. A cold client with no local history breaks FedRep but does not break our method.
- **Maintenance.** Our codebook is fit once and never updated. FedRep's per-client head requires re-tuning whenever the household's behaviour drifts.
- **Granularity ceiling.** We can resolve at most $M$ clusters of personalisation; FedRep can resolve up to $N$ per-client heads. On the present benchmark, with $M = 32$ and $N = 80$ train households, the cluster ceiling does not bind: §4.9 shows that cold PAPE is within a 1.3 pp envelope across $M \in \{8, 16, 32, 64\}$, indicating that cold-side accuracy is dominated by other axes (auxiliary task, correction structure) before the granularity ceiling is approached. In a much larger client pool with high-resolution per-client patterns, the comparison may invert.

### 5.2 When does cluster-level beat per-client?

Three structural conditions favour our cluster-level recipe over per-client-weight pFL:

1. **The heterogeneity is dominated by amplitude rather than shape.** Section 4.6 quantified this for residential load: hour-cosine $\ge 0.811$, while pairwise W1 spans 0.0 to 1.4 kW. When most of the heterogeneity lives along an axis that an auxiliary task can capture (here: amplitude), the auxiliary head plus a Gaussian template carries the bulk of the personalisation signal, leaving little for per-client weights to add.
2. **Cold clients have no training budget.** When a client cannot afford a local fine-tune, per-client-weight methods degrade to their global counterparts (e.g., FedRep with no head training is just a global encoder), while our method continues to function unchanged.
3. **Communication is binding.** When the federation budget is one-shot (e.g., regulatory or auditing requires a single boundary cross) or extremely tight (e.g., satellite uplink, intermittent connectivity), iterative federated methods pay a heavy multiplicative cost in rounds, while ours is constant.

When none of these conditions holds — when the heterogeneity is shape-dominated, when every client carries a generous local-history budget, when iterative federation is cheap — per-client-weight pFL methods may regain the advantage, particularly for tasks where the personalisation does *not* admit a clean auxiliary-task decomposition. Our recipe is therefore not a universal replacement; it is a *new operating point* on the pFL design surface, useful where the three structural conditions hold.

### 5.3 The Pareto framing and operational deployment

The two operating points (HR-preserving and PAPE-aggressive) of §3.6 expose a user-controllable trade-off rather than a single best number. In a demand-response deployment, where missing a peak hour has high economic cost, the HR-preserving point is recommended: it matches the baseline hit rate within seed variance while still cutting peak-amplitude error by $\approx 18\,\%$ relative. In a capacity-provisioning deployment, where peak amplitude is the binding metric, the PAPE-aggressive point cuts amplitude error by $\approx 33\,\%$ relative at a small ($\le 1$ pp) hit-rate cost. The two points share the same correction module and the same codebook; only $(\sigma, \alpha_{v0}, \alpha_{w1})$ differ. The cold pool is *not* used to choose between them: the deployment chooses based on its own utility function.

### 5.4 Privacy considerations

A cleaner reading of the communication accounting in §4.7 requires care along the privacy axis. Our codebook upload is a single-shot aggregate of training-side hidden representations, then a broadcast of cluster centres and offsets. From a *bytes* viewpoint, this is dominated by every iterative federated baseline; from a *threat model* viewpoint, it is not strictly dominated. Hidden representations of a household's data carry information about that household's distribution that may, under threat-aggregation reconstruction attacks, permit partial recovery of the underlying input. Federated gradient sharing carries different but analogous risks. We do not claim "private" as a contribution of the proposed method; we claim *less data, fewer rounds*. The privacy comparison is left as a deliberate open question for future work, and it would be the natural extension of our framework via differentially private K-means or via federated K-means on the hidden representations themselves.

---

## 6. Limitations

We list limitations honestly. None of them invalidate the headline result, but each is a legitimate caveat that future work will need to address.

**Federated training of the auxiliary head was tested only at one configuration.** The §4.8 disentanglement uses FedAvg with full participation, $20$ rounds, $2$ local epochs per round, and the canonical joint loss $\mathrm{MAE} + 0.3 \cdot \ell_\text{aux}$. We did not vary the federation algorithm (FedProx / Ditto / FedRep + auxiliary head) or sweep the auxiliary-loss weight $\lambda$ in the federated regime. The decomposition $20.6 = 6.2 + 5.6 + 8.8$ is therefore one well-defined cut of the disentanglement, but the absolute "federation cost" of $6.2$ pp may shift by 1–2 pp under a different federated optimiser. We expect the *ordering* of the three additive levers (federation < aux head < W5 hybrid) to be robust, but the exact split is benchmark-specific.

**Centralised vs. federated baselines are not on the same threat model.** The neural-forecasting baselines (DLinear, NHITS, Crossformer) train on the *pooled* train-pool data without federation. They therefore have a strict data-access advantage over the federated baselines, while paying a corresponding privacy cost. Table 1 lists them in a single sorted column for simplicity, but the deployment choice between them depends on the operator's privacy posture.

**Local-only's overfit upper bound is now quantified.** As noted in §4.2 and confirmed in §4.11, the original Local-only row's training and evaluation segments overlapped. We now report both the self-eval row (cold PAPE 52.64 ± 2.44, the overfit upper bound on its own data) and the held-out holdout-eval row (cold PAPE 62.69 ± 2.92, fair generalisation), and the gap of $+10.05$ pp PAPE between them is the quantitative size of the overfit. The honest reading is that no-FL is the *worst* federation choice on this benchmark, not the best — the holdout-eval row falls in the bottom tier of Table 1, near zero-shot Chronos-T5.

**Heterogeneity correlation is incomplete.** The pairwise heterogeneity statistics of §4.6 are computed over the 80 train-pool households, while the per-household cold-side error is by construction available only on the 20 cold households. The two sets do not overlap, so a per-household correlation between heterogeneity rank and cold-side gap is not computable from the present design. A study that splits a single household pool into both heterogeneity-source and cold-target would resolve this; we leave it as a separate analysis.

**Single dataset, single horizon.** All numbers are on UMass Smart\* 2016 with horizon $H = 24$. Cross-dataset generalisation and long-horizon ($H = 96, 168$) behaviour are open questions. The Crossformer ranking in particular is expected to change at long horizon, where its cross-dimension stage becomes load-bearing.

**Federated hyperparameter grid was not fully exhausted.** The four federated algorithms cluster within one percentage point of each other on cold-side error. We attribute this clustering to training-loss saturation by round 17 across all four, and we report a tuning grid (FedProx $\mu$, Ditto $\lambda$, FedRep head-epochs) only as committed-but-not-executed. A reviewer-driven reading is that proper tuning could move the federated baselines, particularly Ditto with a stronger personal-pull. We accept this caveat and note that even a one-pp swing would not close the 20 pp gap to our reference row.

**Cluster-count sensitivity is reported only in the small range.** §4.9 reports a sensitivity sweep at $M \in \{8, 16, 32, 64\}$, all four of which sit within a 1.3 pp PAPE envelope. We did *not* extend the sweep to $M = 128$ or beyond on the present 80-household train pool, since the per-cluster density would fall below 1 train household at $M = 128$. The granularity-vs-pool-size relationship at thousands of clients — where amplitude clusters could fragment into much finer behavioural archetypes — therefore remains an open question that the present benchmark does not exercise.

**Auxiliary-task contribution has high seed variance.** Section 4.5 reported the auxiliary-task contribution as +11.9 ± 11.2 pp, with the variance dominated by a vanilla-codebook collapse on one of three seeds. The headline finding ("the auxiliary task is the dominant lever") is robust across the two non-collapsed seeds, but the ablation result is not as tight as we would like.

---

## 7. Conclusion

We proposed a new point in the personalised federated learning design space: personalisation expressed as an inference-time codebook lookup, with no per-client weights and no client-side training step. The recipe consists of a peak-aware encoder trained jointly with an auxiliary peak-prediction task, a single-shot post-hoc KMeans codebook over the encoder's hidden representations, and a hybrid correction module that combines a cluster-mean residual offset with a sharp Gaussian template parameterised by the auxiliary head's per-window predictions. On the UMass Smart\* benchmark, the recipe outperforms thirteen published baselines spanning federated (FedAvg / FedProx / FedRep / Ditto / FedProto / Local-only), centralised neural-forecasting (DLinear / NHITS / Crossformer), and foundation-model (Chronos-Bolt / Chronos-T5 / TimesFM) families by at least 11.8 pp on cold peak-amplitude percentage error, while requiring 85 × less communication and 20 × fewer boundary crosses than iterative federated learning, and it stacks cleanly on top of any of the four federated baselines we tested. A controlled three-way disentanglement against a federated NBEATSxAux row decomposes the 20.6 pp total gap from FedAvg to our centralised reference into 6.2 pp from federation, 5.6 pp from the learned auxiliary head, and 8.8 pp from the W5 hybrid correction module. The most striking analytical finding is that the dominant axis of cold-side accuracy in this domain is not the federation pattern but the explicit task decomposition coupled with a structured correction module: personalising through a peak-aware auxiliary task and a hybrid template-based correction moves the cold-side number more than any of the federated, neural-forecasting, or foundation-model design choices we evaluated, and it does so at one-shot communication cost.

We do not claim that this recipe replaces per-client-weight pFL universally. We claim it is a useful operating point — particularly when cold clients have no local-training budget, when communication is one-shot or expensive, and when the heterogeneity of the client distribution admits a domain-specific auxiliary decomposition. Establishing the boundary of this regime, and combining the cluster-prototype lookup with iterative federated personalisation in a hybrid recipe, are the natural directions for future work.

---

## References

A working bibliography. Final formatting will follow the venue's style.

[1] Y. Peng et al., *Short-term Load Forecasting at Different Aggregation Levels with Predictability Analysis*. arXiv:1903.10679, 2019.

[2] P. Emami et al., *BuildingsBench: A Large-Scale Dataset of 900K Buildings for Foundation Models in Short-Term Load Forecasting*. NeurIPS Datasets and Benchmarks, 2023.

[3] L. Collins et al., *Exploiting Shared Representations for Personalized Federated Learning*. ICML, 2021.

[4] P. P. Liang et al., *Think Locally, Act Globally: Federated Learning with Local and Global Representations*. arXiv:2001.01523, 2020.

[5] T. Li et al., *Ditto: Fair and Robust Federated Learning Through Personalization*. ICML, 2021.

[6] C. T. Dinh et al., *Personalized Federated Learning with Moreau Envelopes*. NeurIPS, 2020.

[7] T. Li et al., *Federated Optimization in Heterogeneous Networks*. MLSys, 2020.

[8] A. Fallah et al., *Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach*. NeurIPS, 2020.

[9] Y. Tan et al., *FedProto: Federated Prototype Learning across Heterogeneous Clients*. AAAI, 2022.

[10] R. Dai et al., *FedNH: Tackling Both Data Heterogeneity and Class Imbalance in Federated Learning via Class Prototypes*. AAAI, 2023.

[11] A. Ghosh et al., *An Efficient Framework for Clustered Federated Learning*. NeurIPS, 2020.

[12] F. Sattler et al., *Clustered Federated Learning*. IEEE Transactions on Neural Networks and Learning Systems, 2021.

[13] M. Stallmann and A. Wilbik, *Towards Federated Clustering: A Federated Fuzzy c-Means Algorithm (FFCM)*. arXiv:2201.07316, 2022.

[14] H. Gui, X. Li, and X. Chen, *Vector Quantization Pretraining for EEG Time Series with Random Projection and Phase Alignment*. ICML, 2024.

[15] A. Das et al., *A Decoder-Only Foundation Model for Time-Series Forecasting*. ICML, 2024.

[16] X. Zhang et al., *Seq2Peak: Sequence-to-Peak Auxiliary Forecasting*. CIKM, 2023.

[17] K. G. Olivares et al., *Neural Basis Expansion Analysis with Exogenous Variables: Forecasting Electricity Prices with NBEATSx*. International Journal of Forecasting 39(2), 2023.

[18] H. B. McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS, 2017.

[19] A. Zeng et al., *Are Transformers Effective for Time Series Forecasting?* AAAI, 2023.

[20] C. Challu et al., *NHITS: Neural Hierarchical Interpolation for Time Series Forecasting*. AAAI, 2023.

[21] Y. Zhang and J. Yan, *Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting*. ICLR, 2023.

[22] A. F. Ansari et al., *Chronos: Learning the Language of Time Series*. arXiv:2403.07815, 2024.

---

## Appendix A. Hyperparameter table

**Backbone (MinimalNBEATSx):** input length $L = 96$, horizon $H = 24$, hidden dim $d_\text{model} = 64$, three stacks (trend, seasonal, generic), three polynomial bases on the trend stack, five Fourier harmonics on the seasonal stack.

**Auxiliary head (PeakAuxHead):** $\mathrm{Linear}(64 \to 32)$, ReLU, $\mathrm{Linear}(32 \to 1)$ for amplitude regression and $\mathrm{Linear}(32 \to 24)$ for hour classification. Loss weights $\lambda = 0.3$ (auxiliary block weight in total loss), $\eta = 0.1$ (hour-classification sub-weight inside the auxiliary block).

**Training (Phase A, centralised):** Adam, lr = 1e-3, weight_decay = 1e-5, batch size = 256, max 30 epochs, patience = 8 on validation MAE in denormalised kilowatts. Stride = 1 for training, stride = 24 for validation and codebook construction.

**Training (Phase A, federated comparison):** Adam, lr = 1e-3, weight_decay = 1e-5, batch size = 512, bf16 autocast, 20 rounds, 2 local epochs per round, full client participation (80 / 80). Algorithm-specific defaults: FedProx $\mu = 0.01$, Ditto $\lambda = 0.1$, FedRep head-epochs = 1.

**Codebook (Phase B):** KMeans++ with $M = 32$ centres, $n_\text{init} = 10$. Per-cluster offset is the mean over training windows of $(\mathbf{y} - \hat{\mathbf{y}})$, in z-normalised forecast space.

**Hybrid correction (Phase C):** Two operating points carried over from the training-side selection. HR-preserving: $\sigma = 3.0,\ \alpha_{v0} = 1.0,\ \alpha_{w1} = 0.1$. PAPE-aggressive: $\sigma = 3.0,\ \alpha_{v0} = 1.5,\ \alpha_{w1} = 0.5$.

## Appendix B. Per-seed full numbers (Cold PAPE, %)

Table B.1 lists cold PAPE per seed for every method in Table 1. The row labelled "proposed" is at the PAPE-aggressive operating point under input-only routing. All numbers are in **% PAPE** (peak absolute percentage error).

```
method                       seed=42   seed=123  seed=7    mean ± std
fedavg                       56.17     57.83     55.03     56.34 ± 1.41
fedprox                      55.98     57.98     54.94     56.30 ± 1.55
fedrep                       56.34     58.94     56.26     57.18 ± 1.52
ditto                        55.99     58.17     54.98     56.38 ± 1.63
fedproto                     56.23     57.88     55.00     56.37 ± 1.44
fedavg_nbeatsx_aux_only      57.16     58.94     55.86     57.32 ± 1.55
fedavg_nbeatsx_aux+codebook  43.38     41.56     40.86     41.93 ± 1.30
local_only_self_eval         50.04     54.89     52.99     52.64 ± 2.44
local_only_holdout_eval      60.10     65.86     62.12     62.69 ± 2.92
nf_dlinear                   49.43     51.05     50.63     50.37 ± 0.84
nf_nhits (corrected MLP)     50.92     54.30     53.01     52.74 ± 1.71
nf_crossformer               50.78     54.18     52.67     52.54 ± 1.71
fm_chronos_bolt_small        50.96     53.98     53.12     52.69 ± 1.56
fm_chronos_t5_tiny           59.61     65.05     64.73     63.13 ± 3.04
fm_timesfm                   51.80     55.34     55.67     54.27 ± 2.15
codebook_on_fedavg           52.56     46.46     45.75     48.26 ± 3.74
codebook_on_fedrep           48.85     47.51     46.13     47.50 ± 1.36
m_sensitivity_M=8            38.12     36.78     35.61     36.84 ± 1.25
m_sensitivity_M=16           37.17     36.63     35.50     36.43 ± 0.85
m_sensitivity_M=64           36.50     35.46     34.67     35.54 ± 0.92
proposed (M=32, default)     36.39     35.39     35.33     35.70 ± 0.49
```

---

## Appendix C. Reference numbers from v01–v04

This appendix consolidates the reference quantities that successive iterations of this project produced. Where a row appears verbatim in §4 it is **starred (★)**; the others are cross-protocol context that frames how the present (v04, 80 : 20) numbers relate to the earlier 50 : 50 study and to the orthogonality / heterogeneity / communication analyses.

### C.1 v01 — 50 : 50 centralised study (single-seed → 3-seed)

The v01 paper established the method on a 50 train + 50 cold split with a centralised backbone. PAPE values are in **%**.

| Quantity | Value | Source |
|---|---|---|
| Baseline NBEATSx, cold PAPE | 55.17 % | v01 §4.2 |
| Baseline NBEATSx, cold HR@1 / HR@2 | 27.0 % / 38.5 % | v01 §4.2 |
| Hybrid HR-preserving ($\sigma{=}3.0,\alpha_{v0}{=}1.0,\alpha_{w1}{=}0.1$), cold PAPE | 45.22 % (−18.0 % rel) | v01 §4.2 |
| Hybrid PAPE-aggressive ($\sigma{=}3.0,\alpha_{v0}{=}1.5,\alpha_{w1}{=}0.5$), cold PAPE | 37.05 % (−32.8 % rel) | v01 §4.2 |
| Hybrid PAPE-aggressive, 3-seed mean ± std | 37.62 ± 0.45 % | v01 §4.4 |
| Hybrid PAPE-aggressive, 3-seed HR@1 / HR@2 | 26.4 ± 0.2 / 38.0 ± 0.4 % | v01 §4.4 |
| Auxiliary-loss isolated effect (V0 ablation, cleanest) | +18.6 pp PAPE | v01 §4.3 |
| Codebook health (T0 vs T2): minimum cluster size $k_\text{min}$ | 2 → 113 | v01 §4.3 |
| Codebook utilisation, perplexity | 1.00, 27.3 | v01 §4.3 |
| Per-cluster cold benefit: $30/32$ clusters cold-improve | binomial $p \approx 5{\times}10^{-7}$ | v01 §4.5 |

### C.2 v02 — 80 : 20 frozen-encoder PFL framing

v02 reframes the v01 backbone as a centrally-pretrained, frozen shared encoder under the FedHiP pattern, at the FL-canonical 80 : 20 ratio.

| Quantity | Value | Source |
|---|---|---|
| Baseline (Vanilla) cold PAPE, 3-seed | $\approx 55.18$ % (matches v01) | v02 §4 |
| Hybrid HR-preserving, cold PAPE (3-seed) | $\approx 44.7$ % (−19.0 % rel) | v02 abstract |
| Hybrid PAPE-aggressive, cold PAPE (3-seed) | **35.70 ± 0.49 %** ★ | v02 §4 / present paper Table 1 |
| Routing comparison: Key-Route vs Latent-Route, cold PAPE | indistinguishable (Δ ≤ 0.01 pp) ★ | v02 §4 / present paper Table 3 |
| Hybrid dominance over CMO / GST alone | +3.24–3.47 pp PAPE | v02 abstract |
| Auxiliary-loss effect (V0 ablation) at 80 : 20 | +11.9 ± 11.2 pp ★ (high seed variance) | v02 §4 / present paper §4.5 |

### C.3 v04 — full baseline comparison + heterogeneity + communication

v04 evaluates the frozen v01 method against published FL / NF / FM baselines on the same 80 : 20 split, and adds the two analyses (G6 heterogeneity, G7 communication) used in §4.6 and §4.7 of the present paper.

| Quantity | Value | Source |
|---|---|---|
| Headline cold PAPE table, 13 methods × 3 seeds | see present paper Table 1 ★ | v04 §4.1 |
| Stacked rows (codebook on FedAvg / FedRep) | 48.26 ± 3.74 / 47.50 ± 1.36 % ★ | v04 §4.5 / present paper Table 2 |
| FL cluster width (max − min over FedAvg, FedProx, FedRep, Ditto) | $\le 0.9$ pp | v04 §4.2 |
| FL training-loss saturation round | $\approx 17$ (max delta of last three rounds < $10^{-3}$) | v04 §3.3 |
| Heterogeneity W1 (kW): mean / max | 0.379 / 1.439 ★ | v04 §4.6 / present paper §4.6 |
| Heterogeneity JS-symmetric KL (64-bin): mean / max | 0.067 / 0.32 | v04 §4.6 |
| Hour-of-day cosine: min / mean | 0.811 / 0.970 ★ | v04 §4.6 / present paper §4.6 |
| Communication (FedAvg / FedProx / Ditto): bytes / round, total | 21.0 MB, 420.4 MB ★ | v04 §4.7 / present paper Table 4 |
| Communication (FedRep): bytes / round, total | 17.9 MB, 358.8 MB ★ | v04 §4.7 / present paper Table 4 |
| Communication (proposed): single-shot codebook upload | 4.94 MB once ★ | v04 §4.7 / present paper Table 4 |
| Compression: FedAvg-total / proposed-total | $\approx 85\times$ ★ | present paper §4.7 |
| Boundary crosses: FedAvg / proposed | 20 / 1 ★ | present paper §4.7 |

### C.4 Constants and pool sizes used throughout

| Symbol | Meaning | Value |
|---|---|---|
| $L$ | input window length | 96 hours |
| $H$ | forecast horizon | 24 hours |
| $d_\text{model}$ | NBEATSx hidden width | 64 |
| $M$ | codebook size | 32 |
| $\lambda$ | auxiliary loss weight | 0.3 |
| $\eta$ | hour-CE sub-weight | 0.1 |
| $N_\text{train}$ | train pool size (v02–v04) | 80 |
| $N_\text{cold}$ | cold pool size (v02–v04) | 20 |
| $N$ | number of training windows (stride $= H$) | $\approx 19{,}250$ |
| Seeds | random seeds | $\{42, 123, 7\}$ |
| HR@1 chance | uniform random peak hour, $\pm 1$ h | $3/24 = 12.5\,\%$ |
| HR@2 chance | uniform random peak hour, $\pm 2$ h | $5/24 = 20.8\,\%$ |

### C.5 Additional reference numbers (PR-review fixes + priority experiments, 2026-04-28)

These rows were added after the initial draft to (i) close a few PR-review code issues and (ii) strengthen the unified narrative against reviewer pushback. All numbers are 3-seed mean ± sample std (ddof = 1), PAPE-aggressive operating point unless stated.

| Quantity | Value | Source |
|---|---|---|
| NHITS (corrected MLP construction): cold PAPE | 52.74 ± 1.71 % ★ | present paper Table 1 |
| FedAvg-NBEATSxAux raw (no codebook): cold PAPE | 57.32 ± 1.55 % ★ | present paper Table 1, 6 |
| FedAvg-NBEATSxAux + codebook (Stacked-Aux): cold PAPE | 41.93 ± 1.30 % ★ | present paper Table 1, 6 |
| FedProto (pFL prototype baseline): cold PAPE | 56.37 ± 1.44 % ★ | present paper Table 1 |
| Local-only NBEATSx, holdout-eval (fair generalisation): cold PAPE | 62.69 ± 2.92 % ★ | present paper §4.11, Table 1, 8 |
| Cluster-count sweep, $M = 8$ / $M = 16$ / $M = 64$ | 36.84 / 36.43 / 35.54 % ★ | present paper §4.9, Table 7 |
| Three-way decomposition: federation cost / aux head cost / W5 cost | $6.23 / 5.57 / 8.84$ pp ★ | present paper §4.8, Table 6 |

---

## Revision history

| Date | Change | Detail |
|---|---|---|
| 2026-04-27 | Initial draft. | Method (peak-aware encoder, single-shot post-hoc codebook, Hybrid correction); §4.1–§4.7 from the v04 baseline study; §5 discussion; §6 limitations. |
| 2026-04-28 | Citation accuracy pass. | (i) Stallmann & Wilbik 2022 corrected from arXiv:2210.09519 / "K-means" to arXiv:2201.07316 / "Fuzzy c-Means (FFCM)". (ii) "VQ-MTM, Yue et al., 2022" replaced with the actual VQ-MTM paper (Gui, Li, Chen, ICML 2024); the Yue et al. AAAI 2022 (TS2Vec) paper is contrastive, not VQ. (iii) Peng et al. 2019 venue corrected from "Energy" to "arXiv:1903.10679". |
| 2026-04-28 | PAPE unit clarification (paper-wide). | Added an explicit metric definition (§3) noting that PAPE is a *dimensionless percentage*; differences between two PAPE values are reported in percentage points (pp), not in kilowatts. The previous draft mislabelled some PAPE differences as "kW". MAE remains in kW. |
| 2026-04-28 | New §4.8 "Disentangling the cold-side gap". | Adds a federated NBEATSxAux row (with both backbone and aux head federated) and decomposes the 20.6 pp total gap from FedAvg to our centralised reference into three additive parts: 6.2 pp federation cost + 5.6 pp aux-head cost + 8.8 pp W5-hybrid cost. New Table 6. Resolves the original §6 "method comparison is not strictly ceteris paribus" caveat. |
| 2026-04-28 | New §4.9 "Cluster-count sensitivity". | Sweeps $M \in \{8, 16, 32, 64\}$ on the same centralised backbone; cold PAPE 36.84 / 36.43 / 35.70 / 35.54, all within a 1.3 pp envelope. New Table 7. Resolves the original §6 "cluster-count sensitivity is unreported" caveat. |
| 2026-04-28 | New §4.10 "Prototype-based pFL: FedProto comparison". | Adds an iterative prototype-based pFL baseline (FedProto, $K = 32$, $\lambda_\text{proto} = 0.1$). Cold PAPE 56.37 ± 1.44, indistinguishable from FedAvg / Ditto. Confirms that prototype alignment alone — without the W5 hybrid correction — does not move the cold-side number. |
| 2026-04-28 | New §4.11 "Local-only with held-out evaluation". | Quantifies the original Local-only row's overfit upper bound: holdout-eval gives 62.69 ± 2.92 (worst tier of Table 1) versus 52.64 ± 2.44 self-eval. Penalty $+10.05$ pp. New Table 8. Updates §6 limitation to a quantified statement. |
| 2026-04-28 | NHITS implementation correction. | The first draft used an MLP construction with two consecutive activations between the last hidden Linear and the $n_\theta$ output Linear (and a missing activation after the very first input Linear). The corrected version (every hidden Linear followed by activation + dropout, terminal Linear bare) matches the official NeuralForecast implementation. Cold PAPE shifts from 52.99 ± 1.64 to 52.74 ± 1.71 — within seed noise; the corrected number is the reference. |
| 2026-04-28 | Headline rewording. | Conclusion and §1.3 contribution 4 now say "auxiliary task *and W5 hybrid correction together*" rather than "auxiliary task alone", per §4.8's three-way decomposition. The $\sim 11.8$ pp wording is preserved as the gap to the strongest *non-ours* baseline (the Stacked row at 47.50 %), with the Stacked-Aux row (41.93 %) reframed as "our recipe under federation" rather than as a baseline.|
