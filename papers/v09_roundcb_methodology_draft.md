## Methodology: The RoundCB Framework

In this section, we present the architecture of **RoundCB** (Round-wise Federated Residual Codebook), a federation-preserving framework for peak-aware residential load forecasting. As illustrated in Figure \ref{fig:methodology_diagram}, RoundCB couples a shared forecasting backbone with a codebook that is reconstructed at every communication round, orchestrating five components—a *Residual Encoder*, a *Local Quantizer*, a *Codebook Aggregator*, a *Residual Offset Estimator*, and a *Corrector*—to turn the patterns a forecaster fails to explain into a reusable, peak-correcting memory. The framework proceeds in two phases: a **Federated Representation Phase**, in which the backbone is trained across households without ever touching the codebook, and a **Round-wise Codebook Phase**, in which the residual latents produced by the frozen backbone are quantized, aggregated, and turned into a corrective offset table. Crucially, raw latents and raw residuals never leave a household; only cluster-level summaries are communicated (see Appendix \ref{app_sec:federation_contract} for the communication contract).

### Task Formulation

We consider a federation of $K$ households $\mathcal{H} = \{h_i\}_{i=1}^{K}$, each holding a private set of training windows $\mathcal{D}_i$. Every window is a pair $(x, y)$ where $x \in \mathbb{R}^{L}$ is a look-back of length $L$ and $y \in \mathbb{R}^{H}$ is the forecast horizon, both $z$-normalized with statistics estimated on the household's training segment only. The goal is to produce, for each household, a corrected forecast $\hat{y}$ that improves peak-amplitude accuracy without communicating raw load data. The learnable state comprises the backbone parameters $\theta$, a global codebook $C \in \mathbb{R}^{M \times d}$, and a residual offset table $O \in \mathbb{R}^{M \times H}$.

### Residual Encoder

The Residual Encoder $f_\theta$ is a three-stack NBEATSx model that performs iterative residual decomposition. Initializing the residual as $r^{(0)} = x$, each stack $s \in \{\text{trend}, \text{seasonal}, \text{generic}\}$ consumes the running residual, emits a backcast $b^{(s)}$, a forecast $f^{(s)}$, and a hidden representation $h^{(s)}$, and passes the unexplained signal to the next stack:
$$
\big(b^{(s)},\, f^{(s)},\, h^{(s)}\big) = \text{Stack}_s\!\big(r^{(s-1)}\big), \qquad r^{(s)} = r^{(s-1)} - b^{(s)}.
$$
The forecast is the sum of the per-stack forecasts, and we designate the generic-stack hidden as the **residual latent** $z$:
$$
(\hat{y},\, z) = f_\theta(x), \qquad \hat{y} = \!\!\sum_{s}\! f^{(s)}, \qquad z \triangleq h^{(\text{generic})} \in \mathbb{R}^{d}.
$$
Because the generic stack operates on $r^{(2)}$—the residual remaining after the trend and seasonal bases have been subtracted—$z$ encodes precisely the component of the input that the structured bases could not explain. It is this residual latent, rather than the raw forecast, that RoundCB clusters and corrects against.

### Federated Representation Phase

The backbone is trained by standard federated optimization. At round $r$, the server broadcasts $\theta^{(r)}$; each household runs $E$ local epochs of SGD on the peak-aware objective and uploads its update, which the server averages by training-window count:
$$
\theta_i^{(r)} = \text{LocalSGD}\!\big(\theta^{(r)},\, \mathcal{D}_i\big), \qquad
\theta^{(r+1)} = \text{FedAvg}\!\big(\{\theta_i^{(r)}\}_{i=1}^{K}\big),
$$
where the local objective is $\mathcal{L} = \text{MAE}(\hat{y}, y) + \lambda_{\text{aux}} \cdot \ell_{\text{aux}}(\hat{a}, \hat{h}, y)$. The codebook does **not** enter $\mathcal{L}$: representation learning is driven solely by the residual decomposition and the peak-auxiliary head, leaving the codebook a strictly *post-hoc* observer of the backbone.

### Local Quantizer

At the end of round $r$, each household freezes the current backbone and encodes its training windows into a set of residual latents $Z_i = \{\, z : (\cdot,\, z) = f_\theta(x),\ x \in \mathcal{D}_i \,\}$. The Local Quantizer $q$ fits a local $K_{\text{local}}$-means over these latents, returning prototype centroids and their occupancy counts:
$$
\big(\mu_i,\, n_i\big) = q\!\big(Z_i; K_{\text{local}}\big) = \text{KMeans}\!\big(Z_i,\, K_{\text{local}}\big), \qquad \mu_i \in \mathbb{R}^{K_{\text{local}} \times d}.
$$
Only the pair $(\mu_i, n_i)$ is uploaded; the latents $Z_i$ remain on-device.

### Codebook Aggregator

The Codebook Aggregator $\mathcal{A}$ reconstructs the global codebook on the server by stacking all households' prototypes and refitting a mass-weighted $M$-means, where each prototype is weighted by its local count:
$$
C = \mathcal{A}\!\big(\{(\mu_i, n_i)\}_{i=1}^{K}\big) = \text{KMeans}\!\Big( \textstyle\bigcup_i \mu_i,\ M;\ \text{weight} = \textstyle\bigcup_i n_i \Big) \in \mathbb{R}^{M \times d}.
$$
This two-stage hierarchical clustering yields a codebook statistically equivalent to one fit on the pooled latents, while keeping the latents private.

### Residual Offset Estimator

Whereas the codebook captures the geometry of *input-side* residual latents, the Residual Offset Estimator $\mathcal{O}$ captures the *output-side* forecast error associated with each codebook entry. Each household routes its latents to their nearest codebook entry and reports, per cluster, the partial sum of its forecast residuals $\rho(x) = y_z - \hat{y}_z$ together with the cluster count; the server averages cluster-wise:
$$
c^{*}(x) = \arg\min_{m} \big\| z - C_m \big\|_2, \qquad
O_m = \mathcal{O}\big(\{(Z_i, \rho_i)\}_i,\, C\big)_m = \frac{\sum_i \sum_{x:\, c^{*}(x)=m} \rho(x)}{\max\!\big(\sum_i \big|\{x:\, c^{*}(x)=m\}\big|,\, 1\big)}.
$$
Each row $O_m \in \mathbb{R}^{H}$ is therefore the *typical horizon-shaped error* of the windows routed to entry $m$—a prototype memory of how the backbone systematically mispredicts each residual mode. Empty entries receive a zero offset.

### Corrector

At inference, the Corrector $g$ forwards a test window through the frozen backbone, routes its residual latent to the codebook, and adds the corresponding offset, yielding the corrected forecast:
$$
(\hat{y}_{\text{base}},\, z) = f_\theta(x), \qquad
\hat{y} = g\big(\hat{y}_{\text{base}},\, z;\, C, O\big) = \hat{y}_{\text{base}} + \alpha \cdot O_{\,c^{*}(z)},
$$
with correction strength $\alpha$ (a fixed operating point, not re-tuned on test data). Predictions are then denormalized to physical units with the household's $(\mu_a, \sigma_a)$. The Local-Quantizer → Aggregator → Offset → Corrector cycle is executed at the end of **every** communication round, so RoundCB produces not a single codebook but a per-round trajectory $\{(C^{(r)}, O^{(r)})\}_{r=1}^{R}$, enabling analysis of how the corrective lift co-evolves with the backbone.

---

**Figure 1: Overview of our RoundCB framework.** Given per-household load windows and a peak-aware forecasting objective, we first apply a *Federated Representation Phase* in which a shared NBEATSx Residual Encoder is trained across households by FedAvg, exposing a residual latent that captures the patterns left unexplained by the trend and seasonal bases. We then apply a *Round-wise Codebook Phase*: at the end of each communication round, the Local Quantizer fits per-household prototypes over the frozen latents, the Codebook Aggregator merges them by mass-weighted clustering into a global codebook, and the Residual Offset Estimator aggregates per-cluster mean forecast residuals into an offset table. At inference, the Corrector routes each test latent to its nearest codebook entry and adds the corresponding residual offset to produce the final peak-corrected forecast. Throughout, raw latents and residuals never leave the client—only cluster centroids, counts, and residual partial sums are communicated.
