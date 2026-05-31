Experiments – (1) Datasets
Google Flu Trends (GFT)
The dataset which consists of weekly estimates of influenza activity based on volume of certain search queries from 2003 to 2015.
Air-quality (AQ)
The dataset contains hourly averaged responses from an array of five metal oxide chemical sensors embedded in an air quality chemical multi-sensor device in Italy, which recorded from 2004 to 2005.
Traffic
The traffic dataset from the California Department of Transportation contains road occupancy rates (between 0 and 1) measured by 862 sensors in San Francisco Bay area freeways during 2015 and 2016. 

Experiments – (2) Baselines
TMF – AR-based temporal regularization
TRMF (Yu et al., 2016) – AR-based temporal regularization with Markov-chain structure
NoTMF (Chen et al., 2022) – VAR-based temporal regularization

Experiments – (3) Experimental setting
The evaluation task is to perform test for the next 12 weeks for GFT data and for the next 7 days (24 time-points at a time, for 7 windows) for AQ and Traffic data.
Models were trained independently to predict the target future step (horizon) 1, 3, and 6.
Five evaluation metrics
Mean absolute error (MAE)
Root mean squared error  (RMSE)
Mean absolute percentage error (MAPE)
Normalized deviation (ND)
Normalized RMSE (NRMSE)


Result

The proposed method shows the best results compared with the baselines in terms of all prediction accuracy measures for GFT data which contains missing values.

TMF-GNN obtained a statistically significant difference compared to the other baseline models for all prediction measures for GFT data.

For AQ data, TMF-GNN shows the superior performance to other baselines except for the ND measure.

Especially, the prediction accuracies of TMF-GNN for AQ data increased by 12% and 7% on average compared to TRMF (secondly ranked) in terms of MAE and MAPE, respectively, at horizon 1.


Model comparison results for traffic data
Different from GFT and AQ datasets, traffic data is non-missing data → artificial missing values are randomly generated for traffic data by changing missing rates from 20% to 40%.
TMF-GNN shows the better performances than baselines in terms of most prediction accuracy measures.


Contribution

TMF-GNN is an useful method for MTS forecasting when MTS data has missing values which can provide the biased information for modeling and degrade the predictive performance.
TMF-GNN demonstrated the improvement in model performance on MTS forecasting with missingness by capturing spatial and temporal patterns of MTS data, simultaneously.

Limitation

Several user-defined parameters for TMF-GNN, which affect the performance of the prediction model, should be optimally selected. 
TMF-GNN uses three loss functions to construct the final hybrid loss function, so it is important to find the optimal point and time to stabilize the hybrid loss function.


====================================================================
v09 – Round-wise Federated Peak-VQ Codebook (RoundCB)
====================================================================

Experiments – (1) Datasets
UMass Smart* (2016)
The dataset consists of hourly residential electricity load (kW) recorded from 114 individual apartments (households) over the year 2016. Each household is treated as a separate federated client, so the data is naturally non-IID across clients with strongly heterogeneous consumption levels and peak patterns.
Per-client construction
For every apartment we build sliding windows of 96 input hours (4 days) to forecast the next 24 hours (1 day), and split each household chronologically into 70% train / 10% validation / 20% test. Per-apartment z-normalization statistics (mean, std) are computed on the training portion only and applied to all splits; all reported metrics are computed back in raw kW space.

Experiments – (2) Baselines
Foundation-model zero-shot
TimesFM, Chronos-Bolt (small), and Chronos-T5 (tiny) — pretrained time-series foundation models evaluated zero-shot on each client's test split (no training), serving as a training-free lower bound.
Federated backbones
Five federated protocols all sharing the identical NBEATSx-Aux backbone (forecast head + peak-aux head): Centralised / FedAvg, FedProx, FedRep, Ditto, and FedProto. The codebook mechanism is applied orthogonally on top of each backbone.
Neural-forecasting backbones
Three trained deep forecasting architectures, learned in the centralised-pooled regime (every household's training windows pooled into one optimiser, no federation) and evaluated on each household's test split: DLinear / NHITS / Crossformer. They serve as a non-federated upper-bound on backbone capacity, isolating the cost of federation from the cost of the forecasting model itself.

Experiments – (3) Experimental setting
Task: per-client peak-aware day-ahead load forecasting.
Given the past 96 hours, predict the next 24 hours; tested on the last 20% of each household's series.
Federated training
Round-level optimization: 5 local epochs per round × 10 communication rounds
Optimizer: AdamW (lr = 1e-3, weight decay = 1e-5), batch size 512
Backbone (v06 invariants): NBEATSx-Aux, INPUT_SIZE = 96, HORIZON = 24, D_MODEL = 64, λ_aux = 0.3, hr_weight = 0.1
Peak-VQ codebook
Codebook: M = 32 entries, K_local = 2
Test-time correction: ŷ_corr = ŷ_base + α_v0 · offset[c*], α_v0 = 1.0
Four evaluation metrics
Peak Amplitude Percentage Error (PAPE) — |max(ŷ) − max(y)| / |max(y)| × 100, the primary metric
Hit Rate (HR@k) — share of windows whose predicted peak timing falls within k hours of the true peak (k = 1, 2, 3)
Mean Absolute Error (MAE)
Mean Squared Error (MSE, kW²)
All numbers are aggregated as the mean across the 114 clients, and reported as mean ± std across seeds {42, 123, 7}.


Contribution

RoundCB is a backbone-agnostic peak-correction mechanism: the same post-hoc Peak-VQ codebook plugs orthogonally into all five federated protocols (FedAvg, FedProx, FedRep, Ditto, FedProto) and the centralised reference, consistently reducing peak amplitude error (ΔPAPE ≈ −5 to −6) without altering backbone training.
RoundCB is privacy-conscious and communication-light: the codebook is fit from cluster centroid sums/counts so raw representations never leave the client, and the post-hoc design adds negligible state to the federated round.
Post-hoc round-wise fitting is shown to be more stable than joint VQ co-training in the federated regime: jointly trained FedVQ suffers codebook collapse (low utilization / low perplexity) and yields no correction gain, whereas the post-hoc RoundCB codebook stays fully utilized and delivers the peak-error reduction.

Limitation

Strong zero-shot foundation models remain competitive: pretrained time-series models (e.g., Chronos-Bolt) match or exceed the corrected forecaster on peak metrics, so the practical justification for federated training must rest on communication cost, on-device privacy, and per-client adaptation rather than raw accuracy alone.
The correction mainly reshapes magnitude, not timing: it improves peak amplitude (PAPE) but gives little gain in peak-timing hit rate (HR@k) and can slightly raise MAE.
Round-wise codebook broadcasting enlarges the training-time attack surface on representations (TAR) relative to single-shot fitting; the mass-weighted aggregation and EMA blending mitigations are only partial, leaving secure aggregation / differential-privacy noise as future work.
Key hyperparameters (codebook size M, correction strength α_v0, λ_aux) are fixed to the v06 invariants rather than tuned for v09, and the federated results are not yet aggregated over the full seed set {42, 123, 7}.

