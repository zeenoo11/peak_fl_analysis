# v05 FedCB Codebook: Federated Codebook Construction for Privacy-Preserving Peak Correction

## §1 Motivation

In the v02 / v04 pipeline, the VQ codebook is built centrally: all training-window latents
(`h_generic`) are shipped to a single server, KMeans++ is fitted in one shot, and the resulting
centroids are broadcast back. This centralised approach requires each client to upload its full
latent pool — approximately 19,250 windows × 64 floats ≈ 4.93 MB in one boundary crossing.

v05 replaces that single centralised step with a hierarchical two-stage federated protocol:

1. **Stage 1 (local)**: each client runs a small KMeans++ on its own `h_g` pool and sends only
   `K_local` centroids plus counts to the server. Raw latents never leave the client.
2. **Stage 2 (server)**: a global KMeans++ is fitted on the uploaded centroid pool
   (weighted by per-cluster counts), producing an `M=32` codebook that is broadcast back.
3. **Stage 3 (residual)**: each client routes its training windows to the nearest Stage-2
   centroid and sends per-cluster residual sums and counts; the server divides to get
   cluster-mean offsets.

Phase C correction is CMO-only (`ŷ_corr = ŷ_base + α · offset[c*]`), dropping the Gaussian
template term to isolate the effect of federated codebook quality. The method compares against
a v02 §B.3 published anchor (V5-FedCB-0) that uses the centralised codebook under R0 routing
so that any performance gap is attributable solely to the federated construction path.

## §2 Method

### Backbone

The backbone is the FedAvg-NBEATSxAux model trained in v04 `09_fix_rerun`
(`outputs/v04_full_baseline_comparison/09_fix_rerun/seed{S}/fedavg_nbeatsx_aux/final_state_dict.pt`),
loaded with `strict=True` and frozen throughout. No backbone weights are modified in v05.

### Federated codebook construction (`src/fl/codebook_fl.py`)

- **Stage 1**: `local_codebook_step()` — per-client KMeans++ with `K_local_i = min(K_local, N_i)`.
  Raw `h_g` remains on the client; only `(centroids, counts)` are transmitted.
- **Stage 2**: `merge_local_codebooks()` — server-side weighted KMeans++ on the centroid pool,
  `sample_weight = repeat(counts, ...)`, `init='k-means++'`, `n_init=10`.
- **Stage 3**: `federated_residual_offsets()` — per-cluster aggregation of `(y_true_z − ŷ_z)`
  residuals; empty clusters receive zero offset.

### CMO correction

Cold routing uses R1 (nearest Stage-2 centroid by `h_g` 1-NN). Correction:
```
ŷ_corr = ŷ_base + α · offset[c*]
```
with `α=1.0` for V5-FedCB-1/2a/2b and no α sweep (Stage D skipped, see §3).

### V5-FedCB-0 anchor (data-only)

V5-FedCB-0 is **not re-run** by v05. `02_aggregate.py` loads
`outputs/v02_fl_8020_ratio/seed{S}/W_component_results.json` →
`per_operating_point["PAPE-aggressive"]["cells"]["V0"]`
and inserts it as a virtual cell `v5_fedcb_0_paper_anchor`.
This anchor uses **R0 routing** (KEY 5-d pool 1-NN), the v02 T2 backbone, and
`α_v0=1.5` with `α_w1=0` (CMO-only equivalent). Readers should note the routing
difference: V5-FedCB-1/2a/2b use R1 (`h_g` nearest centroid) while the anchor uses R0.

## §3 Experimental Setup

### Data and splits

UMass Smart* 2016, 100 apartments. v02 80:20 train/cold split
(`outputs/v02_fl_8020_ratio/splits/`), loaded via `load_v02_split(seed)`.
80 training clients, 20 cold apartments.

### Protocol

Multi-seed: seeds `{42, 123, 7}`, mean ± std (sample std, ddof=1) reported.

Per-seed driver: `experiments/v05_fedcb_codebook/01_fedcb_codebook.py --seed S --K_local K`.
Each invocation is independent; no seed loop inside the script.

### Experimental cells

| Cell label        | K_local | α    | Stage D? | Notes                          |
|-------------------|---------|------|----------|--------------------------------|
| V5-FedCB-0        | —       | 1.5  | —        | Data-only anchor from v02 §B.3 |
| V5-FedCB-1        | 4       | 1.0  | —        | Default federated              |
| V5-FedCB-2a       | 2       | 1.0  | —        | K_local sweep                  |
| V5-FedCB-2b       | 8       | 1.0  | —        | K_local sweep                  |

Stage D (α sweep) was **gated on Gate 2 miss** (mean PAPE > 52%). Gate 2 result:
mean PAPE = 50.17% ≤ 52.0% → **Stage D skipped**.

### Hyperparameters (fixed across all cells)

`M=32`, `batch_size=512`, `stride=24` (= HORIZON), `D=64` (h_generic dim), `INPUT_SIZE=96`.

### Aggregation script and output

`experiments/v05_fedcb_codebook/02_aggregate.py` → `outputs/v05_fedcb_codebook/multiseed_summary.json`

`experiments/v05_fedcb_codebook/03_communication.py` → `outputs/v05_fedcb_codebook/communication_summary.json`

## §4 Results

### Table 1: 3-seed mean ± std, PAPE / HR@1 / HR@2 / MAE

All numbers are on the cold split. PAPE = peak absolute percentage error (lower is better).
HR@k = peak-hour hit rate within ±k steps (higher is better). MAE in kW (lower is better).
V5-FedCB-0 fl_only column is not applicable (no FL backbone involved in the anchor path).

| Cell                      | Routing | PAPE (%)        | HR@1 (%)        | HR@2 (%)        | MAE (kW)        |
|---------------------------|---------|-----------------|-----------------|-----------------|-----------------|
| V5-FedCB-0 (paper anchor) | R0      | 44.18 ± 0.18    | 25.86 ± 2.59    | 37.12 ± 2.42    | 0.449 ± 0.025   |
| V5-FedCB-1 (K_local=4)    | R1      | 50.17 ± 0.971   | 25.28 ± 1.303   | 37.24 ± 1.862   | 0.439 ± 0.023   |
| V5-FedCB-2a (K_local=2)   | R1      | 50.70 ± 1.287   | 25.17 ± 1.727   | 37.08 ± 2.312   | 0.439 ± 0.023   |
| V5-FedCB-2b (K_local=8)   | R1      | 50.37 ± 1.130   | 25.22 ± 1.759   | 37.27 ± 2.407   | 0.439 ± 0.023   |

Note: V5-FedCB-0 uses the v02 T2 centralised backbone with R0 routing and α_v0=1.5.
V5-FedCB-1/2a/2b use the v04 FedAvg-NBEATSxAux backbone with R1 routing and α=1.0.
The ~6 pp PAPE gap between the anchor and the federated cells conflates (i) backbone
difference (v02 T2 vs v04 FedAvg), (ii) routing difference (R0 vs R1), and (iii) codebook
construction difference (centralised vs federated). Disentanglement is deferred to exp-critic.

### Table 2: Communication cost per K_local (Stage E, seed-independent)

Formulas (plan §4.7):
- `local_upload_per_client = K_local × (D + 1) × 4` bytes
- `broadcast = M × D × 4` bytes
- `residual_per_client = M × (H + 1) × 4` bytes
- `bytes_per_round_total = N_clients × (local_upload + residual_per_client) + broadcast`

N_clients=80, D=64, M=32, H=24.

| K_local | Per-client/round (B) | Total/round (B) | Rounds | Total (B) | Boundary crosses |
|---------|---------------------|-----------------|--------|-----------|------------------|
| 2       | 3,720               | 305,792         | 1      | 305,792   | 2                |
| 4       | 4,240               | 347,392         | 1      | 347,392   | 2                |
| 8       | 5,280               | 430,592         | 1      | 430,592   | 2                |

For comparison (from v04 paper §4.7):
- FedAvg/FedProx/Ditto: 420,377,600 B total, 20 rounds, 20 boundary crosses
- v01/v03 centralised codebook (one-shot): 4,939,264 B total, 1 round, 1 boundary cross

The federated codebook path reduces communication vs. centralised by a factor of ~14×
(347,392 vs. 4,939,264 B) while requiring 2 boundary crosses instead of 1.

### Gate summary

| Gate   | Criterion                                      | Result         | Status |
|--------|------------------------------------------------|----------------|--------|
| Gate 1 | V5-FedCB-0 anchor loaded (3/3 seeds parsed)    | 44.180 ± 0.18  | PASS   |
| Gate 2 | V5-FedCB-1 (K_local=4) mean PAPE ≤ 52%        | 50.17%         | PASS   |
| Gate 3 | At least one K_local/α sweep cell clears 52%   | 50.70%, 50.37% | PASS   |

Stage D (α sweep) skipped because Gate 2 passed.

## §5 Discussion

(Reserved for exp-critic and reporter.)

## §6 Conclusion

(Reserved for exp-critic and reporter.)
