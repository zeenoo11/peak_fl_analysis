# v05-01 — Fully-federated codebook construction (FedCB), CMO-only correction

> Successor to `v04-01_full_baseline_comparison.md`. v04 plan §"What is NOT
> in scope" entry "Federated KMeans / DP-KMeans — orthogonal axis (privacy
> of the codebook itself), v05+" is the parent line item this plan opens.

> **Status (2026-04-29).** Plan only. Implementation gated on V5-FedCB-0
> reproduction (Gate 1 below).

## Motivation

`papers/pfl_unified/paper.md` §4.2 Table 1 reports a **Stacked-Aux** row at
**41.93 ± 1.30 % cold PAPE** — produced by
`experiments/v04_full_baseline_comparison/09_fix_rerun/02_fedavg_nbeatsx_aux.py`
and framed as "our recipe under federation". The pipeline has three
phases:

1. **Phase A** — FedAvg over the 80 train apts of the full NBEATSxAux
   under the combined loss `L = MAE(ŷ, y) + 0.3 · peak_aux_loss(...)`.
   Federated.
2. **Phase B** — `gather_train_segment_aux()` runs *server-side*: server
   reads every train apt CSV, forwards the frozen federated backbone
   over all train windows, and gathers per-window `h_generic`,
   forecast residuals, and 5-d KEY descriptors. KMeans++ (M=32) is fit
   on the gathered pool. **Centralised — this is the FL violation.**
3. **Phase C** — frozen-backbone forward on cold apts; W5 hybrid
   correction `ŷ_corr_z = y_hat_z + α_v0 · o_{c*} + α_w1 · g(t; ĥ, â, σ)`
   at the two op-points (HR-preserving / PAPE-aggressive) carried over
   from v01 §4.2.

KIIE oral-presentation review (2026-04-29) flagged Phase B as a framing
weakness — readers cannot describe the system as "fully FL" without an
asterisk (paper.md §5.4 already discloses this honestly). v05 closes
that gap with **two simplifications relative to the parent recipe**:

- **Phase B → federated** (the v05 method body): hierarchical 2-stage
  single-shot KMeans (local Stage 1 + server Stage 2 + federated
  residual aggregation), keeping CLAUDE.md's "post-hoc 1-shot" property.
- **Phase C → CMO-only** (a deliberate method simplification): drop
  the Gaussian template `α_w1 · g(t; ĥ, â, σ)`. The W5 ablation in
  `papers/v02_draft/v02_fl_8020_ratio.md` §B.3 (Table at line 490–494)
  shows that on the v02 centralised T2 backbone, **CMO alone reaches
  44.18 ± 0.18 PAPE-aggressive** versus 35.70 for full Hybrid — a
  controlled 8.5 pp give-up that buys: (a) a smaller Phase C surface
  (no per-window peak prediction broadcast), (b) a cleaner narrative
  ("federated codebook of cluster-mean offsets", not "federated codebook
  + per-window template generator"), (c) one fewer hyperparameter (single
  α replaces (α_v0, α_w1, σ) at two op-points).

The auxiliary head stays **inside Phase A training** — its loss term
makes `h_generic` peak-aware and is the reason `h_generic` clusters
meaningfully under KMeans (paper.md §3.2). The head's outputs `(â, ĥ)`
are simply unused at inference.

## Goals

**G1.** **Centralised CMO reproduction (Gate 1 baseline).** Reproduce
the v02 paper §B.3 PAPE-aggressive CMO row:

```
backbone      = v02 T2 (centralised NBEATSxAux + aux loss)
codebook      = centralised KMeans++ (M=32) on h_generic
correction    = α_v0 · o_{c*}  with (σ=3.0, α_v0=1.5, α_w1=0.0)
target PAPE   = 44.18 ± 0.18  (per-seed 43.99 / 44.21 / 44.34)
```

This is a *code-correctness* check — the same backbone + same codebook
+ same op-point as v02 paper, just W5's α_w1 forced to 0. If it lands
within ±1.5 pp of 44.18, the v05 inference and metric pipeline is
proven correct independently of the federation question.

**G2.** **Hierarchical federated codebook + CMO (method body).**
Replace centralised KMeans++ with a 2-stage *single-shot* federated
KMeans on top of the FedAvg-NBEATSxAux backbone (Phase A reused from
`09_fix_rerun/02_fedavg_nbeatsx_aux.py`):
- **Stage 1 (per client)**: local KMeans++ with `K_local` clusters;
  raw `h_g` never leaves the client.
- **Stage 2 (server)**: weighted second-pass KMeans++ on stacked local
  centroids (`sample_weight` = local cluster count) → global `M=32`
  codebook.
- **Federated residual offsets**: each client routes locally and returns
  per-cluster residual partial sums + counts; server averages.

CLAUDE.md's "post-hoc 1-shot" requirement holds: each KMeans is fit
once with no iterative centroid disclosure (TAR attack arxiv:2511.07073
surface unchanged from the centralised baseline).

Target: V5-FedCB-1 (`K_local=4`) at ≤ **52 % cold PAPE** (≈ 8 %
relative improvement over the standard FL baseline FedAvg-NBEATSxAux raw
at 56–57 %, Table 1 in `papers/pfl_unified/paper.md`). The 52 % bar
is set deliberately above the V5-FedCB-0 anchor (44.18) — V5-FedCB-1's
backbone is FedAvg, not centralised T2, so the published CMO number is
not directly attainable; a ~8 pp FedAvg-vs-centralised gap is consistent
with paper.md §4.2 / §4.8 (federated NBEATSxAux pays +6.23 pp under
W5; CMO-only is expected to track in the same direction).

**G3.** **K_local sensitivity.** Sweep `K_local ∈ {2, 4, 8}`. If at
least one setting clears the 52 % bar, the KIIE talk can be reframed
as "fully-FL framework with cluster-mean correction".

## Non-goals

- **Iterative federated KMeans** (Stallmann & Wilbik 2022). Repeated
  centroid disclosure re-opens the TAR surface; CLAUDE.md keeps this out
  of scope through v04 and v05 does not lift it.
- **SecAgg / DP-KMeans on local centroid uploads.** The
  `(K_local × 64)` Stage 1 payload still carries gradient-free signal
  about each client's `h_g` distribution. Quantifying or masking that
  is a separate ADR (v06+).
- **Hybrid (W5) correction at inference.** Dropped on purpose; v05
  reports CMO-only. The Hybrid number stays available via paper.md
  §4.2 Stacked-Aux for any reader who wants the comparison.
- **Two-op-point evaluation.** v05 uses a single `α` (default 1.0).
  V5-FedCB-3 sweeps `α ∈ {0.5, 1.0, 1.5, 2.0}` once to settle the
  default; then it is frozen.
- **Method re-design** of the encoder, aux loss, or peak descriptor —
  v01–v04 design is frozen.
- **Backbone re-training.** V5-FedCB-1+ reuses
  `outputs/v04_full_baseline_comparison/09_fix_rerun/seed{S}/fedavg_nbeatsx_aux/final_state_dict.pt`.
  V5-FedCB-0 reuses `outputs/v02_fl_8020_ratio/seed{S}/T2/best.pt`. No
  new training in either case.
- **R0 KEY-pool routing.** Per-window raw KEY would re-introduce
  centralised leakage. v05 uses **R1 only** (h_g 1-NN on the global
  codebook), matching paper.md §3.4.

## Method

### Phase A — Backbone (unchanged; two artefacts depending on cell)

| Cell | Backbone artefact | Source script |
|---|---|---|
| V5-FedCB-0 | (no new forward — V0-only metrics already in `outputs/v02_fl_8020_ratio/seed{S}/W_component_results.json`) | `experiments/v02_fl_8020_ratio/06_W_component_ablation.py` (legacy; *not re-run*) |
| V5-FedCB-1, 2, 3 | `outputs/v04_full_baseline_comparison/09_fix_rerun/seed{S}/fedavg_nbeatsx_aux/final_state_dict.pt` (FedAvg-NBEATSxAux + aux loss) | `experiments/v04_full_baseline_comparison/09_fix_rerun/02_fedavg_nbeatsx_aux.py` |

Both backbones are NBEATSxAux trained with the combined loss
`L = MAE + 0.3 · peak_aux`; only the *training protocol* differs
(centralised pooled vs FedAvg). The auxiliary head's role stops at
training-time `h_generic` peak-awareness (CLAUDE.md §"Method"); v05
inference does not call the aux head.

If a backbone artefact is missing for a seed the v05 driver halts and
points at the upstream script — do **not** silently re-train.

### Phase B — Codebook construction

#### B-centralised (V5-FedCB-0 — data already on disk)

V5-FedCB-0 does **not** require any new code or re-runs. The v02 ablation
script `experiments/v02_fl_8020_ratio/06_W_component_ablation.py` already
produced `V0-only` (= CMO) cells at the PAPE-aggressive op-point on R0
KEY-routing for all three seeds, and the results are persisted at:

```
outputs/v02_fl_8020_ratio/seed{S}/W_component_results.json
  → per_operating_point["PAPE-aggressive"]["cells"]["V0"]  # CMO PAPE/HR/MAE
```

Per-seed PAPE: 43.99 / 44.21 / 44.34 → 44.18 ± 0.18 (paper.md §B.3 Table
line 492). The aggregator (Build order step 3) reads these JSONs directly;
Gate 1 is therefore a **paper-load check**, not a wall-clock re-run.

Note that V5-FedCB-0 uses **R0 routing** (KEY pool 1-NN) per the v02
published anchor — V5-FedCB-1+ uses R1 (h_g 1-NN). The cells therefore
differ on backbone *and* routing, reinforcing Open Question E's "do not
treat V5-FedCB-0 as a head-to-head baseline for V5-FedCB-1; it is a
code-validation reference for the CMO inference path on a known-good
backbone".

#### B' — Federated (V5-FedCB-1, 2, 3)

##### B'-1. Local KMeans (each client `i`)

For each client `i`:

1. Forward client `i`'s training windows (stride=24) through the frozen
   FedAvg-NBEATSxAux backbone → `H_i ∈ R^{N_i × 64}` (stays on client).
2. Run `sklearn.cluster.KMeans(n_clusters=K_local_i, init='k-means++',
   n_init=10, random_state=seed).fit(H_i)`.
   - `K_local_i = min(K_local, N_i)` to handle tiny clients
     (open-question B).
3. Emit upload payload:
   - `C_i ∈ R^{K_local_i × 64}` — local centroids.
   - `n_{i,c} ∈ Z^{K_local_i}` — local cluster counts (`Σ_c n_{i,c} = N_i`).

Per-client upload at `K_local=4`: `4 × 64 × 4 + 4 × 4 = 1040` bytes
(≈ 1 KB).

##### B'-2. Server merge

```
P = vstack([C_i for i]) ∈ R^{(Σ K_local_i) × 64}
w = concat([n_{i,c} for i]) ∈ Z^{Σ K_local_i}
KMeans(n_clusters=M=32, init='k-means++', n_init=10,
       random_state=seed).fit(P, sample_weight=w)
→ C_global ∈ R^{32 × 64}
```

`C_global` is registered into a `VectorQuantizerKMeans(M=32, D=64)`
buffer for downstream API compatibility (cold-side `route_R1`). Server
broadcasts `C_global` (32 × 64 × 4 = 8192 bytes, once per round).

##### B'-3. Federated residual offsets

Each client `i`:
1. Routes its `H_i` through `C_global` → `c*_i ∈ Z^{N_i}` (h_g 1-NN,
   no extra forward pass).
2. Computes per-cluster z-norm residual partial sums (re-uses the
   forecasts already produced for B'-1):
   - `r_{i,c} = Σ_{j: c*_i[j] = c} (y_true_z[j] − y_hat_z[j]) ∈ R^{24}`,
   - `m_{i,c} = #{j: c*_i[j] = c} ∈ Z`.
3. Uploads `(r_{i,c}, m_{i,c})_{c=0..M-1}` (32 × 24 + 32 = 800 floats
   ≈ 3.2 KB).

Server averages cluster-wise:

```
o_c = (Σ_i r_{i,c}) / max(Σ_i m_{i,c}, 1)  ∈ R^{24}
```

Raw window-level residuals never reach the server; only cluster-aggregated
sums do.

### Phase C — Cold inference (CMO-only)

For every cold apt (warm-start z-norm, stride=24, frozen backbone
forward):

```
co = gather_cold(cold_apts, model, batch=512, stride=24)
cold_cluster = route_R1(co["h_g"], C_global)              # h_g 1-NN
corrected_z  = co["y_hat_z"] + α * o[cold_cluster]        # CMO only
metrics      = metrics_z_to_kw(co["y_true_z"], corrected_z, co["mean"], co["std"])
```

`α` default = 1.0 (V5-FedCB-1, 2). `α ∈ {0.5, 1.0, 1.5, 2.0}` for the
V5-FedCB-3 sweep. **No `gauss_template` call; aux-head outputs `(â, ĥ)`
ignored.** `gather_cold` from `src/eval/cold_helpers.py` is reused
verbatim — its `pred_amp` / `pred_hr` fields are simply not consumed.

## Experimental matrix

Five training/eval cells, all on seeds `{42, 123, 7}`, plus one
seed-independent communication cell.

| Cell | Backbone | Codebook | α | Reference / role |
|---|---|---|---|---|
| **V5-FedCB-0** | v02 T2 (centralised) | centralised, R0 routing | 1.5 | v02 §B.3 PAPE-aggressive CMO reference. **Already on disk** at `outputs/v02_fl_8020_ratio/seed{S}/W_component_results.json` `per_operating_point["PAPE-aggressive"]["cells"]["V0"]` — 44.18 ± 0.18 (per-seed 43.99 / 44.21 / 44.34). No re-run needed; aggregator loads it. **Gate 1 = paper-load check.** |
| **V5-FedCB-1** | FedAvg-NBEATSxAux | federated, K_local=4 | 1.0 | Headline v05 result. **Default**: server pool size = 80 × 4 = 320 (10× M=32) — Stage 1 produces ~10 input centroids per final cluster, a balanced split of representation load between client and server. Target ≤ 52 %. **Gate 2 / 3.** |
| **V5-FedCB-2a** | FedAvg-NBEATSxAux | federated, K_local=2 | 1.0 | Stage 1 capacity **floor** — each client emits only 1–2 centroids, suppressing intra-client pattern diversity (server pool size = 80 × 2 = 160; ratio to M=32 is 5×). If Gate clears here, the local clustering load can be trivial — the server's Stage 2 alone is doing nearly all of the representation work. |
| **V5-FedCB-2b** | FedAvg-NBEATSxAux | federated, K_local=8 | 1.0 | Stage 1 capacity **ceiling** — server pool size = 80 × 8 = 640 (20× M); approaches centralised KMeans behaviour as the input pool grows. Measures the marginal gain of extra K_local at 2× the upload cost relative to default. |
| **V5-FedCB-3** (opt) | FedAvg-NBEATSxAux | federated, K_local=4 | {0.5, 1.0, 1.5, 2.0} | α default determination. Skippable if Gate 2 already passes at α=1.0. |
| **V5-FedCB-4** | — | — | — | seed-independent communication accounting (paper.md §4.7 format) for K_local ∈ {2, 4, 8}. |

V5-FedCB-0 uses α=1.5 because v02 §B.3's CMO row was reported at the
PAPE-aggressive op-point (`α_v0=1.5`); reproducing the published number
requires the published α. V5-FedCB-1+ uses α=1.0 by default — the
"natural" residual-correction strength — and revisits via V5-FedCB-3 if
needed.

## Go/No-go gates

| Gate | After | Pass criterion | Fail action |
|---|---|---|---|
| **Gate 1** | V5-FedCB-0 (3 seeds) | 3-seed mean PAPE within ±1.5 pp of v02 §B.3's 44.18 (i.e. ~ [42.7, 45.7]). | Code regression in CMO inference, codebook fit, or v02 backbone load. Halt; debug. **V5-FedCB-1 forbidden until cleared.** |
| **Gate 2** | V5-FedCB-1 (3 seeds) | 3-seed mean PAPE ≤ 52 %. | Hierarchical KMeans capacity issue *or* CMO-only is too weak under FedAvg backbone drift. Proceed to Gate 3 (K_local sweep) and V5-FedCB-3 (α sweep) instead of giving up. |
| **Gate 3** | V5-FedCB-{2a, 2b} + V5-FedCB-3 (3 seeds each) | At least one (`K_local`, `α`) ∈ {2, 4, 8} × {0.5, 1.0, 1.5, 2.0} clears the 52 % threshold. | Method unfit. KIIE talk falls back to centralised codebook with paper.md §5.4 honest-disclosure framing — no upgrade to "fully-FL framework". |

The 52 % threshold = ≈ 8 % relative improvement over the standard
FedAvg-NBEATSxAux raw baseline (56–57 % in `papers/pfl_unified/paper.md`
Table 1). It was relaxed from an earlier 49 % proposal because
V5-FedCB-1 is layered on FedAvg backbone (not centralised T2) — the
published CMO anchor 44.18 was measured on centralised T2, and a
~6 pp FedAvg-vs-centralised drift is already documented at the W5
correction strength (paper.md §4.8). 52 % keeps a meaningful "fully-FL
recipe outperforms raw FL" claim without requiring V5-FedCB-1 to match
the centralised CMO number, which would be a different (unattainable)
contract. Gate 2 and Gate 3 share the threshold; Gate 2 checks the
default cell, Gate 3 lets us salvage via hyperparameter search if the
default misses.

## Build order

| Step | Module | Purpose | Verify |
|---|---|---|---|
| **1** | `src/fl/codebook_fl.py` (new) | Three functional helpers, matching `src/fl/base.py` style (no client class): `local_codebook_step(model, client, K_local, seed)` → dict (centroids, counts, h_g, y_hat_z, y_true_z), `merge_local_codebooks(local_packets, M_global, seed)` → dict (codebook, stage2_inertia, n_empty, util, perplexity), `federated_residual_offsets(local_packets, codebook)` → np.ndarray (M, H). sklearn directly for both KMeans stages — `VectorQuantizerKMeans.fit()` does not expose `sample_weight`, which Stage 2 needs. | pytest: 80 synthetic clients × `K_local=4` × `D=64`; verify Stage 2 deterministic at fixed seed; verify offsets match a centralised reference within fp32 tolerance when `K_local = M`. |
| **2** | `experiments/v05_fedcb_codebook/01_fedcb_codebook.py` (new) | Per-seed driver, **federated mode only**. argparse: `--seed S`, `--K_local K` (default 4), `--alpha A` (default 1.0), `--M 32`. Loads 09_fix_rerun FedAvg-NBEATSxAux backbone, invokes step-1 helpers, applies CMO-only with `α=--alpha`, saves. Centralised mode is *not* implemented in the driver — V5-FedCB-0 is a JSON look-up (see step 3). | smoke: `--seed 42 --K_local 4 --alpha 1.0` end-to-end. |
| **3** | `experiments/v05_fedcb_codebook/02_aggregate.py` (new) | Reads (a) `outputs/v05_fedcb_codebook/seed{42,123,7}/{cell}/result.json` for V5-FedCB-1/2/3, and (b) `outputs/v02_fl_8020_ratio/seed{42,123,7}/W_component_results.json` `→ per_operating_point["PAPE-aggressive"]["cells"]["V0"]` for V5-FedCB-0. Computes 3-seed mean ± std for `fl_only` and `with_codebook_cmo` PAPE/HR@1/HR@2/MAE. Output schema matches `outputs/v04_full_baseline_comparison/multiseed_summary.json` so the existing paper aggregator can ingest it. Includes per-cell Gate pass/fail flag. | one shot. |
| **4** | `experiments/v05_fedcb_codebook/03_communication.py` (new) | Seed-independent. `local_upload_bytes = 80 × K_local × (64 + 1) × 4`, `broadcast_bytes = M × 64 × 4`, `residual_aggregation_bytes = 80 × M × (24 + 1) × 4` for `K_local ∈ {2, 4, 8}`. Reports as paper.md §4.7-compatible CSV/JSON. | one shot. |
| **5** | 3-seed sweep | V5-FedCB-0 = data-only (no run). V5-FedCB-{1, 2a, 2b} = 3 cells × 3 seeds = 9 runs. (+ V5-FedCB-3 opt = 4 α × 3 seeds = 12 runs.) Wall-clock per run ≈ 1–2 min on a 5070 Ti (Phase A reused, only Phase B' + Phase C). Total ≈ 15–30 min serial. | `multiseed_summary.json`, `communication_summary.json`. |
| **6** | `papers/pfl_unified/paper.md` patch | If Gate 3 passes: insert **only V5-FedCB-1** (FedAvg + federated codebook + CMO) into Table 1 main rows + a paragraph in §5.4 swapping "centralised codebook" disclosure for "fully-FL framework via hierarchical KMeans". **V5-FedCB-0 must NOT appear in Table 1**; it is a code-validation anchor on a different backbone (centralised T2) and presenting it side-by-side would invite the wrong reading "centralised codebook beats federated codebook" when the actual delta is the backbone change. Place V5-FedCB-0 in an appendix (e.g. C.5 reference numbers) explicitly labelled "code-correctness reproduction of v02 §B.3 CMO; not a baseline". K_local sweep (V5-FedCB-2) and α sweep (V5-FedCB-3) likewise belong in §4 / appendix as ablations, not Table 1 main rows. If Gate 3 fails: leave §5.4 as-is, add an appendix entry citing v05 as attempted-but-unfit. | reviewer pass. |

## Outputs

```
outputs/v05_fedcb_codebook/
├── seed{42,123,7}/
│   ├── fedcb_centralised/{result.json, codebook.npz}     # V5-FedCB-0
│   ├── fedcb_K2/{result.json, codebook.npz}              # V5-FedCB-2a
│   ├── fedcb_K4/{result.json, codebook.npz}              # V5-FedCB-1
│   ├── fedcb_K8/{result.json, codebook.npz}              # V5-FedCB-2b
│   └── fedcb_K4_alpha{0.5,1.0,1.5,2.0}/result.json       # V5-FedCB-3 (opt)
├── communication_summary.json                             # V5-FedCB-4
└── multiseed_summary.json
```

`result.json` schema (compatible superset of
`09_fix_rerun/02_fedavg_nbeatsx_aux.py`'s output, but with the W5 op-point
blocks collapsed to a single CMO block):

```json
{
  "algorithm": "fedcb_K4",
  "mode": "federated",
  "seed": 42,
  "backbone_source": "outputs/v04_full_baseline_comparison/09_fix_rerun/seed42/fedavg_nbeatsx_aux/final_state_dict.pt",
  "K_local": 4,
  "M_global": 32,
  "alpha": 1.0,
  "config": { "...Phase A hp + Phase B' hp..." },
  "fl_only":          { "pape": ..., "hr@1": ..., "hr@2": ..., "mae": ..., "n_cold_windows": ..., "n_cold_apts": ... },
  "with_codebook_cmo":{ "alpha": 1.0, "metrics": { "pape": ..., "hr@1": ..., "hr@2": ..., "mae": ... } },
  "vq_diagnostics": {
    "utilization": ..., "perplexity": ..., "k_min": ..., "k_max": ...,
    "n_empty_clusters": ..., "stage1_mean_inertia": ..., "stage2_inertia": ...
  },
  "communication_bytes": {
    "local_upload_per_client": ..., "broadcast": ...,
    "residual_per_client":     ..., "total_round": ...
  },
  "n_train_clients": 80,
  "n_train_windows_total": ...,
  "elapsed_seconds": { "phase_b": ..., "phase_c": ..., "total": ... }
}
```

For V5-FedCB-0 the same schema is used with `mode = "centralised"`,
`K_local = null`, `backbone_source` = the v02 T2 path, and the
`vq_diagnostics.stage1_*` / `stage2_inertia` keys absent (single-stage
KMeans).

## Dependencies

- `outputs/v04_full_baseline_comparison/09_fix_rerun/seed{S}/fedavg_nbeatsx_aux/final_state_dict.pt`
  for V5-FedCB-1, 2, 3.
- `outputs/v02_fl_8020_ratio/seed{S}/T2/best.pt` for V5-FedCB-0
  (centralised T2 backbone, same artefact `03_fit_codebook.py` already
  consumes).
- `outputs/v02_fl_8020_ratio/splits/v02_8020_seed{S}.yaml` for the
  80-train / 20-cold partition.
- `src/eval/cold_helpers.py` (`gather_cold`, `route_R1`,
  `metrics_z_to_kw` — `gauss_template` and `OPERATING_POINTS` are *not*
  used in v05).
- `src/fl/base.py` (`build_clients`, `client_loader`, `DEVICE`,
  `apply_state_dict`, `clone_state_dict`).
- `src/models/nbeatsx_aux.py` (`NBEATSxAux(latent_source='h_generic')`).
- `src/models/vq_kmeans.py` (`VectorQuantizerKMeans` — final wrapper
  only; KMeans itself uses sklearn directly because Stage 2 needs
  `sample_weight`).

## Open questions

A. **Stage 2 weighting.** `sample_weight = n_{i,c}` is the natural choice
   (matches centralised KMeans on the original windows in expectation).
   Alternatives if Gate 2 misses: (i) uniform (each local centroid
   counted once), (ii) `sqrt(n_{i,c})` (heavy-client damping). Default
   = sample-count-weighted; revisit only if Gate 3 fails.

B. **Tiny clients.** `K_local_i = min(K_local, N_i)` fallback for any
   client with fewer windows than `K_local`. UMass `N_i` at stride=24
   is roughly 270 windows per apt, so this is unlikely to trigger at
   `K_local ≤ 8`, but step 1 must handle it (sklearn `KMeans` raises if
   `n_clusters > n_samples`).

C. **Empty Stage 2 clusters.** Skewed `sample_weight` can leave some of
   the 32 global clusters empty after fit. Empty offsets stay at `0`
   (matches v01 / `vq_kmeans.fit` convention). `n_empty_clusters`
   reported in `vq_diagnostics`.

D. **α default.** V5-FedCB-1 fixes `α = 1.0`. V5-FedCB-3 sweeps
   `α ∈ {0.5, 1.0, 1.5, 2.0}` at `K_local=4`. If 1.5 (the v02 paper
   value) wins on the FedAvg backbone, the v05 default flips to 1.5
   *post-hoc* — but **only after Gate 3 closes**, since cold-side α
   tuning is what v01 §5.4.1 explicitly warned against. Document the
   sweep result, freeze the chosen α, then re-report V5-FedCB-1 / 2 at
   the frozen α if it differs from 1.0.

E. **V5-FedCB-0 backbone and routing choice.** V5-FedCB-0 reuses v02
   §B.3 published numbers (centralised T2 backbone + R0 KEY-routing) —
   *neither* the FedAvg backbone *nor* the R1 routing of V5-FedCB-1+.
   This is intentional: V5-FedCB-0 is a paper-anchor lookup, not a
   federation comparison. The 44.18 anchor is the only published CMO
   number; without a same-backbone-same-routing equivalent for FedAvg+R1,
   mixing them into Gate 1 would conflate three variables (backbone,
   routing, codebook construction). The honest comparison Δ is therefore
   "V5-FedCB-1 vs absolute 52 % bar" (Gates 2/3), not "V5-FedCB-1 vs
   V5-FedCB-0".

## Conventions (carried over from CLAUDE.md and v04-01)

- **Per-seed argparse.** All v05 drivers take `--seed S`; the
  `{42, 123, 7}` sweep is dispatched externally (memory:
  `feedback_argparse_per_seed`).
- **No MLflow.** This repo logs via `result.json` + `print`. The original
  ADR's "MLflow 의무" line is dropped to match repo convention.
- **Output namespacing.** `outputs/v05_fedcb_codebook/seed{S}/{cell}/result.json`.
- **bf16 + batch=512** for backbone forward (eval only). Phase B'-1
  KMeans uses fp32 numpy (sklearn).
- **Method frozen.** No re-tuning of the encoder, aux head, or peak
  descriptor. Cold-side α is allowed to sweep *once* (V5-FedCB-3) and
  then frozen — see open question D.

## What is NOT in scope

- Iterative federated KMeans / DP-KMeans / SecAgg (see §Non-goals).
- Re-training Phase A.
- New W-mechanisms / Hybrid correction at inference / new aux head.
- A second dataset.
- R0 KEY-pool routing (dropped, not federated).
