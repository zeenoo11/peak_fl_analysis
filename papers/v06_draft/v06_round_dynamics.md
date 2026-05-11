# v06 Round-Level FL Training Dynamics: Federated Training, Peak-Aux Loss, and Post-Hoc Codebook Stacking

## §1 Motivation

v01–v05 of this project all evaluate forecasting models on a **cold-client zero-shot** held-out
partition: a fraction of UMass apartments (20 % in v02–v04, 50 % in v01) is excluded from training
and only entered at inference time. This is a *cold personalisation* protocol — it measures how
well a federated or centralised model generalises to apartments it has never seen, not how
*federation itself* affects training. v06 abandons the cold partition entirely and asks two
distinct questions:

1. **Phase 1 — round-level FL training dynamics.** Every UMass apartment participates in training;
   each client is evaluated on its own internal val/test windows; metrics are logged at every
   FL communication round. This is a clean "FedAvg vs FedSGD"-style protocol from McMahan 2017.
2. **Phase 2 — post-hoc codebook stacking on the trained backbones.** v01–v05's signature
   mechanism — a Peak-VQ codebook + cluster-mean residual offset — is layered on top of each
   v06 backbone and evaluated on the per-client test windows. The codebook is fitted under
   *federated* (V5 FedCB) construction for FL backbones and pooled KMeans for the centralised
   reference, so a single Δ PAPE answers "does this method-agnostic correction module also
   help when the cold partition is gone?"

Together these two phases produce three findings:

- **Phase 1 negative result.** The peak-aux head loss term (`λ=0.3`) carried over from v01–v05
  *hurts* PAPE by ~3 points in the round-level FL setting — even though it helps in the
  cold-zero-shot setting. The MAE-only ablation closes the gap to centralised by ~2 PAPE.
- **Phase 1 algorithm equivalence.** FedAvg / FedProx / FedRep / Ditto / FedProto sit within
  0.5 PAPE of each other after 20 rounds. The cost-efficient choice (FedRep, −18 % comm) is
  Pareto-dominant.
- **Phase 2 codebook lift.** Across all 6 backbones, post-hoc codebook stacking improves test
  PAPE by **−4.5 to −5.9** points. The federated codebook produces a quality-equivalent codebook
  to pooled KMeans (utilization 1.0, perplexity 25–27, no empty clusters). Federated codebook
  stacking *closes most of the +2 PAPE FL deficit* (residual gap +0.98 vs +2.06 before
  correction), turning the codebook into an additive lift on top of FL training.

## §2 Method

### Phase 1 backbone (frozen across Phase 2)

`NBEATSxAux(latent_source='h_generic')` — `MinimalNBEATSx` 3-stack body (trend / seasonal /
generic) + `PeakAuxHead`. Combined loss:

```
L = MAE(ŷ, y) + λ_aux · peak_aux_loss(â, ĥ, y; hr_weight=0.1)
```

Default `λ_aux = 0.3` (CLAUDE.md invariant carried over from v01). The MAE-only ablation
sets `λ_aux = 0` and is namespaced as `*-MAEonly` so the default and the ablation never
overwrite each other.

### Phase 1 per-client splits (`src/dataloader/per_client_split.py`)

Every UMass 2016 apartment that passes `filter_valid_apartments(min_hours=7000)` (114 of them)
is included. For each apartment, the hourly series is split chronologically:

- `train` 70 % (sliding window stride = 24, non-overlapping)
- `val`   10 % (with INPUT_SIZE look-back into the train segment, stride 24)
- `test`  20 % (with INPUT_SIZE look-back into the val segment, stride 24)

Per-apartment z-norm is fit on the train portion only (CLAUDE.md invariant). The cache file
`outputs/v06_round_dynamics/seed{S}/per_client_split.pkl` is reused by every Phase-1 and
Phase-2 driver.

### Phase 1 cells

`V6-Dyn-A_centralised` — pooled SGD upper bound. All 114 apartments' train windows are
concatenated into a single DataLoader; a single `NBEATSxAux` is trained for 40 epochs (Adam
1e-3, weight decay 1e-5, batch 512, bf16 on CUDA).

`V6-Dyn-B-{FedAvg, FedProx, FedRep, Ditto, FedProto}` — five FL algorithms, all on the same
NBEATSxAux backbone, all logged round-by-round through a shared `RoundLogger`. Common
hyperparameters:

| Hyperparameter | Value |
|---|---|
| Rounds         | 20 |
| Local epochs   | 2  |
| Optimiser      | Adam (lr=1e-3, weight_decay=1e-5) |
| Batch          | 512 |
| Participation  | full (C=1.0; 114 of 114 clients) |
| AMP            | bf16 on CUDA |

Algorithm-specific extras (paper-default values, no per-cell tuning):

| Algorithm | Extra |
|---|---|
| FedProx  | μ = 0.01 (proximal weight) |
| FedRep   | head_epochs = 1 (1 head + 1 encoder out of 2 local epochs) |
| Ditto    | λ = 0.1 (personal-pull weight) |
| FedProto | K = 32, λ_proto = 0.1 |

### Phase 1 round logger contract (`src/fl/round_logger.py`)

After every round (or epoch for centralised), the logger writes one JSONL row containing:

- `val` block — per-client val forward, kW metrics aggregated as across-client mean / std (ddof=1).
- `test` block — same shape as `val`, on the per-client test windows. (This is the
  "Option A" round-level test trajectory adopted in v06; it adds ~2× val-forward cost per
  round but matches the McMahan 2017 / FedProx 2020 paper convention of plotting test
  metrics vs round.)
- `train` block — `loss_mean_last_epoch`, `n_steps_round`.
- `comm` block — algorithm-specific upload + broadcast bytes (cumulative).
- `drift_l2` — mean ‖θ_i − θ_global^pre‖₂ over float-tensor parameters across the round's
  client end-states.
- `wall_seconds_round`.

A terminal row (`round = -1`) is appended after training for the final val and test
evaluations, using a fresh `init_backbone_aux(seed)` reloaded with `strict=True`.

### Phase 2 codebook stacking (`experiments/v06_round_dynamics/08_codebook_stacking.py`)

For each Phase-1 cell × seed, the frozen backbone (`final_state_dict.pt`, loaded
`strict=True`) is forwarded over all clients' train windows to extract `h_generic` ∈ ℝ⁶⁴ along
with `(ŷ_z, y_z)`. The codebook protocol depends on whether the backbone is centralised or
federated:

- **V6-Dyn-A_centralised** — pooled KMeans++. All clients' `h_generic` are concatenated and
  a single KMeans++(M=32, n_init=10) is fitted; cluster-mean residual offsets are computed
  on the same pooled stream.
- **V6-Dyn-B-* (5 cells)** — 2-stage hierarchical *federated* KMeans (`src/fl/codebook_fl.py`,
  carried over from v05):
  - Stage 1 — each client fits a local KMeans++(K_local=2) on its own `h_g`; only
    `(centroids, counts)` upload.
  - Stage 2 — server runs a weighted KMeans++(M=32) on the stacked centroids, broadcasts
    a single `(M, 64)` codebook.
  - Stage 3 — each client routes its train windows against the codebook and uploads
    per-cluster `(residual_partial_sum, count)`; the server divides cluster-wise to obtain
    `offsets ∈ ℝ^{M×24}`.

Raw `h_g` never leaves the client in the FL path. Both paths produce a `(32, 64)` codebook and
`(32, 24)` offsets with identical schema.

### Phase 2 correction (CMO-only)

```
ŷ_corr = ŷ_base + α_v0 · offsets[argmin_c ‖h_g_test − codebook[c]‖₂]
```

with `α_v0 = 1.0` (operating point carried over; not re-tuned on test). The Gaussian template
term from v01's W5 hybrid is dropped (`α_w1 = 0`) — Phase 2 isolates the codebook's
contribution.

### Phase 2 evaluation

Each apartment's `test_x` (20 % per-client test) is forwarded once through the frozen backbone
to obtain `(h_g_cold, ŷ_base_z)`. The codebook produces `c_idx`, the offset is added,
predictions are denormalised to kW with the apartment's `(mean, std)`, and per-apt
PAPE / HR@1 / HR@2 / MAE / MSE(kW²) are computed and aggregated as across-apt mean / std (ddof=1).

## §3 Experimental Setup

- Data: UMass Smart* 2016 hourly, 114 apartments (after `filter_valid_apartments(min_hours=7000)`).
- Per-client splits: `outputs/v06_round_dynamics/seed{S}/per_client_split.pkl`.
- Seeds: `{42, 123, 7}`; reported numbers are mean ± std (ddof=1) across seeds.
- Per-seed argparse: every driver takes `--seed S` per invocation; multi-seed is the
  launcher's responsibility (memory: feedback_argparse_per_seed).
- Output namespace: `outputs/v06_round_dynamics/seed{S}/{cell}/...`.

## §4 Phase 1 — round-level FL training dynamics

### §4.1 Default cell (λ_aux = 0.3)

| Cell | val.PAPE | **test.PAPE** | HR@1 (test) | drift L2 | Upload (MB) | wall (s) |
|---|---|---|---|---|---|---|
| V6-Dyn-A centralised | 66.33 ± 0.80 | **49.43 ± 0.36** | 20.81 ± 0.03 | 0 | 0 | 35 |
| FedAvg            | 81.62 ± 3.00 | 51.36 ± 0.61 | 13.46 ± 0.16 | 2.42 | 641  | 725 |
| FedProx (μ=0.01)  | 81.73 ± 2.35 | 51.40 ± 0.63 | 13.78 ± 0.18 | **1.71** | 641 | 2384 |
| FedRep (head_ep=1)| 78.24 ± 1.79 | 51.36 ± 0.68 | 13.78 ± 0.62 | 2.20 | **527** | 897  |
| Ditto (λ=0.1)     | 84.49 ± 2.42 | 51.79 ± 0.47 | 13.84 ± 0.35 | 2.42 | 641  | 4410 |
| FedProto (K=32)   | 80.93 ± 2.95 | 51.50 ± 0.54 | 13.46 ± 0.34 | 2.43 | 660  | 1574 |

**Centralised vs FL.** Centralised SGD attains test PAPE = 49.43 %; the best FL test PAPE is
51.36 % (FedAvg). The +2 PAPE deficit holds across all five FL algorithms with std ~0.6
indicating a real gap, not noise.

**Algorithm equivalence.** The five FL test PAPEs span 51.36 to 51.79 (range 0.43 PAPE)
within their ~0.6 PAPE std bands. Algorithm choice is *not* a discriminator at 20 rounds with
2 local epochs. FedProx halves client drift (1.71 vs ~2.42) but does *not* convert that into
PAPE improvement, consistent with the FedProx 2020 finding that drift control ≠ accuracy gain
on convex-ish workloads.

**Cost efficiency.** FedRep uploads 527 MB total (head not broadcast), an 18 % saving versus
FedAvg's 641 MB at equivalent test PAPE — the Pareto-dominant FL choice. FedProto pays for
prototype broadcast (+8 KB / round) but the resulting accuracy difference is statistically
zero.

### §4.2 MAE-only ablation (λ_aux = 0)

We re-ran all six cells with the peak-aux loss zero'd (`--aux_lambda 0`) to isolate the
contribution of the peak-aux head in the round-level FL regime.

| Cell | val.PAPE | **test.PAPE** | HR@1 (test) | drift L2 | wall (s) |
|---|---|---|---|---|---|
| V6-Dyn-A-MAEonly       | 64.94 ± 1.76 | 48.91 ± 0.70 | 20.97 ± 0.54 | 0   | 61   |
| FedAvg-MAEonly         | 80.94 ± 1.26 | 48.42 ± 0.37 | 15.68 ± 0.37 | 2.56 | 1086 |
| FedProx-MAEonly        | 78.41 ± 1.05 | 48.51 ± 0.03 | 15.86 ± 0.42 | **1.67** | 2361 |
| FedRep-MAEonly         | 84.85 ± 0.87 | 49.08 ± 0.50 | 15.27 ± 0.37 | 2.45 | 878  |
| **Ditto-MAEonly**      | 78.79 ± 1.55 | **48.28 ± 0.32** | **16.09 ± 0.69** | 2.56 | 3676 |
| FedProto-MAEonly       | 81.03 ± 1.05 | 48.49 ± 0.31 | 15.81 ± 0.71 | 2.57 | 751  |

**Negative result.** Switching off the peak-aux head improves test PAPE by 0.5 PAPE on
centralised and by 2.3–3.5 PAPE on the FL cells. HR@1 simultaneously improves by 0.2–2.4
points. The peak-aux loss term, which was a positive contributor in the v01 cold-zero-shot
protocol (T2 vs T0 in v01 §6), is a *negative* contributor when training itself is federated.

A plausible mechanism: the auxiliary head's hour-classification CE term operates on a 24-class
label whose distribution is *highly heterogeneous across apartments* (different households peak
at different times). FedAvg averages clients' aux gradients, diluting the signal. With λ_aux = 0
that heterogeneous label noise is removed and FedAvg's MAE-only signal averages cleanly.

**FL deficit closes under MAE-only.** The +2 PAPE gap in §4.1 collapses to between
−0.6 and +0.2 in §4.2: Ditto-MAEonly (48.28) and FedAvg-MAEonly (48.42) actually beat
centralised-MAEonly (48.91), suggesting that the "FL is fundamentally worse than centralised"
narrative is partially an artifact of the loss term, not the algorithm.

### §4.3 Per-round trajectory (F1 family)

`outputs/v06_round_dynamics/figures/F1_round_vs_val_pape.png` (val PAPE vs round) and the
sibling F1b (test PAPE vs round), F1c (train loss vs round), F4/F5 (MAEonly variants) show
the typical FedAvg trajectory: most of the descent in rounds 1–6, plateau by round 12,
no benefit from rounds 13–20 except a small late descent for FedRep. The trajectories
support §4.1's algorithm-equivalence finding visually.

`F2_bytes_vs_val_pape.png` confirms FedRep's Pareto dominance — its trajectory sits
left-of-and-below FedAvg's at every round.

`F3_drift_vs_round.png` shows FedProx flat at ~1.7 from round 5 onwards while the other
four FL algorithms drift up to 2.4–2.6, the cleanest visual signature of FedProx's effect.

## §5 Phase 2 — Post-hoc codebook stacking

### §5.1 Codebook lift across all six backbones (λ_aux = 0.3)

| Cell | test.PAPE BEFORE | **test.PAPE AFTER** | **ΔPAPE** | ΔHR@1 | ΔHR@2 | ΔMAE | ΔMSE(kW²) |
|---|---|---|---|---|---|---|---|
| V6-Dyn-A centralised | 49.43 ± 0.35 | **44.92 ± 0.14** | **−4.51 ± 0.21** | +0.78 | +0.97 | +0.0095 | −0.0183 |
| V6-Dyn-B-FedAvg     | 51.36 ± 0.63 | 45.92 ± 0.51 | −5.44 ± 0.15 | +0.43 | +0.72 | +0.0065 | −0.0202 |
| V6-Dyn-B-FedProx    | 51.42 ± 0.64 | 46.00 ± 0.45 | −5.42 ± 0.20 | +0.40 | +0.86 | +0.0067 | −0.0206 |
| V6-Dyn-B-FedRep     | 51.37 ± 0.66 | **45.77 ± 0.22** | −5.60 ± 0.45 | +0.41 | +0.74 | +0.0064 | −0.0209 |
| **V6-Dyn-B-Ditto**  | 51.80 ± 0.44 | 45.92 ± 0.26 | **−5.88 ± 0.27** | +0.17 | +0.47 | +0.0057 | **−0.0253** |
| V6-Dyn-B-FedProto   | 51.51 ± 0.56 | 45.89 ± 0.29 | −5.61 ± 0.33 | +0.77 | **+1.23** | +0.0062 | −0.0208 |

(F6 figure: `outputs/v06_round_dynamics/figures/F6_codebook_lift.png`. BEFORE = faded
bar, AFTER = saturated bar, Δ label centred above the pair.)

**Lift universality.** Every cell — centralised and all five FL — gains 4.5–5.9 PAPE points
from post-hoc codebook stacking. Algorithm choice does not change the conclusion; the lift
range is narrow (range 1.4 PAPE between cells, std 0.15–0.45 within each cell).

**FL deficit closes.** Phase 1's +2.06 PAPE gap (centralised 49.43 vs FL avg 51.49)
shrinks to +0.98 PAPE (centralised 44.92 vs FL avg 45.90). FL receives a *larger* codebook
lift than centralised, suggesting the codebook compensates for FL training's PAPE shortfall
rather than simply adding the same lift everywhere.

**Trade-off.** All cells show a small positive ΔMAE (+0.006 to +0.010 kW), i.e. horizon-mean
absolute error gets slightly worse. This is the known v01 W5 trade-off: cluster-mean offsets
move predictions toward cluster-typical peak shapes, which improves PAPE / HR / MSE but pulls
non-peak hours away from their MAE-optimal values. The ratio (PAPE −5 vs MAE +0.007) makes
the trade favourable for any peak-prioritised application.

### §5.2 Federated codebook quality vs centralised codebook quality

| Codebook diagnostic | centralised | FL avg (5 cells) |
|---|---|---|
| utilization        | 1.000 ± 0.000 | 1.000 ± 0.000 |
| perplexity (M=32)  | 26.74 ± 0.95  | 26.18 ± 0.40  |
| n_empty_clusters   | 0             | 0             |
| k_max              | 2991 ± 406    | 2026 ± 180    |
| k_min              | 109 ± 45      | 46 ± 19       |

The federated 2-stage hierarchical KMeans produces a codebook of *equal effective quality*
to pooled KMeans on every diagnostic. The pooled codebook is slightly more skewed (larger
k_max, larger k_min) while the federated one distributes mass more uniformly across
clusters — but both achieve full utilization and 81 % of ideal perplexity. The federation
contract (raw `h_g` does not leave the client) is paid in zero accuracy.

### §5.3 Codebook lift on the MAE-only backbone

A natural follow-up question to §5.1 is whether the codebook still helps when
the underlying backbone is trained without the peak-aux head (`λ_aux = 0`). If
codebook lift requires the peak-aware `h_generic` structure that the aux head
induces, then a backbone trained with MAE only should produce a smaller (or
zero) lift. We re-ran the Phase-2 pipeline on every `*-MAEonly` cell from §4.2
and obtained:

| Cell | test.PAPE BEFORE | test.PAPE AFTER | ΔPAPE | (vs default ΔPAPE) |
|---|---|---|---|---|
| centralised-MAEonly | 48.90 ± 0.68 | **44.41 ± 0.29** | **−4.49 ± 0.56** | (default −4.51) |
| FedAvg-MAEonly      | 48.43 ± 0.38 | 44.59 ± 0.34 | −3.84 ± 0.06 | (default −5.44) |
| FedProx-MAEonly     | 48.50 ± 0.02 | 44.84 ± 0.17 | −3.66 ± 0.17 | (default −5.42) |
| FedRep-MAEonly      | 49.07 ± 0.50 | 45.49 ± 0.68 | −3.58 ± 0.90 | (default −5.60) |
| **Ditto-MAEonly**   | 48.29 ± 0.32 | **44.20 ± 0.27** | −4.09 ± 0.08 | (default −5.88) |
| FedProto-MAEonly    | 48.47 ± 0.30 | 44.54 ± 0.26 | −3.93 ± 0.12 | (default −5.61) |

**Lift survives but shrinks.** Every MAE-only cell still gains 3.6–4.5 PAPE
points from codebook stacking — the codebook is *not* a free rider on the
peak-aux head. The lift on the centralised cell is unchanged (−4.49 vs −4.51),
suggesting that pooled-data training already induces enough peak-relevant
structure in `h_generic`. The FL cells lose 1.2–2.0 PAPE of lift versus their
default counterparts, indicating that on the *federated* training path the
peak-aux head is the main contributor of cluster-friendly latent geometry.

**Counter-intuitive consequence — MAE-only + codebook beats default + codebook.**
Comparing AFTER values across §5.1 and §5.3:

| Cell | AFTER default (λ=0.3) | AFTER MAEonly (λ=0) | MAEonly is better by |
|---|---|---|---|
| centralised | 44.92 | **44.41** | −0.51 |
| FedAvg      | 45.92 | **44.59** | **−1.33** |
| FedProx     | 46.00 | **44.84** | −1.16 |
| FedRep      | 45.77 | 45.49     | −0.28 |
| **Ditto**   | 45.92 | **44.20** | **−1.72** |
| FedProto    | 45.89 | **44.54** | −1.35 |

The §4.2 negative result on the peak-aux head is **not erased by Phase-2
codebook stacking**. Across every cell, dropping the peak-aux head and then
applying the codebook produces a *strictly better* test PAPE than keeping the
peak-aux head and applying the same codebook. The peak-aux head is genuinely
counterproductive in the round-level FL training regime — both directly (§4.2)
and after post-hoc correction (this section). The recommendation for v06's
operating recipe is therefore **`λ_aux = 0` + federated codebook stacking**, with
test PAPE of 44.2–44.8 % (FL) and 44.4 % (centralised).

### §5.4 Codebook correction strength α_v0 (Pareto curve)

§5.1 fixed `α_v0 = 1.0` as the carry-over operating point from v01. The v01
draft also documented a `PAPE-aggressive` setting at `α_v0 = 1.5`. v06 adds two
more grid points (0.5 and 2.0) and reports the full PAPE / MAE Pareto.

| α_v0 | centralised PAPE / ΔMAE | FL avg PAPE / ΔMAE |
|---|---|---|
| 0.5  | 47.18 / +0.0018 | 48.66 / **−0.0004** |
| 1.0  | 44.92 / +0.0095 | 45.90 / +0.0064 |
| 1.5  | 42.79 / +0.023  | 43.37 / +0.019 |
| 2.0  | **40.78** / +0.040 | **41.21** / +0.037 |

(F7 figure: `outputs/v06_round_dynamics/figures/F7_alpha_pareto.png` — ΔMAE
on x-axis, ΔPAPE on y-axis, 6 curves × 4 α-points each.)

**Three operating points emerge.**

- **α_v0 = 0.5 — MAE-zero-cost.** The FL cells' ΔMAE is *negative* (FL
  improves MAE marginally) while ΔPAPE remains −2.7 to −3.0. This is a Pareto
  point not previously documented in v01.
- **α_v0 = 1.0 — HR-preserving** (v01 carry-over). ΔPAPE −4.5 to −5.9, ΔMAE
  +0.006 to +0.010 kW.
- **α_v0 = 1.5 — PAPE-aggressive** (v01 carry-over). ΔPAPE pushes to ~−6.6
  (centralised 42.79), ΔMAE ~+0.020.
- **α_v0 = 2.0 — PAPE-extreme.** ΔPAPE up to −8.65 (centralised 40.78,
  Phase 1 was 49.43 — a 17.5 % relative reduction), ΔMAE +0.04.

**The trade-off is monotonic.** No α value dominates another; the choice
follows the application's relative weighting of peak amplitude vs horizon-mean
absolute error. The α=0.5 point is the new addition: it adds a "MAE-neutral"
operating choice that v01 did not document. For deployments that cannot
tolerate any MAE regression, α=0.5 still delivers ~3 PAPE of lift, whereas the
v06 baseline (α=1.0) trades +0.006-0.010 ΔMAE for double that PAPE lift.

### §5.5 Federated codebook K_local sweep

§5.2 inherited K_local = 2 from v05 FedCB without re-validation. We swept
K_local ∈ {1, 2, 4, 8} on the five FL cells (the centralised path uses pooled
KMeans and is K_local-invariant) and obtained:

| Cell | K=1 | **K=2** (baseline) | K=4 | K=8 |
|---|---|---|---|---|
| FedAvg   | −4.36 | **−5.44** | −5.58 | −5.71 |
| FedProx  | −4.41 | **−5.42** | −5.53 | −5.76 |
| FedRep   | −4.42 | **−5.60** | −5.69 | −5.79 |
| Ditto    | −5.01 | **−5.88** | −6.00 | **−6.19** |
| FedProto | −4.51 | **−5.61** | −5.85 | −5.90 |

(F8 figure: `outputs/v06_round_dynamics/figures/F8_klocal_sweep.png` — 5 lines
× 4 points, log₂ x-axis, baseline K=2 marked.)

**Diminishing returns past K=2.** The K=1 → K=2 step yields ~+1.0 PAPE of lift
on every FL cell. The subsequent K=2 → K=4 step adds only ~+0.1 PAPE, and
K=4 → K=8 adds another ~+0.1. K=8 is the maximum gain (−6.19 on Ditto) but
the marginal lift past K=2 is statistically marginal versus seed std
(~0.2-0.5).

**v05's K_local = 2 choice re-validated.** v05 FedCB arrived at K=2 by an
elbow argument on `K_local × (codebook_inertia, lift)` under the cold-zero-shot
protocol. v06 reproduces the same conclusion under the round-level FL protocol,
confirming that **K=2 is robust to evaluation protocol**. K=1 (one centroid
per client = client mean h_g) loses ~1 PAPE of lift, indicating that some
intra-client peak diversity must be preserved. K=8 quadruples upload cost
(centroid + count payload) for sub-noise gains — Pareto-dominated.

**Codebook diagnostics are flat in K.** Utilization stays at 1.000 ± 0.000 and
perplexity at 25.6-27.8 across all four K values, so the lift difference is
not driven by Stage-2 quality but by the granularity of the Stage-1 input
centroids fed to Stage-2.

### §5.6 Per-algorithm Stage-1 inertia (FL only)

The Stage-1 mean inertia across clients varies by algorithm:

| Algorithm | Stage-1 mean inertia | Interpretation |
|---|---|---|
| FedProto | **194** (lowest) | prototype regulariser pulls h_g toward cluster centres |
| FedAvg   | 339              | baseline |
| FedProx  | 333              | μ-proximal bonus → slightly tighter clusters |
| Ditto    | 419              | personal-pull spreads h_g across clients |
| FedRep   | **1003** (highest)| encoder shared but head local → h_g most dispersed |

This is a side-channel result: FedProto's prototype loss creates a codebook-friendly latent
geometry as a by-product. However it does not translate into a Stage-2 or ΔPAPE advantage
(all five algorithms land within 0.5 ΔPAPE of each other), so the implication is descriptive,
not actionable.

## §6 Discussion

### Negative result on the peak-aux head

v01–v05 carried `λ_aux = 0.3` as an invariant, validated under the cold-zero-shot protocol.
v06's MAE-only ablation shows that this value *hurts* PAPE in the round-level FL setting.
Two possible causes:

1. **Heterogeneous label distribution.** The auxiliary head's CE on the 24-class peak-hour
   label is sensitive to per-client peak-hour distributions; FedAvg averaging dilutes the
   client-specific signal.
2. **Loss-weighting mis-tune.** `λ_aux = 0.3` was tuned on v01's centralised training.
   Federation may need `λ_aux` smaller (or a per-client adaptive value).

Path forward: a `λ_aux ∈ {0, 0.05, 0.1, 0.2, 0.3}` sweep would localise the optimum. v06
does not run this sweep — the negative finding is reported as-is for the paper; future work
can refine.

### FL gap closing via codebook

Phase 1 establishes a +2 PAPE FL deficit; Phase 2 reduces it to +0.98 PAPE via post-hoc
codebook stacking. The interpretation is that **FL backbone training and codebook
construction are orthogonal contributors** — the codebook recovers the peak-relevant
structure that the FL training process under-specifies. This frames v01–v05's codebook as a
*reusable correction module* that improves any backbone, not as a v01-specific artefact.

### Federation contract preservation

The federated codebook (FL cells, 2-stage hierarchical) and the pooled codebook (centralised
cell) yield numerically equivalent diagnostic profiles (utilization 1.0, perplexity 26).
A privacy-conscious deployment can adopt the federated path with no measurable accuracy
penalty.

### Algorithm equivalence

Across both Phase 1 (BEFORE codebook) and Phase 2 (AFTER codebook), the five FL algorithms
sit within 0.5 PAPE of each other. None of FedProx's drift control, FedRep's
encoder/head separation, Ditto's personal model, or FedProto's prototype regulariser produces
a discriminating PAPE advantage at this scale (114 apartments, 20 rounds, 2 local epochs).
The recommended choice is **FedRep** (lowest comm cost) for FL training and **federated
codebook stacking** (Phase 2) for an additional ~5 PAPE.

### Limitations

1. **Single dataset.** All v06 numbers come from UMass 2016. Generalisation to UK-DALE,
   Pecan Street, or larger client populations is open. Deferred to future work.
2. **20 rounds with 2 local epochs.** Larger E or larger R may surface algorithm
   differences that are invisible at this scale. v07 (`plans/v07-01_loss_and_budget_sweeps.md`)
   sweeps `(E, R)` at fixed total budget T=80 to localise these effects; v06 paper does not
   pre-empt that sweep.
3. **Loss-weighting sweep deferred.** The peak-aux negative result of §4.2 was hypothesised
   to be either (i) heterogeneous label dilution under FedAvg or (ii) a `λ_aux` mis-tune
   carried over from v01's centralised setting. Discriminating between these is v07-A's job
   (`λ_aux ∈ {0, 0.05, 0.1, 0.2, 0.3}` × 6 cells × 3 seeds). v06 reports the negative result
   as-is; §5.3 establishes that the negative result *survives* Phase-2 codebook stacking
   (MAE-only + codebook strictly beats default + codebook), so the operating recipe for v06
   is `λ_aux = 0` + codebook regardless of v07's sweep outcome.
4. **Centralised codebook reference.** V6-Dyn-A_centralised is an "upper bound" reference but
   is itself a centralised protocol — a federated codebook on a federated backbone (FedCB on
   FedAvg-Aux) is the closest privacy-preserving pipeline, validated above as
   numerically equivalent (§5.2).
5. **Round-trajectory codebook.** Whether codebook lift grows monotonically with backbone
   round count or plateaus early is open; this requires Phase-1 re-execution with intermediate
   checkpointing. Deferred to v07-C.

## §7 Conclusion

v06 reframes peak-aware residential load forecasting from a cold-personalisation problem
(v01–v05) to a round-level FL training-dynamics problem. Five findings:

1. **The peak-aux head loss term hurts PAPE under round-level FL training**, opposite to its
   effect in the cold-zero-shot setting. A simple `λ_aux = 0` ablation reduces the FL PAPE
   gap from +2 to ~0.
2. **FL algorithm choice is not a discriminator at the v06 scale.** The five algorithms span
   0.5 PAPE within their seed std. FedRep wins on cost (−18 % comm).
3. **Post-hoc codebook stacking universally lifts PAPE by 4.5–5.9 points** across all
   backbones. The federated codebook is quality-equivalent to a pooled codebook. Together
   these reduce the FL-vs-centralised gap from +2.06 PAPE (BEFORE) to +0.98 PAPE (AFTER),
   making federated peak forecasting a viable Pareto choice when paired with the codebook
   correction.
4. **The peak-aux negative result is not erased by Phase-2 codebook stacking** (§5.3).
   Across every cell, dropping the peak-aux head and applying the codebook produces a
   *strictly better* test PAPE than keeping the peak-aux head and applying the same codebook.
   The recommended v06 operating recipe is **`λ_aux = 0` + federated codebook stacking**,
   yielding test PAPE = 44.2–44.8 % on FL cells and 44.4 % centralised — a ~5 PAPE absolute
   improvement over Phase 1's `λ_aux = 0.3` cells without correction.
5. **Codebook hyperparameters are robust** (§5.4, §5.5). The `α_v0` parameter spans an
   informative Pareto from MAE-zero-cost (α=0.5, ΔPAPE −3, ΔMAE ~0) to PAPE-extreme
   (α=2.0, ΔPAPE −8.65, ΔMAE +0.04). The federated `K_local = 2` design choice from v05
   FedCB is re-validated under the round-level FL protocol — K=1 loses ~1 PAPE of lift,
   K=8 adds ~0.5 PAPE for 4× the upload cost.

## §8 Reproducibility

| Driver | Role |
|---|---|
| `experiments/v06_round_dynamics/01_centralised.py` | V6-Dyn-A centralised pooled SGD |
| `experiments/v06_round_dynamics/02_fl_dynamics.py` | V6-Dyn-B-{Algo} 5-FL drivers |
| `experiments/v06_round_dynamics/06_aggregate.py`   | Phase 1 multi-seed aggregator |
| `experiments/v06_round_dynamics/07_make_figures.py`| F1 / F1b / F1c / F2 / F3 / F4 / F5 |
| `experiments/v06_round_dynamics/08_codebook_stacking.py` | Phase 2 per-cell stacking (`--ablation_suffix` for §5.3 / §5.4 / §5.5 sweeps) |
| `experiments/v06_round_dynamics/09_aggregate_codebook.py`| Phase 2 multi-seed aggregator |
| `experiments/v06_round_dynamics/10_make_codebook_figure.py`| F6 |
| `experiments/v06_round_dynamics/11_make_ablation_figures.py`| F7 (α_v0 Pareto) + F8 (K_local sweep) |

All scripts take `--seed S` per invocation; multi-seed launchers are listed in
`experiments/v06_round_dynamics/Readme.md` (sections "0501", "0502", "0503").

| Output | Path |
|---|---|
| Per-seed Phase 1 logs | `outputs/v06_round_dynamics/seed{S}/{cell}/round_log.jsonl` |
| Phase 1 multi-seed summary | `outputs/v06_round_dynamics/multiseed_summary.json` |
| Per-seed Phase 2 results | `outputs/v06_round_dynamics/seed{S}/{cell}/codebook_lift.json` |
| Phase 2 multi-seed summary | `outputs/v06_round_dynamics/codebook_lift_summary.json` |
| Phase 2 ablation outputs (§5.3 / §5.4 / §5.5) | `outputs/v06_round_dynamics/seed{S}/{cell}/codebook_lift{_alpha{V},_K{K}}.json` |
| Figures | `outputs/v06_round_dynamics/figures/F{1,1b,1c,2,3,4,5,6,7,8}_*.png` |

Tests:

```
uv run pytest tests/
# 32 passed (8 v06 codebook stacking + 6 round logger + 6 driver naming + ...)
```

Seeds: `{42, 123, 7}`. Backbone init reproducibility verified by
`init_backbone_aux(seed=S)` reseeding `torch.manual_seed` and `np.random.seed` per call.
