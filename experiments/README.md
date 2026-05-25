# Experiments — `experiments/`

Code-side per-version experiment scripts. Each `v{NN}_*/` directory contains numbered Python scripts (`01_*.py`, `02_*.py`, ...) run in order. v01–v07 are **reference**; v10 is active.

For paper-side status (which drafts are done) see [`../papers/README.md`](../papers/README.md). For per-version protocol design see `../plans/v{NN}-*.md`.

## Per-version themes

| Version | Directory | Theme | Plan |
|---|---|---|---|
| v01 | `v01_peak_from_latent/` | 50:50 train:cold, centralised framing, full method validation | `plans/v01-01_peak_from_latent_test.md` |
| v02 | `v02_fl_8020_ratio/` | 80:20 zero-shot under PFL (FedHiP) framing + R0/R1 routing ablation | `plans/v02-01_fl_8020_ratio.md` |
| v03 | `v03_kshot_pfl/` | K=1 month K-shot adaptation, F2 family (head / last-layer / LoRA) | `plans/v03-01_kshot_pfl.md` |
| v04 | `v04_full_baseline_comparison/` | FL × Neural Forecasting × Foundation Model baseline matrix | `plans/v04-01_full_baseline_comparison.md` |
| v05 | `v05_fedcb_codebook/` | FedCB — 2-stage hierarchical federated codebook + FedAvg-Aux backbone | `plans/v05-01_fedcb_codebook.md` |
| v06 | `v06_round_dynamics/` | Round-level FL training dynamics — per-client 70/10/20 (no cold), 6 cells × 3 seeds × 20 rounds, `local_epochs=40` (T=800 epoch-equiv; plan originally E=2, see audit S1 in `docs/v06_v07_crosscheck_audit.md`) | `plans/v06-01_round_dynamics.md` |
| v07 | `v07_loss_budget_sweeps/` | 2-axis loss-weight sweep on v06 protocol: λ_aux × hr_weight, ~107 runs. Centralised λ*=0.1; all FL cells λ*=0 ("peak-aux is FL-incompatible at any positive λ"). v07-B / v07-C deferred | `plans/v07-01_loss_and_budget_sweeps.md` |
| conference | `conference/` | KIIE conference final FL-only pipeline (personalised path deferred). `pipeline/01_phase_a_*.py` (FedAvg-NBEATSxAux backbone), `02_phase_b_*.py` (federated codebook via `src/fl/codebook_fl.py`), `03_phase_c_*.py` (CMO-only cold inference). Outputs under `outputs/conference/{pipeline,ablation}/seed{S}/`. Pairs with `papers/conference_draft/` | — |

## Method (frozen across all versions)

The method is fixed at v01's design — later versions vary only protocol/framing.

1. **NBEATSx backbone** (`src/models/nbeatsx.py:MinimalNBEATSx`) — 3 stacks (trend / seasonal / generic). Forward returns `(y_hat, hiddens={h_trend, h_seasonal, h_generic})`, each ∈ ℝ⁶⁴. **Layer names match `Peak_Analysis/v10_b2` for `strict=True` checkpoint reuse — do not rename.**
2. **Peak-aux head** (`src/models/peak_aux_head.py`) — peak amplitude (MSE) + peak hour (CE). Loss = `MAE(y) + λ · peak_aux(y)` with λ=0.3, hr_weight=0.1. Wrapped via `NBEATSxAux(latent_source='h_generic'|'h_concat')`. `h_concat` ∈ ℝ¹⁹² (T3 arm); the aux head adapts but VQ does not — fitting VQ on h_concat needs `embedding_dim=192`.
3. **Post-hoc Peak-VQ** (`src/models/vq_kmeans.py:VectorQuantizerKMeans`) — 1-shot KMeans++ with M=32, D=64 fit on `h_generic` of train windows. Frozen after fit; no STE, no training-time quantisation. Iterative federated KMeans is deferred (TAR attack, arxiv:2511.07073). v05's `src/fl/codebook_fl.py` realises the single-shot hierarchical federated variant.
4. **W5 hybrid correction** — `ŷ_corr = ŷ_base + α_v0·o_{c*} + α_w1·g(t; ĥ, â, σ)` where `o_{c*}` is the cluster's residual offset and `g` is a Gaussian template parameterised by the aux head's predicted `(ĥ, â)`. Two operating points carried **unchanged** across all versions:
   - HR-preserving: σ=3.0, α_v0=1.0, α_w1=0.1
   - PAPE-aggressive: σ=3.0, α_v0=1.5, α_w1=0.5
5. **Routing** — KEY descriptor (`src/probes/peak_descriptor.py`) is a 5-d input-only summary `[max, argmax/96, mean, std, last24_max]`. v01 routes via 5-d KEY-NN (R0); v02 adds R1 = 64-d `h_g_cold` → nearest centroid.

## Hard-coded shapes (`src/config.py`)

`INPUT_SIZE=96`, `HORIZON=24`, `D_MODEL=64`, `N_POLYNOMIALS=3`, `N_HARMONICS=5`, `TRAIN_RATIO=0.7`, `VAL_RATIO=0.1`, `RANDOM_SEED=42`. Per-apartment z-norm computed on the training portion only.

## Metrics (`src/utils/metrics.py`)

PAPE (peak abs % error on horizon-max), HR@k (peak-position hit rate within ±k), MAE, MSE, plus `seven_axis_metrics` aggregator. Definitions are **bit-exact ports** of `Peak_Analysis/src/peak_analysis/metrics.py` and **must not drift** — v01 numbers must remain comparable across versions.

## Code conventions

- Numbered scripts run in order. `sys.path.insert(0, 'src')` at script top is intentional — a checkout works without `uv sync`-ing the editable `peak-proto` package itself.
- Per-seed CLI: every script takes `--seed S` per invocation. Never put the `{42, 123, 7}` loop inside a script.
- Output paths: `outputs/v{NN}_<theme>/seed{S}/...`. Conference uses `outputs/conference/{pipeline,ablation}/seed{S}/`. Never write to a flat `outputs/`.
- Backbone is treated as frozen across v02/v03 (FedHiP framing): only the codebook (v02) or a small adapter (v03 F2 family) varies. v04 adds external baselines, v05 federates the codebook fit, conference restricts to the FL-only path.

## Cross-version artefacts

- **Unified paper** (`papers/pfl_unified/`) consumes results from v02 / v04 / v05 — single-shot inference-time codebook routing as a pFL design point. Does not introduce a new evaluation protocol.
- **Conference pipeline** (`experiments/conference/`) re-uses extracted helpers from v04/v05 via `src/fl/codebook_fl.py` forwarders only (federation contract enforced by import structure — no centralised pooling helpers).
