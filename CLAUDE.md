# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Research codebase for **peak-aware residential load forecasting** under a (personalized) federated framing. UMass Smart* 2016 hourly data, NBEATSx backbone with a peak-auxiliary head, post-hoc Key–Value Vector Quantization (Peak-VQ), and a W5 hybrid Gaussian-template correction.

The repo is organised by **paper version**, each one a self-contained re-evaluation of the same method under a different protocol:

| Version | Status | Theme |
|---|---|---|
| v01 (`v01_peak_from_latent`) | **complete** | 50:50 train:cold, centralized framing, full method validation |
| v02 (`v02_fl_8020_ratio`) | **complete** | 80:20 zero-shot under PFL (FedHiP) framing + R0/R1 routing ablation |
| v03 (`v03_kshot_pfl`) | **complete** | K=1 month K-shot adaptation, F2 family (head / last-layer / LoRA) |
| v04 (`v04_full_baseline_comparison`) | **complete** | FL × Neural Forecasting × Foundation Model baseline matrix |
| v05 (`v05_fedcb_codebook`) | **complete** | FedCB — 2-stage hierarchical *federated* codebook + FedAvg-Aux backbone |
| **`papers/pfl_unified/`** | **drafted** | Cross-version unified pFL paper — single-shot inference-time codebook routing as a new pFL design point. Consumes results from v02 / v04 / v05. |
| **`experiments/conference/`** | **drafted** | KIIE-conference final pipeline (FL-only, personalised path deferred). Phase A/B/C drivers in `pipeline/` + `ablation/codebook_module_effect.py`; pairs with `papers/conference_draft/presentation.md`. |
| v06 (`v06_round_dynamics`) | **drafted** | Round-level FL training dynamics — per-client 70/10/20 split (no cold partition), 6 cells (centralised + 5 FL) × 3 seeds × 20 rounds, local_epochs=40 (T=800 epoch-equiv; plan originally specified E=2, see audit S1). Phase 1 = round-level trajectory; Phase 2 = post-hoc federated codebook stacking. 8 figures F1–F8. |
| v07 (`v07_loss_budget_sweeps`) | **drafted** | Loss-weight sensitivity sweep on the v06 protocol — 2-axis coordinate sweep: λ_aux ∈ {0, 0.05, 0.1, 0.2, 0.3} at fixed hr_weight=0.1, and hr_weight ∈ {0.05, 0.1, 0.5, 1.0} at fixed λ=0.1; each axis × 6 algos × 3 seeds (~107 unique runs, not a 5×4 cross-product). Centralised has interior optimum λ=0.1; all 5 FL cells have strict boundary optimum λ=0 ("peak-aux is FL-incompatible at any positive λ"). v07-B (round budget, T=800 epoch-equiv) and v07-C (round-trajectory codebook) deferred. |

Each version `v{NN}` ties together four directories: `plans/v{NN}-*.md` (design doc), `experiments/v{NN}_*/` (numbered scripts), `papers/v{NN}_draft/` (paper draft), `outputs/v{NN}_*/` (results, gitignored). The unified paper and the conference pipeline live outside this `v{NN}` grid: they consume already-produced per-version artefacts rather than introducing a new protocol of their own.

## Environment & commands

Python 3.11, `uv`-managed (`pyproject.toml` + `uv.lock`). Editable install of `peak-proto` exposing `src/` modules.

```bash
uv sync                                                     # install deps + create .venv
uv run python experiments/v01_peak_from_latent/01_train_arms.py --arms T0 T2 --seed 42
uv run python -c "from models.nbeatsx import MinimalNBEATSx"   # smoke import check
```

There are no tests, no linter config, and no build step. Scripts are run directly. `pyproject.toml` declares `pythonpath = ["src"]` for pytest, but no `tests/` exists yet.

PyTorch is pinned to a CUDA 12.8 nightly index (Windows-only environment per `tool.uv.environments`). On CPU-only machines, `DEVICE` falls back automatically.

## Data location

- Raw UMass CSVs live under `data/raw/Umass/2016/Apt{N}_2016.csv`. **`data/` is gitignored** and license-restricted; never commit data files.
- The household train/cold split for v01 is loaded from an **external file**: `../Peak_Analysis/configs/v10_households.yaml` (sibling repo). `src/dataloader/splits.py:load_v10_split` will raise `FileNotFoundError` if that path is missing. v02 will introduce its own 80:20 split YAML under `outputs/v02_fl_8020_ratio/splits/`.

## Architecture

### Method (frozen across versions)

The method is fixed at v01's design — v02 and v03 only change protocol/framing:

1. **NBEATSx backbone** (`src/models/nbeatsx.py:MinimalNBEATSx`) — 3 stacks (trend / seasonal / generic). Forward returns `(y_hat, hiddens)` where `hiddens = {h_trend, h_seasonal, h_generic}` ∈ ℝ⁶⁴ each. Layer names match `Peak_Analysis/v10_b2` for `strict=True` checkpoint reuse.
2. **Peak-aux head** (`src/models/peak_aux_head.py`) — regress peak amplitude (MSE) + classify peak hour (CE). Loss: `MAE(y) + λ · peak_aux(y)` with `λ=0.3`, `hr_weight=0.1`. Wrapped via `NBEATSxAux(latent_source='h_generic'|'h_concat')`.
3. **Post-hoc Peak-VQ** (`src/models/vq_kmeans.py:VectorQuantizerKMeans`) — 1-shot KMeans++ with M=32, D=64 fit on `h_generic` of train windows. Frozen after fit; no STE, no training-time quantization.
4. **W5 hybrid correction** — `ŷ_corr = ŷ_base + α_v0·o_{c*} + α_w1·g(t; ĥ, â, σ)` where `o_{c*}` is the cluster's residual offset and `g` is a Gaussian template parameterised by the aux-head's predicted `(ĥ, â)`. Two operating points carried across all versions:
   - **HR-preserving**: σ=3.0, α_v0=1.0, α_w1=0.1
   - **PAPE-aggressive**: σ=3.0, α_v0=1.5, α_w1=0.5
5. **Routing** — KEY descriptor (`src/probes/peak_descriptor.py`) is a 5-d input-only summary `[max, argmax/96, mean, std, last24_max]`. v01 routes via 5-d KEY-NN (R0); v02 adds R1 = 64-d `h_g_cold` → nearest centroid.

### Hard-coded shapes (`src/config.py`)

`INPUT_SIZE=96`, `HORIZON=24`, `D_MODEL=64`, `N_POLYNOMIALS=3`, `N_HARMONICS=5`, `TRAIN_RATIO=0.7`, `VAL_RATIO=0.1`, `RANDOM_SEED=42`. Per-apartment z-norm is computed on the training portion only.

### Per-version experiment scripts

Each `experiments/v{NN}_*/` directory contains numbered Python scripts (`01_*.py`, `02_*.py`, ...) that are run in order. They prepend `src/` to `sys.path` manually rather than relying on the installed package — this is intentional so a checkout works without `uv sync` of the `peak-proto` package itself. v01 has 21 scripts; v02–v05 are similarly ordered.

### Conference pipeline (`experiments/conference/`)

The KIIE conference submission is the **FL-only finalised path** (personalised K-shot adaptation deferred). It re-uses extracted helpers from v04/v05 rather than copying internals:

- `pipeline/01_phase_a_train_backbone.py` — FedAvg-NBEATSxAux backbone (bit-equivalent to v04 09_fix_rerun's `02_fedavg_nbeatsx_aux.py`).
- `pipeline/02_phase_b_fit_federated_codebook.py` — 2-stage hierarchical federated KMeans + federated residual offsets via `src/fl/codebook_fl.py` forwarders only (no centralised pooling helpers — federation contract enforced by import structure).
- `pipeline/03_phase_c_cmo_inference.py` — cold inference with **CMO-only** correction (Gaussian template α_w1 dropped; α_v0 default 1.0). Saves `cold_arrays.npz` so ablations can recompute PAPE / HR@k / kW²-MSE without re-running the model.
- `ablation/codebook_module_effect.py` — multi-seed aggregator that reproduces the *Backbone* vs *Backbone + Codebook* table in `papers/conference_draft/presentation.md`.

Outputs are namespaced under `outputs/conference/{pipeline,ablation}/seed{S}/` rather than under a `v{NN}` root.

### Unified paper (`papers/pfl_unified/`)

`paper.md` + `figures/` (F1 Pareto, F6 decomposition, F7 M sensitivity, F8 sorted unified) consolidate v02 routing, v04 baseline matrix, and v05 federated-codebook results into a single pFL submission framing. It does **not** introduce a new evaluation protocol; the figures' source numbers come from the per-version `outputs/v{NN}_*/` artefacts.

### Metrics (`src/utils/metrics.py`)

`PAPE` (peak abs % error on horizon-max), `HR@k` (peak-position hit rate within ±k), `MAE`, `MSE`, plus a `seven_axis_metrics` aggregator. Definitions are bit-exact ports from `Peak_Analysis/src/peak_analysis/metrics.py` and **must not drift** — v01 numbers must remain comparable across versions.

## Conventions

- **Multi-seed**: all reported numbers use seeds `{42, 123, 7}`. Aggregation is "mean ± std across seeds".
- **Version increments** when the *evaluation protocol or framing* changes, not when the method changes (the method is frozen at v01's NBEATSx + W5 hybrid).
- **Operating points are carried over unchanged** across v02/v03 — do not re-tune (σ, α_v0, α_w1) on the cold split, that would re-introduce the v01 §5.4.1 concern.
- **Backbone is treated as frozen across v02/v03** (FedHiP framing): only the codebook (v02) or a small adapter (v03 F2 family) varies. v04 adds external baselines, v05 federates the codebook fit itself, and the conference pipeline restricts to the FL-only path.
- **Output paths are version-namespaced**: `outputs/v{NN}_<theme>/seed{S}/...`. Never write to a flat `outputs/` root.
- v01 paper artifacts live in `papers/v01_draft/` (md + tex + figures); see `papers/README.md` for the cross-version manifest.

## Things to know before changing model code

- `MinimalNBEATSx` and `NBEATSxAux` state_dict keys are load-bearing — v10 b2 checkpoints load `strict=True`. Do not rename layers without coordinating a checkpoint migration.
- `latent_source='h_concat'` (T3 arm) returns `cat(h_trend, h_seasonal, h_generic)` ∈ ℝ¹⁹². The aux head's `in_dim` adapts; the codebook does not — fitting VQ on `h_concat` requires `embedding_dim=192`.
- `VectorQuantizerKMeans.fit()` is **post-hoc 1-shot**. Iterative federated KMeans is explicitly out of scope through v03 (deferred to v04+) because of the TAR attack (arxiv:2511.07073). v05's `fl/codebook_fl.py` realises the *single-shot* hierarchical federated variant; iterative refinement is still deferred.

## v06 — round-level FL training dynamics (planned)

v01–v05 all evaluate on a **cold-client zero-shot** held-out partition — that is a *cold personalisation* protocol, not a federated-learning protocol per se. v06 abandons the cold partition entirely and measures **federated learning itself**: every valid UMass apartment (N=100, after `filter_valid_apartments(min_hours=7000)`) participates in training; each apartment is evaluated on its own internal val/test windows; metrics are logged at every FL communication round.

Per-apartment internal split: `TRAIN_RATIO=0.7`, `VAL_RATIO=0.1`, test = remaining 0.2 (CLAUDE.md `src/config.py` constants). v06 does **not** read `outputs/v02_fl_8020_ratio/splits/` or `Peak_Analysis/configs/v10_households.yaml` — those are cold-protocol artefacts.

Per-round evaluation contract (FedAvg-style cells):

```
For round t = 1 .. R:
    For each client i ∈ all 100:
        local-train E epochs on i's training portion
    Server: weighted-mean aggregation → new global θ
    Broadcast θ
    Eval: each client computes per-window forecast on its val portion;
          server averages PAPE/HR@k/MAE across clients
    Log: per-round val metrics, cumulative comm bytes, client drift L2
```

Comparison axis = **FedSGD vs FedAvg** (McMahan 2017's original two-methodology framing); see `docs/fl_methodologies_fedsgd_vs_fedavg.md` for the analysis. Actual v06 execution used `local_epochs=40` (plan originally specified 2; see audit S1 in `docs/v06_v07_crosscheck_audit.md`), giving T=800 epoch-equiv per client. v07-B budget sweep uses T=800 as the fixed budget. v07 headline experiment is an `E ∈ {1, 2, 5, 10, 20}` sweep at fixed total budget T=800 epoch-equivalent, plus a FedSGD reference (1 SGD step / round, ≈800 rounds), plus a centralised pooled SGD upper bound. Outputs are *round-vs-val-PAPE*, *bytes-vs-val-PAPE*, and *drift-vs-round* trajectory figures. The W5 / Peak-VQ codebook is **out of scope** for v06 (it is a cold-side correction module; without a cold partition it has no role). Scaffolding lives under `plans/v06-01_round_dynamics.md`; `experiments/v06_*/`, `papers/v06_draft/`, `outputs/v06_*/` will appear when implementation starts.
