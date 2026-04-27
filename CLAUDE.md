# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Research codebase for **peak-aware residential load forecasting** under a (personalized) federated framing. UMass Smart* 2016 hourly data, NBEATSx backbone with a peak-auxiliary head, post-hoc Key–Value Vector Quantization (Peak-VQ), and a W5 hybrid Gaussian-template correction.

The repo is organised by **paper version**, each one a self-contained re-evaluation of the same method under a different protocol:

| Version | Status | Theme |
|---|---|---|
| v01 (`v01_peak_from_latent`) | **complete** | 50:50 train:cold, centralized framing, full method validation |
| v02 (`v02_fl_8020_ratio`) | scaffolding | 80:20 zero-shot under PFL (FedHiP) framing + R0/R1 routing ablation |
| v03 (`v03_kshot_pfl`) | scaffolding (blocked on v02) | K=1 month K-shot adaptation, F2 family (head / last-layer / LoRA) |

Each version `v{NN}` ties together four directories: `plans/v{NN}-*.md` (design doc), `experiments/v{NN}_*/` (numbered scripts), `papers/v{NN}_draft/` (paper draft), `outputs/v{NN}_*/` (results, gitignored).

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

Each `experiments/v{NN}_*/` directory contains numbered Python scripts (`01_*.py`, `02_*.py`, ...) that are run in order. They prepend `src/` to `sys.path` manually rather than relying on the installed package — this is intentional so a checkout works without `uv sync` of the `peak-proto` package itself. v01 has 21 scripts; v02 plans 7; v03 plans 5+1.

### Metrics (`src/utils/metrics.py`)

`PAPE` (peak abs % error on horizon-max), `HR@k` (peak-position hit rate within ±k), `MAE`, `MSE`, plus a `seven_axis_metrics` aggregator. Definitions are bit-exact ports from `Peak_Analysis/src/peak_analysis/metrics.py` and **must not drift** — v01 numbers must remain comparable across versions.

## Conventions

- **Multi-seed**: all reported numbers use seeds `{42, 123, 7}`. Aggregation is "mean ± std across seeds".
- **Version increments** when the *evaluation protocol or framing* changes, not when the method changes (the method is frozen at v01's NBEATSx + W5 hybrid).
- **Operating points are carried over unchanged** across v02/v03 — do not re-tune (σ, α_v0, α_w1) on the cold split, that would re-introduce the v01 §5.4.1 concern.
- **Backbone is treated as frozen across v02/v03** (FedHiP framing): only the codebook (v02) or a small adapter (v03 F2 family) varies.
- **Output paths are version-namespaced**: `outputs/v{NN}_<theme>/seed{S}/...`. Never write to a flat `outputs/` root.
- v01 paper artifacts live in `papers/v01_draft/` (md + tex + figures); see `papers/README.md` for the cross-version manifest.

## Things to know before changing model code

- `MinimalNBEATSx` and `NBEATSxAux` state_dict keys are load-bearing — v10 b2 checkpoints load `strict=True`. Do not rename layers without coordinating a checkpoint migration.
- `latent_source='h_concat'` (T3 arm) returns `cat(h_trend, h_seasonal, h_generic)` ∈ ℝ¹⁹². The aux head's `in_dim` adapts; the codebook does not — fitting VQ on `h_concat` requires `embedding_dim=192`.
- `VectorQuantizerKMeans.fit()` is **post-hoc 1-shot**. Iterative federated KMeans is explicitly out of scope through v03 (deferred to v04+) because of the TAR attack (arxiv:2511.07073).
