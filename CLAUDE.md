# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Research codebase for **peak-aware residential load forecasting** under a (personalized) federated framing. UMass Smart* 2016 hourly data, NBEATSx backbone with a peak-auxiliary head, post-hoc Key–Value Vector Quantization (Peak-VQ), and a W5 hybrid Gaussian-template correction.

Each `v{NN}` ties together four directories: `plans/v{NN}-*.md` (design), `experiments/v{NN}_*/` (numbered scripts), `papers/v{NN}_draft/` (paper), `outputs/v{NN}_*/` (results, gitignored).

| Version | Status | Theme |
|---|---|---|
| v01 (`v01_peak_from_latent`) | complete | 50:50 train:cold, centralized, full method validation |
| v02 (`v02_fl_8020_ratio`) | complete | 80:20 zero-shot, PFL (FedHiP) + R0/R1 routing ablation |
| v03 (`v03_kshot_pfl`) | complete | K=1 month K-shot, F2 family (head / last-layer / LoRA) |
| v04 (`v04_full_baseline_comparison`) | complete | FL × Neural Forecasting × Foundation Model baseline matrix |
| v05 (`v05_fedcb_codebook`) | complete | FedCB — 2-stage hierarchical federated codebook + FedAvg-Aux |
| `papers/pfl_unified/` | drafted | Cross-version unified pFL paper; consumes v02/v04/v05 |
| `experiments/conference/` | drafted | KIIE FL-only finalised pipeline (Phase A/B/C + codebook ablation) |
| v06 (`v06_round_dynamics`) | drafted | Round-level FL dynamics, N=114, per-client 70/10/20, **no cold partition**. Phase 1 = round trajectory; Phase 2 = post-hoc federated codebook stacking. `local_epochs=40` → T=800 epoch-equiv |
| v07 (`v07_loss_budget_sweeps`) | drafted | 2-axis loss-weight sweep on v06 protocol (λ_aux, hr_weight); FL cells have strict boundary optimum λ=0; v07-B/C deferred |

The unified paper and conference pipeline **consume** per-version artefacts rather than introducing new protocols.

## Environment & commands

Python 3.11, uv-managed (`pyproject.toml` + `uv.lock`). Editable install of `peak-proto` exposing `src/` modules. PyTorch pinned to CUDA 12.8 nightly (Windows-only per `tool.uv.environments`); CPU fallback via `DEVICE`.

```bash
uv sync
uv run python experiments/v01_peak_from_latent/01_train_arms.py --arms T0 T2 --seed 42
uv run pytest                                # tests under tests/
```

Experiment scripts take `--seed S` per invocation; never put the `{42, 123, 7}` loop inside the script. No linter / no build step — scripts run directly and prepend `src/` to `sys.path`.

## Data

- Raw UMass CSVs at `data/raw/Umass/2016/Apt{N}_2016.csv`. `data/` is gitignored and license-restricted.
- v01 split is loaded from an **external** sibling-repo file: `../Peak_Analysis/configs/v10_households.yaml` (`src/dataloader/splits.py:load_v10_split` raises `FileNotFoundError` if missing).
- v02+ split YAMLs live under `outputs/v{NN}_*/splits/`. v06+ uses `filter_valid_apartments(min_hours=7000)` → N=114, **no cold partition**.

## Method (frozen across versions — v02+ only change protocol/framing)

1. **NBEATSx backbone** (`src/models/nbeatsx.py:MinimalNBEATSx`) — 3 stacks (trend / seasonal / generic). Returns `(y_hat, hiddens={h_trend, h_seasonal, h_generic} ∈ ℝ⁶⁴)`. Layer names match `Peak_Analysis/v10_b2` for `strict=True` checkpoint reuse.
2. **Peak-aux head** (`src/models/peak_aux_head.py`) — `MAE(y) + λ · peak_aux(y)` with `λ=0.3`, `hr_weight=0.1`. Wrapped via `NBEATSxAux(latent_source='h_generic'|'h_concat')`.
3. **Post-hoc Peak-VQ** (`src/models/vq_kmeans.py:VectorQuantizerKMeans`) — 1-shot KMeans++, M=32, D=64, fit on `h_generic` of train windows. Frozen after fit.
4. **W5 hybrid correction** — `ŷ_corr = ŷ_base + α_v0·o_{c*} + α_w1·g(t; ĥ, â, σ)`. Two operating points carried **unchanged** across all versions (do not re-tune on cold split):
   - HR-preserving: σ=3.0, α_v0=1.0, α_w1=0.1
   - PAPE-aggressive: σ=3.0, α_v0=1.5, α_w1=0.5
5. **Routing** — KEY descriptor (`src/probes/peak_descriptor.py`) is 5-d `[max, argmax/96, mean, std, last24_max]`. v01 = R0 (5-d KEY-NN); v02 adds R1 (64-d `h_g_cold` → nearest centroid).

Hard-coded shapes in `src/config.py`: `INPUT_SIZE=96`, `HORIZON=24`, `D_MODEL=64`, `TRAIN_RATIO=0.7`, `VAL_RATIO=0.1`. Per-apartment z-norm fit on training portion only.

## Invariants (don't drift)

- **state_dict keys are load-bearing**: `MinimalNBEATSx` / `NBEATSxAux` layer names load v10 b2 checkpoints with `strict=True`. Don't rename without checkpoint migration.
- **`latent_source='h_concat'`** (T3 arm) returns ℝ¹⁹²; fitting VQ on it requires `embedding_dim=192`.
- **`VectorQuantizerKMeans.fit()` is post-hoc 1-shot**. Iterative federated KMeans is deferred (TAR attack, arxiv:2511.07073). v05 and the conference pipeline use the single-shot hierarchical federated variant via `src/fl/codebook_fl.py`.
- **Metrics** (`src/utils/metrics.py`: PAPE, HR@k, MAE, MSE, `seven_axis_metrics`) are bit-exact ports from `Peak_Analysis/src/peak_analysis/metrics.py` — must not drift across versions.

## Conventions

- Multi-seed `{42, 123, 7}`, reported as mean ± std.
- Version increments on **protocol/framing** change, not method change. Backbone is treated as frozen across v02/v03 (FedHiP framing).
- Output paths are version-namespaced: `outputs/v{NN}_<theme>/seed{S}/...`. Never a flat `outputs/` root. Conference pipeline outputs live under `outputs/conference/{pipeline,ablation}/seed{S}/`.
- Cross-version paper manifest: see `papers/README.md`.
- **Markdown prose is not hard-wrapped** — one paragraph = one line. Tables, code fences, list items, blockquotes wrap naturally; only mid-paragraph prose stays on a single line. Applies to `plans/`, `papers/`, `docs/`.
