# CLAUDE.md

Orchestration guidance for Claude Code working in this repository. Detail lives in pointed-to files; keep this lean.

## Project context

Peak-aware residential load forecasting on UMass Smart* 2016 hourly data. NBEATSx + peak-aux head + post-hoc Peak-VQ + W5 hybrid Gaussian correction. **Method is frozen at v01**; only protocol/framing changes across versions.

- v01–v07 are past versions, **reference only**. See [`experiments/README.md`](experiments/README.md) (themes + frozen method spec + code conventions) and [`papers/README.md`](papers/README.md) (paper status manifest).
- **v10 is the active work** — v06-centric round-dynamics recap. New work goes there.

## Environment & commands

Python 3.11, `uv`-managed (`pyproject.toml` + `uv.lock`). Editable install of `peak-proto` exposing `src/` modules. PyTorch pinned to a CUDA 12.8 nightly index (Windows-only per `tool.uv.environments`); CPU machines fall back automatically. No tests, no linter, no build step. `pyproject.toml` declares `pythonpath = ["src"]` for pytest, but `tests/` does not exist yet.

Target hardware: RTX 5070 Ti (16 GB VRAM) + 64 GB system RAM. Batch sizes, federated client fan-out, and any in-memory window caching should be sized against the 16 GB VRAM ceiling — assume single-GPU, no model parallelism.

```bash
uv sync
uv run python experiments/v10_<theme>/01_<step>.py --seed 42
uv run python -c "from models.nbeatsx import MinimalNBEATSx"   # smoke import
```

## Data

- Raw CSVs: `data/raw/Umass/2016/Apt{N}_2016.csv`. **`data/` is gitignored and license-restricted** — never commit data files.
- External cold-protocol split: `../Peak_Analysis/configs/v10_households.yaml` (sibling repo). Required by `src/dataloader/splits.py:load_v10_split` for v01–v04 (raises `FileNotFoundError` if missing). **v06+ (incl. v10) does not use this file** — round-dynamics protocols use per-client internal splits.

## Method invariants (load-bearing — do not drift)

For the full method spec see [`experiments/README.md`](experiments/README.md). Hard constraints that apply every session:

- `MinimalNBEATSx` / `NBEATSxAux` state_dict keys are load-bearing — `Peak_Analysis/v10_b2` checkpoints load `strict=True`. Renaming layers requires a coordinated checkpoint migration.
- `VectorQuantizerKMeans.fit()` is **post-hoc 1-shot**. Iterative federated KMeans is deferred (TAR attack, arxiv:2511.07073). v05 implements the single-shot hierarchical federated variant.
- W5 operating points (HR-preserving σ=3.0/α_v0=1.0/α_w1=0.1; PAPE-aggressive σ=3.0/α_v0=1.5/α_w1=0.5) are carried **unchanged** across versions — do not re-tune on cold splits (would re-introduce v01 §5.4.1 concern).
- `src/config.py` constants (`INPUT_SIZE=96`, `HORIZON=24`, `D_MODEL=64`, `TRAIN_RATIO=0.7`, `VAL_RATIO=0.1`, `RANDOM_SEED=42`) are hard-coded. Per-apartment z-norm computed on training portion only.
- `src/utils/metrics.py` (PAPE, HR@k, MAE, MSE, `seven_axis_metrics`) is a **bit-exact port** of `Peak_Analysis/src/peak_analysis/metrics.py`. Definitions must not drift — v01 numbers must remain comparable across all versions.

## v10 — active work

- **Goal**: recap v06 round-dynamics results in a polished form. v01–v07 artefacts in `outputs/v{NN}_*/` are read-only reference, not to be re-run.
- v06 protocol summary (for context): per-client 70/10/20 internal split (no cold partition), all valid UMass apts via `filter_valid_apartments(min_hours=7000)`, FedSGD vs FedAvg axis, `local_epochs=40` → T=800 epoch-equiv per client. W5 / Peak-VQ are **out of scope** for round-dynamics protocols (cold-side correction; no cold partition exists).
- Layout: `plans/v10-*.md` (design), `experiments/v10_<theme>/NN_*.py` (numbered, run in order), `papers/v10_draft/`, `outputs/v10_*/seed{S}/...`.

## Conventions

- Multi-seed: all reported numbers use seeds `{42, 123, 7}`. Aggregate as **mean ± std across seeds**.
- **Per-seed CLI**: scripts take `--seed S` per invocation. Never put the seed loop inside the script.
- Output paths are version-namespaced: `outputs/v{NN}_<theme>/seed{S}/...`. Never write to a flat `outputs/`.
- Numbered scripts (`01_*.py`, `02_*.py`, ...) run in order; `sys.path.insert(0, 'src')` at script top is intentional.
- Version increments only when **protocol/framing** changes, not method.

## Agent workflow (`claude team`)

Default fan-out for v10 work: `lab-leader` → `engineer` → `executor` → `exp-critic`.

- **lab-leader** — coordinator. Reads project state, manages TODOs, dispatches to specialists. Use first when starting any new task.
- **engineer** — builder. Writes/edits scripts under `experiments/v10_*/` and helpers under `src/`. Adds pytest. Does **not** run multi-seed sweeps (executor's job).
- **executor** — verifier + runner. Integrity-checks engineer's code (state_dict load, normalisation, seeds, MLflow wiring), runs `{42, 123, 7}` as background tasks, records to `papers/v10_draft/`. Does **not** write code (sends back to engineer if a bug is found).
- **exp-critic** — sign-off on whether results are conclusive before they enter the paper.
- **Explore** — read-only search ("where is X / which files reference Y"). Use instead of doing manual grep tours.

Use `Agent` with `isolation: "worktree"` when parallel agents may touch overlapping files.
