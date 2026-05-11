# v07 — Loss-Weight Sensitivity, Round-Budget Sweep, Round-Trajectory Codebook

> Successor sweep paper to v06. v07 changes **no method** — backbone, codebook
> protocol, evaluation protocol all carry over from v06 unchanged. v07 quantifies
> how three orthogonal hyperparameters (peak-aux loss weight, round/local-epoch
> budget, codebook-fit timing) affect the v06 conclusions.
>
> Plan: `plans/v07-01_loss_and_budget_sweeps.md`.

## Output namespace

All v07 results live under `outputs/v07_loss_budget_sweeps/seed{S}/{cell}/...`.
The v06 driver scripts (`01_centralised.py`, `02_fl_dynamics.py`,
`08_codebook_stacking.py` under `experiments/v06_round_dynamics/`) accept
`--output_namespace v07_loss_budget_sweeps` to redirect their writes here
without touching the `outputs/v06_round_dynamics/` tree (v06 results stay
frozen for the v06 paper).

## Driver matrix (plan §5)

| Sub-paper | Stage | Driver | Reuse? |
|-----------|-------|--------|--------|
| v07-A     | Phase 1 train | `experiments/v06_round_dynamics/01_centralised.py --output_namespace v07_loss_budget_sweeps` | reuse v06 |
| v07-A     | Phase 1 train | `experiments/v06_round_dynamics/02_fl_dynamics.py --output_namespace v07_loss_budget_sweeps` | reuse v06 |
| v07-A     | aggregate | `05_aggregate_aux.py` | new (v07) |
| v07-A     | figure    | `08_make_figures.py --section aux` | new (v07) |
| v07-B     | Phase 1 train (FedAvg E sweep) | `experiments/v06_round_dynamics/02_fl_dynamics.py` w/ `--rounds R --local_epochs E` | reuse v06 |
| v07-B     | Phase 1 train (FedSGD)        | `03_fedsgd.py` | new (v07) — TBD |
| v07-B     | Phase 1 train (centralised)   | `experiments/v06_round_dynamics/01_centralised.py --epochs T` | reuse v06 |
| v07-B     | aggregate / figure | `06_aggregate_budget.py`, `08_make_figures.py --section budget` | new (v07) — TBD |
| v07-C     | Phase 1 with checkpoint_every | (v06 02 + new `--checkpoint_every` arg) | TBD |
| v07-C     | codebook re-stacking | `04_codebook_trajectory.py` | new (v07) — TBD |
| v07-C     | aggregate / figure | `07_aggregate_traj.py`, `08_make_figures.py --section traj` | new (v07) — TBD |

> **Status (2026-05-03 prep)**: only v07-A drivers are implemented. v07-B
> (`03_fedsgd.py`) and v07-C (`04_codebook_trajectory.py`) are scoped in
> `plans/v07-01` and will appear once v07-A results land.

## v07-A — λ_aux sweep (54 new runs, ~10h overnight)

**Goal**: localise the FL-optimal `λ_aux ∈ {0, 0.05, 0.1, 0.2, 0.3}`. v06 already
covers `λ=0` (`-MAEonly` namespace) and `λ=0.3` (default), so v07-A only
schedules `{0.05, 0.1, 0.2}` × 6 cells × 3 seeds = **54 new runs**.

The v07 launcher *re-uses* the existing v06 driver — only the
`--output_namespace` and `--aux_lambda` flags change. v06 results are not
touched.

### Reproducibility commands

PowerShell launchers (Windows; for bash on Linux replace `& "..."` with the
plain path).

```powershell
# v07-A λ_aux sweep — 54 runs (3 lambdas × 6 cells × 3 seeds).
# Recommended as one nightly batch. Each invocation is single-seed × single-cell
# × single-lambda (memory: feedback_argparse_per_seed).

$PY = "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe"
$ROOT = "c:/Users/HOME/JW/Research Docs/FL_Peak_Project"
$NS = "--output_namespace v07_loss_budget_sweeps"

# ====================================================================
# seed 42 × {0.05, 0.1, 0.2} × {centralised, FedAvg, FedProx, FedRep, Ditto, FedProto}
# ====================================================================
foreach ($lam in 0.05, 0.1, 0.2) {
  & $PY "$ROOT/experiments/v06_round_dynamics/01_centralised.py" `
       --seed 42 --epochs 40 --aux_lambda $lam --output_namespace v07_loss_budget_sweeps
  foreach ($algo in "fedavg", "fedprox", "fedrep", "ditto", "fedproto") {
    & $PY "$ROOT/experiments/v06_round_dynamics/02_fl_dynamics.py" `
         --algorithm $algo --seed 42 --local_epochs 40 --aux_lambda $lam `
         --output_namespace v07_loss_budget_sweeps
  }
}

# ====================================================================
# seed 123 × {0.05, 0.1, 0.2} × 6 cells
# ====================================================================
foreach ($lam in 0.05, 0.1, 0.2) {
  & $PY "$ROOT/experiments/v06_round_dynamics/01_centralised.py" `
       --seed 123 --epochs 40 --aux_lambda $lam --output_namespace v07_loss_budget_sweeps
  foreach ($algo in "fedavg", "fedprox", "fedrep", "ditto", "fedproto") {
    & $PY "$ROOT/experiments/v06_round_dynamics/02_fl_dynamics.py" `
         --algorithm $algo --seed 123 --local_epochs 40 --aux_lambda $lam `
         --output_namespace v07_loss_budget_sweeps
  }
}

# ====================================================================
# seed 7 × {0.05, 0.1, 0.2} × 6 cells
# ====================================================================
foreach ($lam in 0.05, 0.1, 0.2) {
  & $PY "$ROOT/experiments/v06_round_dynamics/01_centralised.py" `
       --seed 7 --epochs 40 --aux_lambda $lam --output_namespace v07_loss_budget_sweeps
  foreach ($algo in "fedavg", "fedprox", "fedrep", "ditto", "fedproto") {
    & $PY "$ROOT/experiments/v06_round_dynamics/02_fl_dynamics.py" `
         --algorithm $algo --seed 7 --local_epochs 40 --aux_lambda $lam `
         --output_namespace v07_loss_budget_sweeps
  }
}

# ====================================================================
# aggregate + figure
# ====================================================================
& $PY "$ROOT/experiments/v07_loss_budget_sweeps/05_aggregate_aux.py" --seeds 42 123 7
& $PY "$ROOT/experiments/v07_loss_budget_sweeps/08_make_figures.py" --section aux
```

A non-PowerShell flat alternative is provided as `01_run_aux_sweep.py` — a
launcher that fires all 54 runs sequentially via `subprocess.run`.

```bash
uv run python experiments/v07_loss_budget_sweeps/01_run_aux_sweep.py \
    --seeds 42 123 7 --lambdas 0.05 0.1 0.2 --algorithms fedavg fedprox fedrep ditto fedproto
```

### Cell name reference (v06 `_aux_suffix` carry-over)

| `--aux_lambda` | centralised cell                    | FL cell example                  |
|----------------|-------------------------------------|----------------------------------|
| 0.0  (v06 -MAEonly)  | `V6-Dyn-A_centralised-MAEonly`    | `V6-Dyn-B-FedAvg-MAEonly`     |
| 0.05 (v07 new) | `V6-Dyn-A_centralised-aux0.05`        | `V6-Dyn-B-FedAvg-aux0.05`        |
| 0.1  (v07 new) | `V6-Dyn-A_centralised-aux0.1`         | `V6-Dyn-B-FedAvg-aux0.1`         |
| 0.2  (v07 new) | `V6-Dyn-A_centralised-aux0.2`         | `V6-Dyn-B-FedAvg-aux0.2`         |
| 0.3  (v06 default)   | `V6-Dyn-A_centralised`            | `V6-Dyn-B-FedAvg`             |

The v06 directories at `outputs/v06_round_dynamics/seed{S}/V6-Dyn-A_centralised/`
are NOT copied or touched — they are reused only by the v07 aggregator,
which reads them via `--include_v06_baseline`.

### Aggregator + figure outputs

```
outputs/v07_loss_budget_sweeps/
├── seed{42,123,7}/V6-Dyn-{A_centralised, B-Algo}-aux{0.05,0.1,0.2}/...
├── aux_sweep_summary.json   # multi-seed aggregator output (algo × λ matrix)
└── figures/F-aux.png        # final test PAPE vs λ_aux per algorithm
```

## Expected wall-clock

Per-run cost matches v06 (same backbone + same protocol; only `λ_aux` changes):

| Cell | per-seed | × 3 lambdas × 3 seeds |
|------|----------|------------------------|
| centralised | ~25 s | ~4 min |
| FedAvg      | ~750 s | ~110 min |
| FedProx     | ~1600 s | ~240 min |
| FedRep      | ~700 s  | ~105 min |
| Ditto       | ~2900 s | ~435 min |
| FedProto    | ~900 s  | ~135 min |
| **total**   |          | **~17 h** |

Recommend nightly batch on a single CUDA GPU. If two GPUs are available, run
`{centralised, FedAvg, FedRep, FedProto}` on GPU 0 and `{FedProx, Ditto}` on
GPU 1 in parallel — total wall ≈10 h.

## v07-A1 — codebook stacking on the λ-swept centralised backbone (3 runs)

**Goal**: extend v06 §5.3 ("MAEonly + codebook beats default + codebook")
to the new interior-optimum centralised backbone `λ_aux = 0.1`. The
endpoints (`λ=0`, `λ=0.3`) are reused from v06's already-on-disk
`codebook_lift.json`; only the centralised `λ=0.1` cell needs new compute.

```powershell
$PY = "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe"
$ROOT = "c:/Users/HOME/JW/Research Docs/FL_Peak_Project"
foreach ($SEED in 42, 123, 7) {
  & $PY "$ROOT/experiments/v06_round_dynamics/08_codebook_stacking.py" `
       --seed $SEED --cell V6-Dyn-A_centralised-aux0.1 `
       --output_namespace v07_loss_budget_sweeps
}
& $PY "$ROOT/experiments/v07_loss_budget_sweeps/08_make_figures.py" --section codebook_lambda
```

Wall-clock: ≈15 s total. Output: `figures/F-codebook-vs-lambda.png` +
3 × `seed{S}/V6-Dyn-A_centralised-aux0.1/codebook_lift.json`.

## v07-A2 — hr_weight sweep (54 runs, ~5h overnight)

**Goal**: at fixed `λ_aux = 0.1` (centralised v07-A optimum), sweep
`hr_weight ∈ {0.05, 0.1, 0.5, 1.0}` to discriminate whether the
FL-incompatibility of peak-aux is concentrated in the peak-hour CE term
or carried by the entire peak-aux gradient. v07-A's `aux0.1` cells already
provide the `hr=0.1` (default) point; only the three new hr values run.

54 new runs = 3 hr_weights × 6 cells × 3 seeds. The launcher
`02_run_hr_weight_sweep.py` calls the v06 drivers with both
`--aux_lambda 0.1 --hr_weight {V}` and `--output_namespace v07_loss_budget_sweeps`.

```bash
uv run python experiments/v07_loss_budget_sweeps/02_run_hr_weight_sweep.py \
    --seeds 42 123 7 --hr_weights 0.05 0.5 1.0
uv run python experiments/v07_loss_budget_sweeps/08_make_figures.py --section hr
```

Cell-name suffix: `-aux0.1-hr{V}` (e.g.
`V6-Dyn-B-FedAvg-aux0.1-hr0.5`). Figure: `figures/F-hr.png`.

## Notes

- v07 does **not** introduce a new evaluation protocol or change the codebook
  hyperparameters (M=32, K_local=2, stride=24).
- v07 does **not** re-tune `α_v0` on the test split (CLAUDE.md / v01 §5.4.1
  invariant). The α sensitivity ablation lives in `papers/v06_draft` §5.4.
- All drivers take `--seed S` per invocation; multi-seed sweep is the
  launcher's job (memory: `feedback_argparse_per_seed`).
