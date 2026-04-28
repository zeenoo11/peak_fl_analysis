# v04 — 09_fix_rerun: PR #1 fix re-runs + new priority experiments

> Successor folder under `experiments/v04_full_baseline_comparison/`. Contains
> *only* the cells that need to be (re-)run as a result of (a) bug fixes from
> PR #1's Copilot review and (b) four new priority experiments requested to
> strengthen the unified pFL paper (`papers/pfl_unified/paper.md`).

## Why this folder exists

PR #1's Copilot review caught four bugs (all already fixed in `4fdab45`):

1. `04_peakvq_on_fl.py:219` — `n_cold_apts` expression returned `1` for empty
   cold pools. Replaced with `int(len(cold_apts))`. **Affects no result file**
   (cold pool was always non-empty in practice).
2. `src/models/nhits.py` — MLP construction had an extra trailing activation
   between the last hidden `Linear` and the `theta` head, plus a missing
   activation after the first hidden `Linear`. Fixed to match the
   NeuralForecast official layout: every hidden `Linear` is followed by
   activation + optional dropout, the final `Linear -> n_theta` is bare.
   **The NHITS architecture changed**, so the existing v04 numbers
   (52.99 ± 1.64 PAPE) are stale — NHITS must be re-run. Param count is
   unchanged (1,057,098).
3. `src/fl/local_only.py` docstring — claimed "evaluate on held-out segment"
   but evaluated on the training segment. Docstring updated to honestly say
   "self-train + self-eval, overfit upper bound on its own data". **The
   evaluation behaviour was kept**; this folder's `05_local_only_holdout.py`
   is the redesigned proper held-out variant for fair comparison.
4. (Citation errors in the paper draft — paper-only, no code re-run.)

The parent folder (`01_*` … `08_*`) is **not re-run**. Only NHITS is re-run
because of bug #2; the other 12 cells are unaffected.

## Four new priority experiments

These were requested to ceteris-paribus the unified pFL paper's headline
"v01-v03 method outperforms every v04 baseline by 11.8 kW PAPE" finding.
Without these the headline conflates several independent factors
(centralised vs federated training, with vs without aux head, M=32 vs other
M, prototype-aligned representations).

| Priority | Script | Question answered |
|---|---|---|
| 1A | `02_fedavg_nbeatsx_aux.py` | "Is the gap due to *centralised pretraining* or *the auxiliary head*?" — federate the entire NBEATSxAux (backbone + aux_head). |
| 1B | `03_m_sensitivity.py` | "Is M=32 a load-bearing choice?" — sweep M ∈ {8, 16, 64} on the existing v02 T2 backbone. |
| 2C | `04_fedproto.py` | "Does a published prototype-based pFL baseline (FedProto) beat or match cluster-prototype recipe?" |
| —  | `05_local_only_holdout.py` | "What is Local-only's *fair generalisation* PAPE (not the overfit self-train + self-eval upper bound that the parent folder reports)?" |

## Scripts

| # | Script | Role | Output | Wall-clock per seed |
|---|---|---|---|---|
| 01 | `01_rerun_nhits.py` | Re-run NHITS with the fixed MLP. | `seed{S}/nf_nhits_fixed/{result.json, best.pt}` | ~6 min |
| 02 | `02_fedavg_nbeatsx_aux.py` | Priority 1A: FedAvg of NBEATSxAux (backbone + aux_head jointly), then Phase B Peak-VQ fit + Phase C cold W5 hybrid eval. | `seed{S}/fedavg_nbeatsx_aux/{result.json, final_state_dict.pt}` | ~15 min |
| 03 | `03_m_sensitivity.py` | Priority 1B: refit codebook on v02 T2 backbone for M ∈ {8, 16, 64}; cold eval at both op-points × R0 routing. | `seed{S}/m_sensitivity/result.json` | ~5 min |
| 04 | `04_fedproto.py` | Priority 2C: FedProto-style prototype-aligned FL; cold inference via 1-NN on global prototypes (no W5). | `seed{S}/fedproto/{result.json, prototypes.npz}` | ~25 min |
| 05 | `05_local_only_holdout.py` | Local-only redesigned with held-out evaluation (`series[train_end:val_end]`). Reports both self-eval (matches existing row) and the new holdout-eval. | `seed{S}/local_only_holdout/result.json` | ~15 min |

## Conventions

- **Per-seed argparse.** Every script takes `--seed S` per invocation; never
  put a `{42, 123, 7}` loop inside the script. The 3-seed sweep is dispatched
  externally (matches the parent folder's pattern).
- **bf16 + batch=512 for FL training.** The two FL-training scripts here
  (`02_fedavg_nbeatsx_aux.py`, `04_fedproto.py`) use bf16 autocast +
  `batch_size=512`, identical to the parent folder's FL pattern.
- **Output namespacing.** All results go under
  `outputs/v04_full_baseline_comparison/09_fix_rerun/seed{S}/{task}/result.json`
  to keep them separate from the parent folder's results.
- **`result.json` shape.** Same shape as the parent folder cells
  (`cold_metrics`, `seed`, `elapsed_seconds`, `config`, plus
  task-specific extras), so an aggregator that reads the parent folder
  can read these too with the additional task names.

## Running the sweep

The 3-seed sweep is dispatched externally — these scripts are NOT
self-looping. The user (or a launcher) issues:

```bash
for SEED in 42 123 7; do
    uv run python experiments/v04_full_baseline_comparison/09_fix_rerun/01_rerun_nhits.py --seed $SEED
    uv run python experiments/v04_full_baseline_comparison/09_fix_rerun/02_fedavg_nbeatsx_aux.py --seed $SEED
    uv run python experiments/v04_full_baseline_comparison/09_fix_rerun/03_m_sensitivity.py --seed $SEED
    uv run python experiments/v04_full_baseline_comparison/09_fix_rerun/04_fedproto.py --seed $SEED
    uv run python experiments/v04_full_baseline_comparison/09_fix_rerun/05_local_only_holdout.py --seed $SEED
done
```

Total estimated wall-clock: 3 seeds × (6 + 15 + 5 + 25 + 15) min ≈ **3.3 h**
serial (or ~70 min if dispatched concurrently as the parent folder did).

## Dependencies on prior outputs

- `03_m_sensitivity.py` requires
  `outputs/v02_fl_8020_ratio/seed{S}/T2/best.pt` (v02 frozen backbone).
- `02_fedavg_nbeatsx_aux.py` requires the v02 split YAML
  (`outputs/v02_fl_8020_ratio/splits/v02_8020_seed{S}.yaml`) — same
  dependency as every other v04 cell.
- `04_fedproto.py` requires the same v02 split YAML.
- `01_rerun_nhits.py`, `05_local_only_holdout.py` — split YAML only.

## What is NOT in this folder

- The parent folder's 01-08 scripts. They are *not* re-run; only NHITS is
  affected by the PR #1 fixes.
- v04 paper updates. The paper text is updated only after the new cells
  finish — paper edits live in `papers/v04_draft/` and `papers/pfl_unified/`.
- A `00_*` hyperparameter tuning script. The new cells use defaults
  consistent with the parent folder (rounds=20, local_epochs=2, lr=1e-3,
  batch=512); FedProto's `lambda_proto` default is documented in
  `04_fedproto.py`'s docstring with a 1-line justification.
