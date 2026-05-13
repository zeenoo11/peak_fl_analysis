# v07-01 — Loss-Weight Sensitivity, Round-Budget Sweep, and Round-Trajectory Codebook

> Successor to `v06-01_round_dynamics.md`. v06 finalised the round-level FL protocol (per-client 70/10/20 split, Phase 1 = 6 cells × 3 seeds, Phase 2 = federated codebook stacking). v06 surfaced two open questions that v06's compute budget could not answer and one trajectory question that requires Phase 1 re-execution; v07 owns those three deliverables.
>
> v07 introduces **no new method** — backbone, codebook protocol, evaluation protocol are all carried over from v06 unchanged. v07 is a *sweep paper* that quantifies how three orthogonal hyperparameters (peak-aux loss weight, round/local-epoch budget, codebook-fit timing) affect the v06 conclusions.

> **Status (2026-05-03).** Plan only. v06 paper is drafted in `papers/v06_draft/v06_round_dynamics.md`; v07 implementation begins after v06 finalisation.

---

## §0 Motivation — v06 unanswered questions

### v06 §6 limitation 1 — peak-aux loss weight

v06 reports a *negative* effect of `λ_aux = 0.3` (peak-aux head loss term) under round-level FL training: every FL cell's test PAPE worsens by 2.3–3.5 points relative to its `λ_aux = 0` (MAE-only) ablation. v06 §6 hypothesises two possible mechanisms:

1. *Heterogeneous label distribution.* The auxiliary head's 24-class CE on peak hour is sensitive to per-client peak-hour distributions; FedAvg gradient averaging dilutes the signal.
2. *Loss-weighting mis-tune.* `λ_aux = 0.3` was tuned on v01's centralised training; federation may need a smaller value.

v06 reports the negative result as-is and defers the explanation. v07 §1 performs a `λ_aux ∈ {0, 0.05, 0.1, 0.2, 0.3}` sweep to localise the optimum and discriminate between hypotheses (1) and (2).

### v06 §6 limitation 2 — round / local-epoch budget

v06 fixes `R = 20, E = 40` for all FL cells (actual execution; plan originally specified E=2 — see audit S1). T = 800 epoch-equiv per client. The chosen budget is not v06-specifically optimal: more local epochs accelerate convergence at the cost of larger client drift, and FedSGD (E = 1, large R) is the natural reference at the other end of the spectrum.

`docs/fl_methodologies_fedsgd_vs_fedavg.md` already lays out the analysis plan: an `E ∈ {1, 2, 5, 10, 20}` sweep at fixed total budget T = 800 epoch-equivalent (matching v06 actual T=800), plus an FedSGD reference (1 SGD step per round, ~800 rounds) and a centralised pooled-SGD upper bound. v07 §2 implements this sweep on the v06 backbone / split / aggregator.

### v06 Phase 2 follow-up — round-trajectory codebook

v06 Phase 2 reports a single number per cell: the codebook lift evaluated on the *final* (round-20) backbone. An open question is whether the codebook lift grows monotonically with backbone round count (i.e. a better-trained backbone has a richer `h_generic` cluster structure → larger codebook lift) or plateaus early. v07 §3 saves intermediate backbones at rounds {5, 10, 15, 20} for every Phase-1 cell × seed and re-runs the v06 Phase-2 codebook stacking on each saved checkpoint.

---

## §1 v07-A — λ_aux sweep

### Goal

Localise the loss-weighting optimum for round-level FL training. Discriminate between the heterogeneous-label hypothesis (peak-aux is fundamentally incompatible with FL averaging) and the mis-tune hypothesis (a smaller non-zero `λ_aux` is the FL-optimal value).

### Cells

| λ_aux | Notes |
|---|---|
| 0    | Already in v06 (`*-MAEonly` namespace) — re-use, do not re-run |
| 0.05 | New                                                            |
| 0.1  | New                                                            |
| 0.2  | New                                                            |
| 0.3  | Already in v06 (default) — re-use, do not re-run               |

### Cells × algorithms × seeds

5 algorithms (FedAvg, FedProx, FedRep, Ditto, FedProto) + 1 centralised = **6 cells**. **3 new λ values** × 6 cells × 3 seeds = **54 new runs**. Together with the 36 v06 runs (λ=0 + λ=0.3) the full sweep has 90 cells.

### Hyperparameters

All v06 hyperparameters held fixed except `--aux_lambda`: `R = 20, E = 40, lr = 1e-3, batch = 512, weight_decay = 1e-5, hr_weight = 0.1`. AMP bf16 on CUDA.

### Output namespacing

v06's `_aux_suffix` already returns `-aux{V}` for non-default non-zero values, so cell directories will be:

```
outputs/v07_loss_budget_sweeps/seed{S}/V6-Dyn-A_centralised-aux0.05/
outputs/v07_loss_budget_sweeps/seed{S}/V6-Dyn-A_centralised-aux0.1/
outputs/v07_loss_budget_sweeps/seed{S}/V6-Dyn-A_centralised-aux0.2/
outputs/v07_loss_budget_sweeps/seed{S}/V6-Dyn-B-FedAvg-aux0.05/
... (90 cell dirs total)
```

(v07 sub-directory namespace; v06 outputs at `outputs/v06_round_dynamics/...` are not modified.)

### Driver

Re-use `experiments/v06_round_dynamics/{01_centralised, 02_fl_dynamics}.py` unchanged — both already accept `--aux_lambda` and emit the correct `_aux_suffix` namespace. v07 only adds an outer launcher and a v07-specific `OUTPUT_DIR_OVERRIDE` (or symlink) so v06 and v07 outputs do not collide in the same `seed{S}/` namespace.

### Expected wall-clock

Per cell wall = same as v06 (~1.6 min FedAvg → ~75 min Ditto). 54 new runs ≈ **8–14 hours** depending on FL algorithm distribution. Recommend nightly batch.

### Aggregation + figure

`07_aggregate_aux_sweep.py` (new) + `08_make_aux_sweep_figure.py` (new):

- F-aux (a): test PAPE vs `λ_aux` per algorithm — line plot, 6 algorithms, 5 λ points, mean ± std.
- F-aux (b): cross-tabulation table — algorithm × λ_aux → mean test PAPE.

### Discrimination criterion

If the optimum λ for FL cells is *strictly* 0 → hypothesis (1) is supported (peak-aux is incompatible with FL averaging). If the optimum is in (0, 0.3] → hypothesis (2) is supported (mis-tune; the FL-optimal value is just smaller than the cold-tuned 0.3).

---

## §1.5 v07-A2 — hr_weight sweep at λ_aux=0.1

### Goal

Ask whether the FL-incompatibility of `peak_aux` is concentrated in the *peak-hour CE term* (which is inherently more sensitive to per-client label distribution skew than MSE) or lies in `peak_aux` as a whole. The v06 default `hr_weight=0.1` was carried over from v01's centralised cold-tune; v07-A2 varies this parameter at the v07-A centralised optimum `λ_aux=0.1`.

### Cells

`hr_weight ∈ {0.05, 0.1, 0.5, 1.0}` × 6 algorithms (centralised + 5 FL) × 3 seeds. The default `hr=0.1` cell overlaps with v07-A's `aux0.1` results (reuse, do not re-run); only `hr ∈ {0.05, 0.5, 1.0}` × 6 × 3 = **54 new runs**. Total sweep: 72 cells (54 new + 18 overlap).

### Hyperparameters

All v06 hyperparameters held fixed except `--hr_weight`, with `--aux_lambda` fixed at `0.1`: `R = 20, E = 40, lr = 1e-3, batch = 512, weight_decay = 1e-5, λ_aux = 0.1`.

Cell suffix: `-aux0.1-hr{V}` (e.g. `V6-Dyn-B-FedAvg-aux0.1-hr0.5`).

### Driver

`experiments/v07_loss_budget_sweeps/02_run_hr_weight_sweep.py` (exists).

### Discrimination criterion

- If raising `hr_weight` (scaling peak-hour CE up) *worsens* FL cells → incompatibility is concentrated in peak-hour CE term (heterogeneous-label hypothesis, CE variant).
- If raising `hr_weight` does *not* worsen FL cells (or slightly improves) → incompatibility lies in `peak_aux` as a whole / peak-amp MSE term; simple label-distribution hypothesis is ruled out.

### Expected outcome (paper §5)

All 5 FL cells are **monotone decreasing** in `hr_weight`: `hr=1.0` beats `hr=0.1` default by −0.23 to −0.70 PAPE (small but consistent). Centralised is robust (total range ~1.0 PAPE, optimum at default `hr=0.1`). This rules out the simplest heterogeneous-CE hypothesis; the FL-incompatibility of `peak_aux` is carried at least as much by the peak-amplitude MSE term.

### Output namespacing

```
outputs/v07_loss_budget_sweeps/seed{S}/V6-Dyn-A_centralised-aux0.1-hr0.05/
outputs/v07_loss_budget_sweeps/seed{S}/V6-Dyn-B-FedAvg-aux0.1-hr0.5/
... (54 new cell dirs)
```

### Aggregation + figure

Handled by `05_aggregate_aux.py` (shared with v07-A). Figure: **F-hr** — `hr_weight` (log-x) vs test PAPE per algorithm, mean ± std band.

---

## §2 v07-B — round / local-epoch budget sweep

### Goal

Replicate the McMahan 2017 FedAvg Figure 2 setup at the v06 protocol: trade off communication rounds vs local-step compute at a fixed total epoch-equivalent budget. Establish the FedSGD upper-cost / FedAvg intermediate-cost / centralised lower-cost frontier.

### Cells

Two-axis design: `(E, R)` with constant `T = E · R = 800` epoch-equivalents (matching v06 actual: E=40 × R=20 = 800; plan originally used T=80 based on the mis-specified E=2 — see audit S1).

| Cell label                  | E (local epochs / round) | R (rounds) | T = E·R |
|----------------------------|---------------------------|-------------|----------|
| V7-FedSGD-E1-R800          | 1 SGD step (≈ E=ε)       | 800         | ~800    |
| V7-FedAvg-E1-R800          | 1                         | 800         | 800     |
| V7-FedAvg-E2-R400          | 2                         | 400         | 800     |
| V7-FedAvg-E5-R160          | 5                         | 160         | 800     |
| V7-FedAvg-E10-R80          | 10                        | 80          | 800     |
| V7-FedAvg-E20-R40          | 20                        | 40          | 800     |
| V7-FedAvg-E40-R20          | 40 (= v06 default)        | 20          | 800     |
| V7-Centralised-T800        | (n/a)                     | (n/a)       | 800 epochs |

8 cells × 3 seeds = **24 runs**.

### Note on FedSGD

`docs/fl_methodologies_fedsgd_vs_fedavg.md` defines FedSGD as 1 SGD step per round, not 1 epoch. Implementing FedSGD requires a small extension to `src/fl/round_aux.py` to support `--fedsgd_steps` (1 mini-batch per local round instead of `n_epochs`). Implementation is not large but the wall-clock is the bottleneck: 800 rounds × 114 clients × 1 mini-batch ≈ same total compute as FedAvg-E1-R800 but with 800× more communication overhead in real-network simulation.

### Output

```
outputs/v07_loss_budget_sweeps/seed{S}/V7-FedAvg-E5-R160/...
outputs/v07_loss_budget_sweeps/seed{S}/V7-FedSGD-E1-R800/...
... (24 cell dirs)
```

### Aggregation + figure

- F-budget (a): trajectory plot — round vs val PAPE (one curve per `(E, R)` cell), x-axis normalised to *total epoch equivalent* on a shared budget scale.
- F-budget (b): bytes vs val PAPE Pareto — exposes FedSGD's communication cost vs FedAvg's local-compute cost.
- F-budget (c): final test PAPE vs `E` at fixed `T = 800`, with centralised upper bound and FedSGD reference annotated.

### Expected wall-clock

Roughly proportional to v06 wall-clock × budget-scaling. FedAvg-E20-R40 ≈ 2× v06 (40 rounds), FedSGD-E1-R800 ≈ 40× v06 (800 rounds, same total compute). Total ≈ 30–60 hours; multi-day batch.

---

## §3 v07-C — round-trajectory codebook

### Goal

Test whether codebook lift grows monotonically with backbone training round. If yes → "more rounds → richer h_generic → larger lift". If plateau → "codebook lift saturates; backbone improvement is the bottleneck".

### Method

Re-run v06 Phase 1 with checkpointing modification: save `final_state_dict.pt` *every 5 rounds* (not just terminal). Then for each saved backbone, run v06 Phase 2 (`08_codebook_stacking.py`) and record the lift.

For 6 v06 cells × 3 seeds × 4 round-points {5, 10, 15, 20} = **72 codebook runs** (cheap: ~5 s each = ~6 min).

The expensive part is Phase 1 re-execution to capture intermediate checkpoints: 6 cells × 3 seeds = 18 runs at v06 wall-times = **6–24 hours**. This dominates v07-C's cost.

### Implementation

Single-line modification to `src/fl/round_logger.py` (or to each round-loop in `src/fl/round_aux.py`) — add a `checkpoint_every` arg that triggers `torch.save(global_state, cell_dir / f'state_round{r}.pt')` at every Nth round. Re-use `08_codebook_stacking.py` with a new `--backbone_checkpoint state_round{R}.pt` argument that overrides the default `final_state_dict.pt`.

### Output

```
outputs/v07_loss_budget_sweeps/seed{S}/V6-Dyn-B-FedAvg/state_round05.pt
outputs/v07_loss_budget_sweeps/seed{S}/V6-Dyn-B-FedAvg/state_round10.pt
...
outputs/v07_loss_budget_sweeps/seed{S}/V6-Dyn-B-FedAvg/codebook_lift_round05.json
outputs/v07_loss_budget_sweeps/seed{S}/V6-Dyn-B-FedAvg/codebook_lift_round10.json
...
```

### Figure

- F-traj: round vs codebook lift (ΔPAPE) — one curve per cell, 4 round-points, mean ± std band.

### Discrimination criterion

If lift is monotonic increasing in round count → backbone training is the limiting factor; v07-A (λ tuning) and v07-B (budget) compound with codebook. If lift plateaus at round 5–10 → codebook is robust to backbone partial training; codebook can be applied early in the FL session for cheaper deployment.

---

## §4 Suggested execution order (multi-day plan)

1. **Day 1** — v07-A `λ_aux` sweep launched as nightly batch (~10 hours).
2. **Day 2 morning** — v07-A aggregate + figure; decide whether v07-B and v07-C proceed in parallel or sequentially based on disk + GPU availability.
3. **Day 2 afternoon** — v07-C round-trajectory: re-run v06 Phase 1 with `--checkpoint_every 5` (one-line driver mod). 6–24 hours overnight.
4. **Day 3** — v07-C codebook stacking on saved backbones (~6 min). Aggregate
   + figure.
5. **Day 4–5** — v07-B budget sweep (~30–60 hours, the biggest single cost).
6. **Day 6** — v07 paper draft (`papers/v07_draft/`) consolidating all three axes into a single sweep paper.

Phase ordering is flexible; v07-A and v07-C can run in parallel on the same GPU if VRAM permits (each cell uses ≤300 MB), but the nightly batch recommendation is to keep them sequential for simpler post-mortem.

---

## §5 Output namespace and reproducibility

### Output root

`outputs/v07_loss_budget_sweeps/seed{S}/{cell}/...`

This directory does **not** overlap with v06's `outputs/v06_round_dynamics/seed{S}/{cell}/...`. v06 results stay frozen.

### Drivers

| Driver | Role |
|---|---|
| `experiments/v07_loss_budget_sweeps/01_run_aux_sweep.py`       | v07-A λ_aux sweep (centralised + 5 FL × 3 new λ values) |
| `experiments/v07_loss_budget_sweeps/02_run_hr_weight_sweep.py` | v07-A hr_weight sweep (fixed λ=0.1, hr ∈ {0.05,0.1,0.5,1.0}) |
| `experiments/v07_loss_budget_sweeps/05_aggregate_aux.py`       | λ_aux + hr_weight sweep aggregator → aux_sweep_summary.json |
| `experiments/v07_loss_budget_sweeps/08_make_figures.py`        | F-aux figures |
| `experiments/v07_loss_budget_sweeps/03_fedsgd.py`              | **(deferred — v07-B)** FedSGD per-round single-batch driver |
| `experiments/v07_loss_budget_sweeps/04_codebook_trajectory.py` | **(deferred — v07-C)** re-run v06 08 with `--backbone_checkpoint` |
| `experiments/v07_loss_budget_sweeps/06_aggregate_budget.py`    | **(deferred — v07-B)** budget sweep aggregator |
| `experiments/v07_loss_budget_sweeps/07_aggregate_traj.py`      | **(deferred — v07-C)** trajectory codebook aggregator |

### Tests

New pytests under `tests/` for:

- `test_v07_aux_sweep_naming.py` — verifies `_aux_suffix` produces correct cell names for each `λ_aux ∈ {0.05, 0.1, 0.2, 0.3}`.
- `test_v07_budget_argparse.py` — verifies `--rounds R --local_epochs E` combinations land in the expected output directory.
- `test_v07_fedsgd_step.py` — single-batch SGD step contract.
- `test_v07_checkpoint_roundtrip.py` — `state_round{R}.pt` save/load parity with `final_state_dict.pt`.

### Per-seed argparse

Every driver takes `--seed S` per invocation. Multi-seed sweep is the launcher's job (memory: feedback_argparse_per_seed).

---

## §6 Open questions

- (a) Should v07-B include a sweep over `clients_per_round C` (partial participation)? v06 fixes C=1.0 (full participation, 114/114 each round); partial participation is the McMahan FedAvg-paper headline axis.
- (b) Should v07-C save backbones at *every* round (not just every 5) for finer trajectory granularity? Disk cost = `~0.27 MB × 20 × 18 = ~100 MB`, manageable.
- (c) Should v07 paper include an *NF baseline* (Crossformer / NHITS / DLinear) as it is in v04? Currently scoped out — v07 is a sweep paper, not a comparison paper.
- (d) Exact FedSGD definition — McMahan 2017 §1: "1 mini-batch per communication round". Should we use full client batch (size = client's train set) or fixed mini-batch (size 512)? Full batch matches the original paper; fixed mini-batch matches v06's batch=512 invariant. Default to matching v06 (`batch=512`, full client iteration in 1 round).

---

## §7 Cross-version notes

- v07 does **not** modify CLAUDE.md invariants: backbone is frozen at NBEATSxAux + W5 hybrid (v01's design), codebook hyperparameters M=32, K_local=2, stride=24 unchanged.
- v07 does **not** introduce a new evaluation protocol — uses v06's per-client 70/10/20 split exactly.
- v07 does **not** re-tune `α_v0` on test (v01 §5.4.1 invariant). The α sensitivity ablation, if needed, lives separately in `papers/v06_draft` follow-up (v06 §5.1) and not in v07.
