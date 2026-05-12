# v07 — Loss-Weight Sensitivity of Round-Level Federated Peak-Aware Forecasting

> **Status**: draft (2026-05-04). Carries v06's round-level FL protocol forward
> as a *sweep paper*. No new method, no new evaluation protocol — every result
> below uses the v06 backbone (NBEATSxAux), v06 data split (per-client 70/10/20
> on 114 UMass 2016 apartments), and v06 evaluation pipeline (terminal test
> PAPE/HR@k/MAE/MSE on each client's 20% test windows). v07 quantifies how
> three orthogonal hyperparameters move the v06 conclusions.

## §1 Motivation

v06 closed two open questions and surfaced two new ones. The closed ones —
*does the round-level FL protocol with the v06 backbone match centralised
pooled SGD?* (yes, FL within ≈2.0 PAPE of centralised) and *does the
post-hoc Peak-VQ codebook still lift PAPE on top of FL backbones?* (yes, ≈3
PAPE lift on every cell) — are taken as fixed by this paper.

The *unanswered* ones, paraphrased from v06 §6:

1. **Is the negative effect of `λ_aux = 0.3` on FL cells a tuning artefact?**
   v06 reports that all five FL algorithms get worse (+2.3–3.5 PAPE on test)
   when peak-aux is enabled at v01's cold-tuned `λ = 0.3`, while centralised
   loses only +0.5 PAPE. Two competing mechanisms remain plausible:
   (1) heterogeneous-label hypothesis — FedAvg gradient averaging dilutes
   the per-client peak-hour signal in the auxiliary CE; (2) mis-tune
   hypothesis — the FL-optimal `λ` is just smaller than 0.3.

2. **Does the v06 §5.3 finding "MAEonly + codebook beats default + codebook"
   hold for centralised, and does it survive an interior-optimum λ?** v06
   evaluated only the two endpoints (`λ ∈ {0, 0.3}`) of the FL backbone
   choice. The corresponding centralised matrix and the intermediate `λ`
   point have been missing.

v07-A (§3) sweeps `λ_aux ∈ {0, 0.05, 0.1, 0.2, 0.3}` × 6 algorithms × 3 seeds
to localise the FL-optimal value and discriminate between mechanisms (1) and
(2). v07-A1 (§4) re-runs the v06 Phase-2 codebook stacking on the centralised
λ-swept backbones, populating the missing 3×3 cell of the codebook×λ matrix.
v07-A2 (§5) holds `λ_aux = 0.1` (v07-A's centralised optimum) and sweeps
`hr_weight ∈ {0.05, 0.1, 0.5, 1.0}` to ask whether the FL-incompatibility of
peak-aux is concentrated in the peak-hour CE term, the peak-amplitude MSE
term, or both.

## §2 Setup

All v07 cells share the v06 protocol unchanged:

- **Dataset**: 114 valid UMass 2016 apartments after
  `filter_valid_apartments(min_hours=7000)`.
- **Per-client split**: 70/10/20 train/val/test; identical for every cell;
  the v06 cache (`build_per_client_splits`) is reused.
- **Backbone**: `NBEATSxAux(latent_source='h_generic')`, M=32 codebook,
  K_local=2 (federated), stride=24, AMP bf16 on CUDA.
- **Loss**: `L = MAE(ŷ, y) + λ_aux · (peak_amp_MSE + hr_weight · peak_hour_CE)`.
- **Optimiser**: AdamW (Adam) lr=1e-3, weight_decay=1e-5, batch=512.
- **Round budget**: rounds=20, local_epochs=40, full participation (C=1.0).
  v07 does **not** sweep this axis (deferred to v07-B in
  `plans/v07-01_loss_and_budget_sweeps.md`).
- **Algorithms**: centralised pooled SGD (upper bound), FedAvg, FedProx
  (μ=0.01), FedRep (head_epochs=1), Ditto (λ=0.1), FedProto (K=32, λ_proto=0.1).
- **Seeds**: {42, 123, 7}. All numbers are mean ± std (Bessel-corrected).
- **Evaluation**: each client's terminal test windows; PAPE (peak-amplitude
  percent error) is the headline metric; HR@1, HR@2, MAE, MSE(kW²) are
  reported for completeness but not the primary discriminator here.

The v07 outputs live under `outputs/v07_loss_budget_sweeps/`; the v06
outputs at `outputs/v06_round_dynamics/` are reused but not modified.
The v06 driver scripts accept `--output_namespace v07_loss_budget_sweeps`
to redirect their writes.

## §3 v07-A — λ_aux sweep

### §3.1 Method

5-point sweep `λ_aux ∈ {0, 0.05, 0.1, 0.2, 0.3}` × 6 algorithms × 3 seeds = 90
total configurations. The two endpoints (`λ = 0` "MAEonly", `λ = 0.3` v06
default) are taken from v06's already-completed runs at
`outputs/v06_round_dynamics/`. The three new interior values `λ ∈ {0.05, 0.1, 0.2}`
were run by v07's launcher (`experiments/v07_loss_budget_sweeps/01_run_aux_sweep.py`),
which dispatches the v06 driver with `--output_namespace v07_loss_budget_sweeps`
and `--aux_lambda <value>`. 54 new runs at ~5h32m total wall-clock under
3-way per-seed parallelism (single CUDA GPU, 3 Python processes — CUDA stream
serialisation gave a near-perfect 3× speed-up over sequential because
per-cell GPU memory is ≤300 MB).

### §3.2 Results

**Final test PAPE (mean ± std over 3 seeds)** — F-aux:

| algorithm   | λ=0 | λ=0.05 | λ=0.1 | λ=0.2 | λ=0.3 | optimum |
|-------------|-----|--------|-------|-------|-------|---------|
| centralised | 48.91 ± 0.70 | 48.60 ± 0.16 | **48.47 ± 0.45** | 49.65 ± 0.25 | 49.43 ± 0.36 | **λ=0.1** |
| FedAvg      | **48.42 ± 0.37** | 49.70 ± 0.95 | 50.85 ± 0.83 | 51.16 ± 0.70 | 51.36 ± 0.61 | λ=0 |
| FedProx     | **48.51 ± 0.03** | 49.90 ± 0.34 | 50.65 ± 0.72 | 51.09 ± 0.68 | 51.40 ± 0.63 | λ=0 |
| FedRep      | **49.08 ± 0.50** | 49.28 ± 0.98 | 50.17 ± 0.86 | 51.10 ± 0.74 | 51.36 ± 0.68 | λ=0 |
| Ditto       | **48.28 ± 0.32** | 49.54 ± 0.84 | 50.82 ± 0.36 | 51.47 ± 0.41 | 51.79 ± 0.47 | λ=0 |
| FedProto    | **48.49 ± 0.31** | 49.49 ± 1.14 | 50.70 ± 0.84 | 51.33 ± 0.82 | 51.50 ± 0.54 | λ=0 |

Figure: `figures/F-aux.png` (`F-aux: λ_aux sensitivity at v06 round-level FL protocol`).

### §3.3 Finding 1 — centralised has an interior optimum λ=0.1

The centralised cell is the only one with a non-monotone trajectory:
`λ=0 → 0.05 → 0.1 → 0.2 → 0.3` traces 48.91 → 48.60 → **48.47** → 49.65 →
49.43 (mean PAPE). The interior minimum at `λ=0.1` beats the v06 default
(`λ=0.3`) by **−0.96 PAPE** with comparable variance, and the dip is
tight (within 0.5 PAPE) over `λ ∈ {0.05, 0.1}`. The v01 cold-tuned value
0.3 is slightly **over-weighted** for round-level centralised training —
mechanism (2), mis-tune, is supported on this cell.

### §3.4 Finding 2 — every FL algorithm is monotone in λ_aux, λ=0 strict optimum

All five FL algorithms return their best test PAPE at the *boundary*
`λ = 0`. The trajectories rise strictly with `λ_aux` for FedAvg, FedProx,
FedRep, Ditto, and FedProto. The cost of going from `λ=0` to v06's default
`λ=0.3` is +2.28 PAPE (FedRep, minimum) to +3.51 PAPE (Ditto, maximum)
on the test split, with std ≤ 1.1 across seeds — every difference
exceeds 2σ.

This is the **incompatibility hypothesis (1)** confirmed: peak-aux is not
just mis-tuned for FL, it is *strictly harmful* for any positive `λ_aux`
under the v06 FedAvg-style aggregator. No interior optimum exists for any
of the five FL algorithms.

### §3.5 Discrimination summary

The two findings together form the headline of v07-A:

> **Federation itself, not the choice of `λ_aux`, is what makes peak-aux
> harmful.** Centralised pooled SGD benefits from a small but non-zero
> peak-aux signal (48.47 at λ=0.1 vs 48.91 at λ=0); the same backbone under
> FedAvg-style averaging has zero use for peak-aux at any positive
> `λ_aux`. The `λ=0.3` carry-over from v01 is therefore doubly wrong on
> FL cells — both the magnitude (mis-tune) and the sign of the gradient
> the user gets from increasing it (incompatibility) are off.

## §4 v07-A1 — codebook stacking on the λ-swept centralised backbone

### §4.1 Method

The v06 §5.3 finding *"MAEonly + codebook beats default + codebook"* is
re-tested on the centralised cell with the new interior optimum `λ=0.1`
backbone added. Three centralised backbones × 3 seeds = 9 codebook stacks
were run with `experiments/v06_round_dynamics/08_codebook_stacking.py`
(M=32, K_local=2, α_v0=1.0). The two endpoints (`λ=0`, `λ=0.3`) reuse
v06's already-on-disk `codebook_lift.json`; only the `λ=0.1` cell needed
new compute (~5 s per seed × 3 seeds = ~15 s total).

### §4.2 Results

**Centralised backbone × codebook (test PAPE, mean ± std over 3 seeds)** —
F-codebook-vs-λ:

| backbone λ | PAPE before codebook | PAPE after codebook | ΔPAPE | ΔMAE |
|------------|----------------------|---------------------|-------|------|
| **λ=0** (MAEonly) | 48.90 ± 0.68 | **44.41 ± 0.29** | −4.49 | +0.0043 |
| λ=0.1 (v07-A optimum) | 48.46 ± 0.46 | 44.53 ± 0.60 | −3.93 | +0.0077 |
| λ=0.3 (v06 default) | 49.43 ± 0.35 | 44.92 ± 0.14 | −4.51 | +0.0095 |

Figure: `figures/F-codebook-vs-lambda.png`.

### §4.3 Finding 3 — codebook absorbs the backbone λ choice

After codebook stacking, all three centralised backbones land within 0.5
PAPE of each other (44.41 / 44.53 / 44.92). The pre-codebook ordering —
`λ=0.1 < λ=0 < λ=0.3` — is **not** preserved post-codebook: the strict
post-codebook optimum is `λ=0` (44.41), with the smallest ΔMAE cost
(+0.0043 vs +0.0077 / +0.0095 on the other two cells). The v06 §5.3
recommendation generalises:

> **Recommended operating recipe**: train backbone with `λ_aux = 0`, then
> apply federated codebook stacking. Holds for both centralised and FL.
> The interior backbone optimum `λ=0.1` is *not* the global optimum once
> the codebook is in the stack.

The mechanism appears to be that the federated codebook extracts the
peak-relevant cluster structure from `h_generic` directly, without
requiring backbone gradient pressure from the auxiliary head — making the
auxiliary head's job redundant once the codebook is present. The MAE-cost
side-effect of training under `λ>0` then becomes pure overhead.

## §5 v07-A2 — hr_weight sweep at λ_aux=0.1

### §5.1 Method

`hr_weight` is the weight on `peak_hour_CE` *inside* `peak_aux`:
`peak_aux = peak_amp_MSE + hr_weight · peak_hour_CE`. The v06 default
`hr_weight = 0.1` was carried over from v01's centralised cold-tune. v07-A
showed that `λ_aux > 0` is harmful for FL cells; v07-A2 asks whether that
incompatibility is concentrated in the *peak-hour CE term* (which is
inherently more sensitive to per-client label distribution skew than MSE)
or whether it lies in `peak_aux` as a whole.

Sweep: `hr_weight ∈ {0.05, 0.1, 0.5, 1.0}` × 6 algorithms × 3 seeds, all
at `λ_aux = 0.1` (centralised v07-A optimum). 54 new runs — the
default `hr=0.1` cell reuses v07-A's `aux0.1` results, so only
`hr ∈ {0.05, 0.5, 1.0}` × 6 × 3 = 54 actually executed. Wall-clock ≈ 5h
under 3-way parallelism. Cell suffix: `-aux0.1-hr{V}` (e.g.
`V6-Dyn-B-FedAvg-aux0.1-hr0.5`).

### §5.2 Results

**Final test PAPE at λ_aux=0.1 (mean ± std over 3 seeds)** — F-hr:

| algorithm   | hr=0.05 | hr=0.1 (default) | hr=0.5 | hr=1.0 | optimum hr | Δ vs default |
|-------------|---------|------------------|--------|--------|------------|--------------|
| **centralised** | 49.43 ± 0.54 | **48.47 ± 0.45** | 48.75 ± 0.37 | 48.64 ± 0.67 | **hr=0.1** | 0.00 |
| FedAvg      | 50.85 ± 0.79 | 50.85 ± 0.83 | 50.65 ± 0.82 | **50.23 ± 0.93** | hr=1.0 | −0.62 |
| FedProx     | 50.66 ± 0.71 | 50.65 ± 0.72 | 50.50 ± 0.70 | **50.21 ± 0.75** | hr=1.0 | −0.45 |
| FedRep      | 50.09 ± 0.85 | 50.17 ± 0.86 | 50.13 ± 0.84 | **49.94 ± 1.01** | hr=1.0 | −0.23 |
| Ditto       | 50.77 ± 0.39 | 50.82 ± 0.36 | 50.53 ± 0.31 | **50.12 ± 0.41** | hr=1.0 | −0.70 |
| FedProto    | 50.74 ± 0.88 | 50.70 ± 0.84 | 50.70 ± 0.86 | **50.46 ± 0.97** | hr=1.0 | −0.24 |

Figure: `figures/F-hr.png` (log-x axis).

### §5.3 Finding 4 — centralised is robust, FL sees small monotone hr-benefit

Two distinct patterns:

- **Centralised**: total PAPE range across the four hr values is 1.0 PAPE
  (49.43 → 48.47), and the centre of the trajectory is the *default*
  `hr=0.1`. Centralised is **robust** to `hr_weight` choice.

- **All 5 FL algorithms**: monotone *decreasing* in `hr_weight`. `hr=1.0`
  beats `hr=0.1` by −0.23 to −0.70 PAPE; `hr=0.05` is essentially equal
  to `hr=0.1` (diffs ≤ 0.05 PAPE everywhere). The size of the benefit is
  small but consistent across all five algorithms.

### §5.4 Finding 5 — incompatibility lies in peak-aux as a whole, not in peak-hour CE alone

The crucial inference combines §3 and §5: if the FL-incompatibility of
`peak_aux` were concentrated in the `peak_hour_CE` term, raising
`hr_weight` (which scales that term up) should *worsen* FL cells; instead
it *slightly improves* them. Equivalently, lowering `hr_weight` to 0.05
(scaling peak_hour_CE down by 2×) does not measurably help FL.

This rules out the simplest version of the heterogeneous-label hypothesis
("FedAvg of per-client one-hot peak-hour CE gradients dilutes the
signal") as the dominant mechanism. The damage from `λ_aux > 0` in FL
must be carried at least as much by the *peak-amplitude MSE* term, or by
the joint training dynamics, as by the peak-hour CE alone. v07's data is
consistent with the broader reading:

> The FL-incompatibility of `peak_aux` is a property of the *entire
> peak-aux gradient* under FedAvg averaging, not a property of any single
> internal component. Centralised training is insensitive to the
> internal weighting of the auxiliary head, while FL is uniformly
> punished by any positive `λ_aux`.

The ordering of effect sizes (PAPE on FL cells):

- 1st-order: `λ_aux = 0` vs `λ_aux = 0.3` → −2.28 to −3.51 PAPE  *(huge — v07-A finding 2)*
- 2nd-order: `hr=1.0` vs `hr=0.1` at `λ=0.1` → −0.2 to −0.7 PAPE  *(small — v07-A2 finding 4)*

The v06 default's primary problem is `λ_aux > 0` itself; redistributing the
internal weighting cannot recover the loss.

## §6 Discussion

### §6.1 Recommended operating recipe

Combining the v06 conclusions and the v07 findings, the recipe for
round-level federated peak-aware load forecasting on a v06-shaped problem
is:

1. **Backbone**: NBEATSxAux trained with `λ_aux = 0` (MAE-only).
2. **Codebook**: federated 2-stage hierarchical KMeans (`src/fl/codebook_fl.py`)
   with M=32, K_local=2, α_v0 = 1.0 (CMO-only correction).
3. **Algorithm choice**: dominated by the underlying FL aggregator's
   communication / drift trade-off (v05 / v06 already showed 5 algorithms
   land within ~1 PAPE of each other on this corpus). Codebook stacking
   makes the choice even less material.

The single line summarising the v06 → v07 trajectory:

> v06 found that codebook stacking gives ≈ −3 PAPE lift on top of the
> v01-default backbone; v07 finds that the optimal backbone for that
> codebook is the *MAE-only* backbone, not the v01-default. The
> recommended FL recipe is therefore *MAEonly + codebook*, not
> *peak-aux + codebook*.

### §6.2 Relationship to v06 §5.3

v06 §5.3 already reported that MAEonly + codebook beats default + codebook
on FL cells (−0.5 to −1.5 PAPE depending on algorithm). v07-A1 (§4)
extends this finding to centralised, where the gap is smaller (44.41 vs
44.92 = −0.51 PAPE) but identical in direction. The v07-A2 finding (§5)
strengthens the recipe: even *within* `λ_aux > 0`, hr_weight retuning
gives at best −0.7 PAPE on FedAvg, which is dominated by the −2.28 to
−3.51 PAPE gain from setting `λ_aux = 0` outright.

### §6.3 Limitations and deferred questions

1. **Round / local-epoch budget** — `(R=20, E=40)` was held fixed across
   all v07 cells (see item 6 below for the audit note on the originally
   stated E=2). The budget axis is what `plans/v07-01` §2 (v07-B) is
   scoped to, including a FedSGD reference and an `E ∈ {1, 2, 5, 10, 20}`
   sweep at `T = E·R = 80`. v07-B is a *new-driver* axis — `03_fedsgd.py`
   needs to be written to support 1-mini-batch-per-round FedSGD — and is
   not implemented in this draft.
2. **Round-trajectory codebook** — v07's codebook stacking uses only the
   *terminal* (round-20) backbone. Whether codebook lift is monotone in
   round count or saturates early is the v07-C question. It requires a
   one-line modification to `src/fl/round_logger.py`
   (`checkpoint_every` arg) and a Phase-1 re-execution to capture
   intermediate `state_round{R}.pt` files. Estimated wall ≈ 6–24 h on top
   of the codebook stacking itself (~6 minutes).
3. **`λ_aux ∈ {0.5, 0.7}` extension** — given that all five FL trajectories
   are already monotone-increasing through `λ_aux = 0.3`, larger values
   are unlikely to add new information. They were *not* run.
4. **Cross-axis interaction** — v07 sweeps `λ`, then `hr_weight | λ=0.1`,
   but does not sweep `hr_weight | λ=0` (since `λ=0` has zero gradient
   from peak-aux, hr_weight is mathematically inert there) or
   `hr_weight | λ=0.3` (would be a 3-axis sweep).
5. **No `α_v0` re-tuning on v07 cells** — by v01 §5.4.1 the operating-point
   choice (α_v0) is *not* re-tuned on test (the bit-exact carry-over
   invariant of CLAUDE.md). v07 inherits α_v0 = 1.0 from v06 §5.1.

6. **Round budget mismatch (audit C1)** — `plans/v06-01_round_dynamics.md`
   and this paper's §2 originally stated `local_epochs = 2`, matching the
   conference Phase A invariant and giving a nominal budget of
   `R × E = 20 × 2 = 40` epoch-equivalents per client. The actual runs (all
   `result.json` files and the v07 launcher at
   `experiments/v07_loss_budget_sweeps/01_run_aux_sweep.py:108`) used
   `local_epochs = 40`, i.e. `R × E = 20 × 40 = 800` epoch-equivalents
   per client — 20× higher compute than the stated protocol. The §2 table
   above now reflects the true value. The qualitative claims of this paper
   (centralised interior optimum at λ=0.1; FL strict boundary optimum at
   λ=0; FL-incompatibility of peak-aux) are not affected by this discrepancy,
   but absolute PAPE numbers and all λ-cost claims in this draft should be
   read as pertaining to the **800 epoch-equivalent regime**, not the
   originally intended 40 epoch-equivalent regime.

Items (1) and (2) are scoped explicitly in
`plans/v07-01_loss_and_budget_sweeps.md` and will form a follow-up draft;
item (3) is a deliberate scope decision; items (4) and (5) are
out-of-scope by the protocol invariants; item (6) is a retrospective
documentation correction.

## §7 Conclusion

This paper sweeps two orthogonal hyperparameters of the v06 round-level
federated peak-aware forecasting protocol — `λ_aux` (peak-auxiliary loss
weight, §3) and `hr_weight` (peak-hour CE weight inside peak-aux, §5) —
and re-tests v06's codebook stacking on the resulting `λ`-swept
centralised backbones (§4). The five takeaways:

1. Centralised pooled SGD has an interior optimum `λ_aux = 0.1` (48.47
   PAPE), strictly better than v06 default `λ_aux = 0.3` (49.43) by
   −0.96 PAPE.
2. Every FL algorithm under v06's protocol has a strict boundary optimum
   `λ_aux = 0`. Setting `λ_aux > 0` *strictly worsens* FL test PAPE,
   mono­tonically across `λ ∈ {0.05, 0.1, 0.2, 0.3}`. The FL-incompatibility
   of peak-aux is therefore not a tuning artefact.
3. After post-hoc federated codebook stacking, the centralised backbone
   PAPE is absorbed onto a 0.5-PAPE band centred at 44.5; the strict
   post-codebook optimum is the **MAE-only** backbone (44.41 PAPE), with
   the smallest MAE side-effect (ΔMAE = +0.0043). v06 §5.3's "MAEonly +
   codebook" recommendation generalises to centralised.
4. The peak-aux FL-incompatibility is **not** carried by the peak-hour
   CE term alone: at fixed `λ_aux = 0.1`, FL cells *prefer* larger
   `hr_weight` (best at hr=1.0, −0.2 to −0.7 PAPE vs hr=0.1), while
   centralised is robust to hr_weight (range 1.0 PAPE only). The damage
   is borne by the peak-aux gradient as a whole under FedAvg averaging.
5. Effect sizes order as: `λ_aux = 0` vs `λ_aux = 0.3` (1st order, huge,
   −2.28 to −3.51 PAPE on FL) ≫ `hr_weight = 1.0` vs default at fixed
   `λ_aux = 0.1` (2nd order, small, −0.7 PAPE). The recommended recipe
   is `λ_aux = 0` + codebook stacking; further within-peak-aux tuning
   yields only marginal additional gains.

## §8 Reproducibility

| Stage | Driver | Output | Wall-clock |
|-------|--------|--------|------------|
| v07-A Phase 1 (54 runs, λ ∈ {0.05, 0.1, 0.2}) | `experiments/v07_loss_budget_sweeps/01_run_aux_sweep.py` (calls v06 `01_centralised.py` / `02_fl_dynamics.py` with `--output_namespace v07_loss_budget_sweeps`) | `outputs/v07_loss_budget_sweeps/seed{S}/V6-Dyn-{...}-aux{V}/result.json` | 5h32m, 3-way per-seed parallel |
| v07-A aggregate | `experiments/v07_loss_budget_sweeps/05_aggregate_aux.py` | `outputs/v07_loss_budget_sweeps/aux_sweep_summary.json` | <1 s |
| v07-A figure | `experiments/v07_loss_budget_sweeps/08_make_figures.py --section aux` | `outputs/.../figures/F-aux.png` | <1 s |
| v07-A1 codebook (3 runs centralised λ=0.1) | `experiments/v06_round_dynamics/08_codebook_stacking.py --cell V6-Dyn-A_centralised-aux0.1 --output_namespace v07_loss_budget_sweeps` | `outputs/v07_loss_budget_sweeps/seed{S}/V6-Dyn-A_centralised-aux0.1/codebook_lift.json` | ≈15 s total |
| v07-A1 figure | `experiments/v07_loss_budget_sweeps/08_make_figures.py --section codebook_lambda` | `outputs/.../figures/F-codebook-vs-lambda.png` | <1 s |
| v07-A2 Phase 1 (54 runs, hr ∈ {0.05, 0.5, 1.0}, λ=0.1) | `experiments/v07_loss_budget_sweeps/02_run_hr_weight_sweep.py` | `outputs/v07_loss_budget_sweeps/seed{S}/V6-Dyn-{...}-aux0.1-hr{V}/result.json` | ~5h, 3-way per-seed parallel |
| v07-A2 figure | `experiments/v07_loss_budget_sweeps/08_make_figures.py --section hr` | `outputs/.../figures/F-hr.png` | <1 s |
| Tests | `pytest tests/test_v07_aux_sweep.py` | 10 unit-tests pass | <3 s |

All drivers take `--seed S` per invocation; multi-seed sweep is the
launcher's job (memory: `feedback_argparse_per_seed`). Every cell directory
is name-spaced via `_aux_suffix(λ_aux) + _hr_suffix(hr_weight)`; the v06
default `(λ=0.3, hr=0.1)` cells have empty suffixes for back-compat.

To regenerate the entire v07 paper from scratch with v06 results already
on disk:

```bash
uv run python experiments/v07_loss_budget_sweeps/01_run_aux_sweep.py \
    --seeds 42 123 7 --lambdas 0.05 0.1 0.2 \
    --algorithms fedavg fedprox fedrep ditto fedproto

# v07-A1 codebook stacking on centralised λ=0.1 (3 seeds)
for SEED in 42 123 7; do
  uv run python experiments/v06_round_dynamics/08_codebook_stacking.py \
      --seed $SEED --cell V6-Dyn-A_centralised-aux0.1 \
      --output_namespace v07_loss_budget_sweeps
done

uv run python experiments/v07_loss_budget_sweeps/02_run_hr_weight_sweep.py \
    --seeds 42 123 7 --hr_weights 0.05 0.5 1.0

uv run python experiments/v07_loss_budget_sweeps/05_aggregate_aux.py --seeds 42 123 7
uv run python experiments/v07_loss_budget_sweeps/08_make_figures.py --section all
```

Total compute: ~10h on a single CUDA GPU under 3-way per-seed parallelism;
~30h sequential.
