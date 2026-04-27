# v03-01 — K-shot Personalized FL with frozen backbone (F2 family ablation)

> Successor to `v02-01_fl_8020_ratio.md`. Reuses v02's frozen backbone +
> codebook artifacts. Adds **cold-side K-shot fine-tuning** with a
> three-way ablation over what gets updated. Whole-backbone fine-tune
> (F1) is deferred to v04.

## Motivation

v02 establishes the zero-shot baseline at 80:20 under the PFL framing
(frozen shared encoder, federated codebook artifact). The natural next
question — anticipated by v01 §D.6 and the broader PFL literature
(FedRep, DFA, FedHiP, LoRA-style PEFT) — is:

> *Can a small amount of cold-side fine-tuning recover personalization
> while preserving the global codebook's compatibility?*

Three established PFL patterns offer different answers:

- **Head-only adaptation** (FedRep, Collins ICML'21) — encoder frozen,
  client learns personalized head only. Provably reduces sample
  complexity from Θ(d) to Θ(k).
- **Last-layer fine-tune** (DFA-style, body→head transition) — last
  generic layers are personalized.
- **Adapter / LoRA** (Hu et al., ICLR'22) — frozen base, low-rank
  adapter learned per client; rank controls capacity.

v03 evaluates all three on UMass cold gucha with K=1 month adaptation
data, and quantifies the **representation drift** each induces relative
to v02's frozen codebook.

## Goals

**G1.** Three-way F2 ablation on UMass cold gucha (n_cold=20, inherited
from v02):

| Variant | Updated parameters | h_g changes? | PFL anchor |
|---|---|---|---|
| **F2a** Head-only | AuxHead only (≈ 2K params) | No | FedRep |
| **F2b** Last-layer | Generic stack final FC + AuxHead (≈ 5K params) | Slight | DFA-style |
| **F2c** LoRA | Low-rank adapter on generic stack, rank ∈ {2, 4, 8} + AuxHead | Yes (rank-bounded) | LoRA-PEFT |

**G2.** Compare each variant to v02 zero-shot baseline → ΔPAPE, ΔHR@k.

**G3.** **Representation drift analysis.** For each F2 variant, measure:
- mean / max ‖h_g_cold − codebook[c*]‖₂ before vs after K-shot,
- whether routing decisions (c*) change after fine-tuning,
- the per-cluster distribution of cold gucha after vs before adaptation.

**G4.** Identify the trade-off frontier: parameter count (capacity) vs
PAPE gain vs codebook drift.

## Non-goals

- **F1 whole-backbone fine-tune** → v04. F1 is the BuildingsBench-style
  comparison point and worth its own paper section.
- **F3 retrieval-only personalization** (cold accumulates h_g pool, no
  learning) → optional appendix in v03 if scope allows; otherwise v04.
  Provides a "no-learning lower bound" for F2 family.
- **Backbone or codebook re-fitting** — v03 reuses v02's frozen artifacts
  exactly. The only thing that changes per cold gucha is the F2-modified
  parameters.
- **Hyperparameter re-tuning** of (σ, α_v0, α_w1) — carry over v01/v02.

## Method

### Frozen artifacts from v02

- `outputs/v02_fl_8020_ratio/seed{S}/T2/best.pt` — peak-aware backbone.
- `outputs/v02_fl_8020_ratio/seed{S}/codebook.npz` — M=32 centroids + offsets.
- v02's chosen routing (R0 or R1, decided by v02 results).

### K-shot adaptation per cold gucha

Each cold gucha's 1-year series → split:

```
Day 0  ─ 30   ─ 37        ─ 365
        ↑       ↑           ↑
       K-shot  buffer      eval
       train   (ignore)
```

- **K = 1 month ≈ 720 h ≈ 30 windows** (stride=24).
- 1-week buffer to avoid leakage from K-shot adaptation into the first
  evaluated forecast.
- Eval segment ≈ 11 months → ~330 windows per cold gucha → ~6,600 cold
  windows total (n_cold=20).

### Adaptation loss

Same as v02's training loss applied locally per cold gucha:

```
ℓ = MAE(ŷ, y) + λ · ℓ_aux,    λ = 0.3
```

with the cold gucha's own K-shot windows providing labels.

### Per-variant adaptation

| Variant | Trainable | Optimizer | Epochs |
|---|---|---|---|
| F2a | `aux_head.*` | Adam, lr=1e-3 | 10–30, early stop on K-shot val (last ~5 windows) |
| F2b | `aux_head.*` + `backbone.stack_generic.fc4.*` + `proj.*` | Adam, lr=5e-4 | 10–30, early stop |
| F2c | `aux_head.*` + LoRA adapters on generic stack FCs (rank ∈ {2, 4, 8}) | Adam, lr=1e-3 | 10–30, early stop |

LoRA implementation: wrap `fc1..fc4` in `stack_generic` with
`y = W₀ x + (W₀ + B A) x` style low-rank delta where A ∈ ℝ^{r×d},
B ∈ ℝ^{d×r}, r ∈ {2, 4, 8}. Only A, B, AuxHead are trainable.

### Routing during evaluation

Each cold gucha at eval time:
1. Forward through *adapted* backbone → `h_g_cold` (now F2-tuned).
2. Look up codebook (frozen, from v02). For F2a this still uses v02's
   exact `h_g_cold` distribution; for F2b/F2c the distribution drifts.
3. Apply v01 W5 hybrid correction with v02's chosen operating point.

### Comparison arms

```
Baseline:     v02-zero-shot (no adaptation)              [from v02 outputs]
v03-F2a:      head-only                                  [main result]
v03-F2b:      last-layer (generic stack final FC)        [main result]
v03-F2c-r2:   LoRA rank=2                                [main result]
v03-F2c-r4:   LoRA rank=4                                [main result]
v03-F2c-r8:   LoRA rank=8                                [main result]
(optional) v03-F3: retrieval-only, no adaptation         [appendix]
```

## Experimental plan

### Splits and seeds

Inherits 80:20 from v02. Same 3 seeds {42, 123, 7}. No new
stratification.

### Run order

For each seed × cold gucha (i ∈ 1..20) × F2 variant:
1. Load v02's frozen `T2/best.pt` and codebook.
2. Apply F2-specific trainable mask.
3. K-shot adapt on cold gucha's first 30 days (with 5-window val for
   early stop).
4. Frozen-codebook eval on cold gucha's day-38..end → per-window PAPE,
   HR@1, HR@2.

### Drift analysis

For each variant, log per-cold-gucha:
- mean / std / max of `‖h_g_cold(after) − h_g_cold(before)‖₂`,
- agreement rate of routing decision `c*(after) == c*(before)`,
- which clusters gain / lose cold windows after adaptation.

## Outputs (target paths)

```
outputs/v03_kshot_pfl/
├── seed{42,123,7}/
│   └── apt{i}/
│       ├── F2a_head/
│       │   ├── adapted.pt              # cold-specific aux head weights only
│       │   └── results.json            # PAPE/HR per window
│       ├── F2b_lastFC/
│       │   ├── adapted.pt
│       │   └── results.json
│       └── F2c_lora_r{2,4,8}/
│           ├── adapted.pt              # LoRA AB matrices + aux head
│           └── results.json
├── ablation_summary.json               # mean ± std across seeds × cold gucha × variant
├── drift_analysis.json                 # h_g shift, routing agreement, per-cluster shifts
├── pareto_capacity_vs_pape.csv         # param count vs ΔPAPE vs drift
└── FINAL_v03_report.md
```

## Deliverables

1. **`papers/v03_kshot_pfl.md`** — short report (8–12 pages).
   - § 1: motivation + PFL anchors (FedRep, DFA, LoRA).
   - § 2: F2 variant definitions + adaptation protocol.
   - § 3: main result table {F2a, F2b, F2c} vs v02 zero-shot.
   - § 4: representation drift analysis.
   - § 5: Pareto trade-off (capacity vs PAPE vs drift).
   - § 6: discussion — when does cold-side fine-tuning pay off?
   - § 7: threats — K=1 month is a single point; sensitivity to K is
     v04 work.

2. **`papers/v03_kshot_pfl.tex`** — IEEE-style.

3. **`papers/figures/v03_F*.png`**:
   - F1: F2 variants Pareto (param count × ΔPAPE).
   - F2: representation drift histogram (per variant).
   - F3: per-cluster cold benefit, before vs after adaptation.
   - F4: routing agreement rate (`c* agreement` vs variant).

## Open questions

1. **K-shot hyperparameters per variant.** Optimizer LR and epoch budget
   may differ across F2a/b/c. Use a small validation sweep on a
   *training* gucha (held-out from v02 train) to set defaults — no cold
   tuning.

2. **LoRA rank sweep granularity.** Three ranks {2, 4, 8} is the minimum
   meaningful sweep. If results show rank-monotonicity, can be expanded
   in v04.

3. **Buffer-week necessity.** 1-week buffer between K-shot and eval
   prevents the first eval window from overlapping with adaptation
   data. Verify in dry-run; if the input window L=96 already prevents
   leakage, buffer can be removed.

4. **F3 retrieval-only inclusion.** Provides "no learning" lower bound
   that strengthens F2 claims. Adds ~1 day. Recommend including as
   appendix; user decides during v03 kickoff.

5. **Drift mitigation for F2b/F2c.** If drift is large, consider adding
   a regularizer that pulls adapted h_g toward original distribution
   (e.g., KL or L2 to original `h_g_cold`). Defer to v04 unless drift
   collapses routing in v03 dry-run.

## Dependencies

- **v02 outputs required**: `outputs/v02_fl_8020_ratio/seed*/T2/best.pt`
  and `codebook.npz` must exist. v03 cannot start before v02 completes.
- v01 codebase reused with additions:
  - `src/models/lora.py` (new) — LoRA wrapper for `nn.Linear`.
  - `src/models/nbeatsx_aux_lora.py` (new, optional) — convenience
    wrapper that applies LoRA to `MinimalNBEATSx.stack_generic`.

## Status

- 2026-04-27 — folder structure created, plan drafted.
- Blocked on: v02 results.
- Next after v02 done: implement `src/models/lora.py`, then port
  `experiments/v03_kshot_pfl/01_kshot_adapt.py` per F2 variant.
