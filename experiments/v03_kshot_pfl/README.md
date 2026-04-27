# v03 — K-shot Personalized FL with frozen backbone (F2 family)

Successor to `experiments/v02_fl_8020_ratio/`. v02 establishes a
zero-shot baseline; v03 adds **cold-side K-shot adaptation** under three
PFL patterns:

- **F2a** Head-only fine-tune (FedRep pattern)
- **F2b** Last-layer fine-tune (DFA-style)
- **F2c** LoRA adapter on generic stack (rank ∈ {2, 4, 8})

Backbone and codebook are inherited frozen from v02. K=1 month per cold
gucha. See `plans/v03-01_kshot_pfl.md` for the full plan.

## Status

Scaffolding only. Blocked on v02 completion.

## Planned script order

| # | Script | Purpose |
|---|---|---|
| 01 | `01_kshot_adapt.py` | Per (seed, cold gucha, F2 variant): load v02 frozen artifacts, K-shot adapt, save adapted weights. |
| 02 | `02_eval_adapted.py` | Per cold gucha × variant: forward through adapted backbone + frozen codebook lookup, compute PAPE/HR@k. |
| 03 | `03_drift_analysis.py` | Per variant: representation drift, routing agreement, per-cluster shift. |
| 04 | `04_aggregate_ablation.py` | Cross-seed × variant aggregation, Pareto table. |
| 05 | `05_make_v03_figures.py` | Generate F1–F4 figures. |
| (opt) | `06_F3_retrieval_only.py` | Optional: no-learning retrieval baseline as F2 lower bound. |

## Key delta from v02

- **Cold-side fine-tuning** with K=1 month of cold gucha history.
- **Three trainable-parameter variants** (F2a/b/c) sharing the same
  frozen v02 backbone + codebook.
- **Drift instrumentation** — quantify how much h_g_cold moves from
  v02's centroid neighborhood and whether routing decisions change.

## Outputs

`outputs/v03_kshot_pfl/` — see plan §"Outputs" for tree.

## What is NOT in scope

- F1 whole-backbone fine-tune → **v04**.
- Federated K-shot (each gucha shares its adapted delta with others) → v04+.
- K sensitivity sweep (K ∈ {1 week, 2 week, 1 month, 3 months}) → v04.
- Codebook re-fit on adapted h_g → v04+ (would defeat the frozen-artifact framing).
- Backbone redesign — method frozen at v01's NBEATSx + W5 hybrid.

## Dependencies

`outputs/v02_fl_8020_ratio/seed{42,123,7}/T2/best.pt` and
`codebook.npz` must exist before v03 starts.
