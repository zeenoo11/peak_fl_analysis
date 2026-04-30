# v05 FedCB Codebook — Experiment Log

Chronological audit trail: SEND-BACK, patch resolution, integrity re-review, and all run stages.

---

## Entry 0: Initial SEND-BACK (2026-04-29)

**Date:** 2026-04-29
**Verdict:** SEND-BACK — 2 findings, sweep NOT started.

Executor integrity review of the first code delivery identified two linked plan-code divergences:

**Finding 1 (blocking):** `_cold_eval_cmo()` in `01_fedcb_codebook.py` used `route_R1` for the
`--mode centralised` path. Plan §Experimental matrix explicitly states V5-FedCB-0 uses R0 routing
(KEY pool 1-NN) to match the v02 §B.3 published anchor (44.18 ± 0.18). Using R1 in centralised
mode would make Gate 1 a different experiment, not a reproduction.

**Finding 2 (linked):** `02_aggregate.py` did not read `outputs/v02_fl_8020_ratio/seed{S}/W_component_results.json`.
The gate comparison was therefore impossible without re-running centralised eval.

All other items PASSed: pytest 4 green, `strict=True` loads, `model.eval()` + `no_grad`,
stride=24, seed CLI, output namespace, no MLflow (correct per repo convention), communication
formula matches plan.

Sent back to engineer. Sweep blocked until patch delivery.

---

## Entry 1: Patch Delivery and Integrity Re-Review (2026-04-29)

**Date:** 2026-04-29
**Verdict:** PASS — both findings resolved.

Engineer delivered patch with the following confirmed changes:

**Finding 1 resolved:** `--mode centralised` argument removed entirely from argparse.
No `if mode == "centralised"` branches remain anywhere in `01_fedcb_codebook.py`.
The script is federated-only. Verified by grep — the word "centralised" appears only
in the docstring (documentation context) and the result JSON `"mode": "federated"` field.
`_cold_eval_cmo()` now exclusively uses `route_R1` for the federated path, which is correct.

**Finding 2 resolved:** `02_aggregate.py` now implements `_load_v02_anchor()` which reads
`outputs/v02_fl_8020_ratio/seed{S}/W_component_results.json` →
`per_operating_point["PAPE-aggressive"]["cells"]["V0"]` for all three seeds.
V02 JSON files confirmed present for seeds 42, 123, 7 with values:
- seed42: PAPE=43.9937, HR@1=26.2578, HR@2=37.4428, MAE=0.4201
- seed123: PAPE=44.2039, HR@1=23.0977, HR@2=34.5530, MAE=0.4680
- seed7: PAPE=44.3425, HR@1=28.2365, HR@2=39.3568, MAE=0.4579

**pytest:** `tests/test_codebook_fl.py -v` → 4 passed in 8.00s (local machine).

Engineer smoke confirm (pre-delivered): seed=42 CMO PAPE=49.94 bit-identical to pre-patch.

Re-review items checked (all PASS, carried from prior PASS list):
- `strict=True` load: confirmed line 291.
- `model.eval()` set at line 289.
- `with torch.no_grad()` present in `gather_cold` (eval helper).
- stride=24 (= HORIZON) at line 265 via `default=HORIZON`.
- Seed plumbed through CLI, set at line 268 (`torch.manual_seed` + `np.random.seed`).
- Output path: `outputs/v05_fedcb_codebook/seed{S}/{cell}/...`.
- No MLflow (correct for this repo).
- `K_local_i = min(K_local, N_i)` enforced in `_local_kmeans`.
- `sample_weight = concat(counts)` in Stage 2 merge.

**Sweep unblocked.**

---

## Entry 2: Stage A — V5-FedCB-0 (data-only) (2026-04-29)

No run. `02_aggregate.py` pulls v02 §B.3 values directly. Skipped per plan §Stage A.

---

## Entry 3: Stage B — V5-FedCB-1 (K_local=4, α=1.0) (2026-04-29)

**Command pattern:**
```
uv run python experiments/v05_fedcb_codebook/01_fedcb_codebook.py --seed {S} --K_local 4 --alpha 1.0
```

| Seed | CMO PAPE | HR@1  | HR@2  | MAE    | Elapsed |
|------|----------|-------|-------|--------|---------|
| 42   | 49.935   | 25.47 | 37.30 | 0.4126 | 64s     |
| 123  | 51.233   | 23.89 | 35.34 | 0.4536 | 73s     |
| 7    | 49.332   | 26.47 | 39.07 | 0.4514 | 73s     |

3-seed mean: PAPE=50.167 ± 0.971

**Gate 2 evaluation:** mean PAPE = 50.17 ≤ 52.0 → **PASS**

**Stage D decision:** Gate 2 PASS → Stage D (α sweep) SKIPPED per plan §Experimental matrix.

Anomaly: background runs launched with `run_in_background=True` produced empty log files
(stdout redirected to tee). seed=42 was re-run as foreground to confirm output. All 3 seeds
confirmed complete via `result.json` presence check.

---

## Entry 4: Stage C — V5-FedCB-2a/2b (K_local sweep) (2026-04-29)

**Command pattern:**
```
uv run python experiments/v05_fedcb_codebook/01_fedcb_codebook.py --seed {S} --K_local {2,8} --alpha 1.0
```

6 runs: seeds × {K_local=2, K_local=8}.

| Cell        | Seed | CMO PAPE | HR@1  | HR@2  | MAE    | Elapsed |
|-------------|------|----------|-------|-------|--------|---------|
| fedcb_K2    | 42   | 51.174   | 25.53 | 37.21 | 0.4125 | 86s     |
| fedcb_K2    | 123  | 51.688   | 23.29 | 34.70 | 0.4532 | 100s    |
| fedcb_K2    | 7    | 49.247   | 26.68 | 39.32 | 0.4513 | 100s    |
| fedcb_K8    | 42   | 50.544   | 25.01 | 36.86 | 0.4125 | 102s    |
| fedcb_K8    | 123  | 51.402   | 23.58 | 35.09 | 0.4536 | 99s     |
| fedcb_K8    | 7    | 49.161   | 27.08 | 39.86 | 0.4505 | 101s    |

K2 3-seed mean: PAPE=50.703 ± 1.287
K8 3-seed mean: PAPE=50.369 ± 1.130

Anomaly note: Background process monitoring showed `tee` output files as empty.
K2 seed=42 was executed foreground for confirmation; all other runs confirmed via
`result.json` presence after background completion notifications.

---

## Entry 5: Stage D — α sweep (2026-04-29)

**Status: SKIPPED**

Gate 2: V5-FedCB-1 mean PAPE = 50.17 ≤ 52.0 → no α sweep required per plan.

---

## Entry 6: Stage E — Aggregate and Communication (2026-04-29)

**02_aggregate.py:**
```
uv run python experiments/v05_fedcb_codebook/02_aggregate.py
```
Output: `outputs/v05_fedcb_codebook/multiseed_summary.json`

Gate summary from aggregator:
- Gate 1: checked, 3/3 seeds, mean PAPE=44.180, pass=True
- Gate 2: checked, mean PAPE=50.167, pass=True
- Gate 3: checked, 2 candidates (K2=50.703, K8=50.369), any_pass=True

**03_communication.py:**
```
uv run python experiments/v05_fedcb_codebook/03_communication.py
```
Output: `outputs/v05_fedcb_codebook/communication_summary.json`

| K_local | Per-client/round (B) | Total/round (B) |
|---------|---------------------|-----------------|
| 2       | 3,720               | 305,792         |
| 4       | 4,240               | 347,392         |
| 8       | 5,280               | 430,592         |

---

## Entry 7: Task 3 — Paper Draft (2026-04-29)

Files written:
- `papers/v05_draft/v05_fedcb_codebook.md`
- `papers/v05_draft/EXPERIMENT_LOG.md` (this file)

Paper section update contains: experimental process, Table 1 (PAPE/HR@1/HR@2/MAE, 3-seed
mean ± std), Table 2 (communication), Gate summary. §5 Discussion and §6 Conclusion are
reserved for exp-critic and reporter per executor protocol.

---

## Summary: Executor-Level Concerns

1. **Routing conflation**: The ~6 pp PAPE gap between V5-FedCB-0 (R0, v02 T2 backbone) and
   V5-FedCB-1/2/3 (R1, v04 FedAvg backbone) cannot be cleanly attributed to federated codebook
   quality alone. The anchor differs in three axes simultaneously: backbone, routing, and
   codebook construction. This is a finding for exp-critic.

2. **Background run monitoring on Windows/tee**: Runs launched with `run_in_background=True`
   and stdout redirected via `tee` produced 0-byte internal task output files, making in-flight
   progress invisible. Workaround: foreground execution for critical confirmations.
   This is an executor-level operational note, not a code bug.

3. **VQ diagnostics nominal**: Stage 2 utilization=1.000 (all M=32 slots populated) across all
   seeds and K_local values. Perplexity ~26-27. No empty clusters. Consistent with well-separated
   h_g distribution.
