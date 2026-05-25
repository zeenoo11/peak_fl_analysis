# v08 audit follow-up: C1 (Ditto re-run with global_state_dict.pt) + H2 (MAEonly 18 runs)
# Sequential — single GPU. Total wall ~5 hours.
#
# Generated 2026-05-18 in response to 3-agent sonnet critical review.

$ErrorActionPreference = "Stop"
$PY = "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe"
$EXP = "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long"
$LOG = "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/outputs/v08_round_dynamics_long/_audit_fix_progress.log"
$null = New-Item -ItemType Directory -Force (Split-Path $LOG)

function Log {
    param([string]$msg)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $LOG -Value $line
}

Log "=========================================="
Log "v08 audit-fix START"
Log "=========================================="

# ============================================================================
# Step 1: Ditto re-run (C1) — 3 seeds × ~25 min = ~75 min
#   Patched 02_fl_dynamics.py now persists global_state_dict.pt alongside
#   final_state_dict.pt; patched 08_codebook_stacking.py uses global trunk
#   for Ditto cells.
# ============================================================================
Log "--- Step 1/4: Ditto re-run (3 seeds, ~75 min total) ---"
foreach ($seed in 42, 123, 7) {
    Log "Ditto seed=$seed ..."
    & $PY "$EXP/02_fl_dynamics.py" --algorithm ditto --local_epochs 5 --rounds 150 --seed $seed
    if ($LASTEXITCODE -ne 0) { Log "ABORT: ditto seed=$seed exit=$LASTEXITCODE"; exit 1 }
}
Log "Step 1 done."

# ============================================================================
# Step 2: MAEonly backbone (H2) — 18 runs total
#   centralised × 3 + 5 FL algos × 3 = 3 + 15 = 18 runs.
#   Per-seed wall (estimated): ce 28s, fa 600s, fp 950s, fr 565s, di 1500s, fpr 825s.
#   Total ≈ 3 × (28+600+950+565+1500+825) = ~3.7 hours.
# ============================================================================
Log "--- Step 2/4: MAEonly backbone (18 runs, ~3.7 hours) ---"
foreach ($seed in 42, 123, 7) {
    Log "MAEonly centralised seed=$seed ..."
    & $PY "$EXP/01_centralised.py" --seed $seed --epochs 40 --aux_lambda 0
    if ($LASTEXITCODE -ne 0) { Log "ABORT: ce-MAEonly seed=$seed"; exit 1 }

    foreach ($algo in "fedavg", "fedprox", "fedrep", "ditto", "fedproto") {
        Log "MAEonly $algo seed=$seed ..."
        & $PY "$EXP/02_fl_dynamics.py" --algorithm $algo --local_epochs 5 --rounds 150 --seed $seed --aux_lambda 0
        if ($LASTEXITCODE -ne 0) { Log "ABORT: $algo-MAEonly seed=$seed"; exit 1 }
    }
}
Log "Step 2 done."

# ============================================================================
# Step 3: Phase 2 codebook stacking — re-stack Ditto (now uses global trunk)
#                                    + new MAEonly cells (all 6)
#   3 (Ditto re-stack) + 18 (MAEonly) = 21 runs, ~5s each.
# ============================================================================
Log "--- Step 3/4: Phase 2 codebook stacking (21 runs, ~2 min) ---"
foreach ($seed in 42, 123, 7) {
    # Ditto re-stack on new global trunk
    Log "Phase 2: Ditto seed=$seed (global trunk) ..."
    & $PY "$EXP/08_codebook_stacking.py" --seed $seed --cell V6-Dyn-B-Ditto
    if ($LASTEXITCODE -ne 0) { Log "ABORT: Ditto restack seed=$seed"; exit 1 }

    # 6 MAEonly cells
    foreach ($cell in "V6-Dyn-A_centralised-MAEonly",
                       "V6-Dyn-B-FedAvg-MAEonly",
                       "V6-Dyn-B-FedProx-MAEonly",
                       "V6-Dyn-B-FedRep-MAEonly",
                       "V6-Dyn-B-Ditto-MAEonly",
                       "V6-Dyn-B-FedProto-MAEonly") {
        Log "Phase 2: $cell seed=$seed ..."
        & $PY "$EXP/08_codebook_stacking.py" --seed $seed --cell $cell
        if ($LASTEXITCODE -ne 0) { Log "ABORT: $cell seed=$seed phase2"; exit 1 }
    }
}
Log "Step 3 done."

# ============================================================================
# Step 4: Aggregate + figures
#   06_aggregate (Phase 1 backbone — picks up new Ditto + new MAEonly cells)
#   07_make_figures (refresh F1/F1b/F1c/F2/F3 with new data)
#   09_aggregate_codebook (Phase 2 codebook lift summary)
#   10_make_codebook_figure (refresh F6)
#   11_make_ablation_figures (MAEonly subplots now populated)
# ============================================================================
Log "--- Step 4/4: Aggregate + figures (~30s) ---"
& $PY "$EXP/06_aggregate.py" --seeds 42 123 7
if ($LASTEXITCODE -ne 0) { Log "ABORT: 06_aggregate"; exit 1 }
& $PY "$EXP/07_make_figures.py"
if ($LASTEXITCODE -ne 0) { Log "ABORT: 07_make_figures"; exit 1 }
& $PY "$EXP/09_aggregate_codebook.py" --seeds 42 123 7
if ($LASTEXITCODE -ne 0) { Log "ABORT: 09_aggregate_codebook"; exit 1 }
& $PY "$EXP/10_make_codebook_figure.py"
if ($LASTEXITCODE -ne 0) { Log "ABORT: 10_make_codebook_figure"; exit 1 }
& $PY "$EXP/11_make_ablation_figures.py"
if ($LASTEXITCODE -ne 0) { Log "WARN: 11_make_ablation_figures non-zero (may not be fatal)" }

Log "=========================================="
Log "v08 audit-fix DONE"
Log "=========================================="
