# v08 Phase 1 (default, λ=0.3) + Phase 2 (codebook stacking) full sweep.
# Phase 1 16 remaining runs (centralised seed 42 + FedRep seed 42 already done).
# Continue-on-error (;) so partial failures don't kill the chain; aggregators
# robustly scan whatever cells exist.

$ErrorActionPreference = "Continue"
$py = "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe"
$root = "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long"

Write-Output "==== v08 full sweep started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===="

# ---------- Phase 1: 16 remaining runs ----------
# Centralised — 2 seeds remaining
& $py "$root/01_centralised.py" --seed 123 --epochs 40
& $py "$root/01_centralised.py" --seed 7   --epochs 40

# FedAvg — 3 seeds
& $py "$root/02_fl_dynamics.py" --algorithm fedavg --seed 42  --local_epochs 5 --rounds 150
& $py "$root/02_fl_dynamics.py" --algorithm fedavg --seed 123 --local_epochs 5 --rounds 150
& $py "$root/02_fl_dynamics.py" --algorithm fedavg --seed 7   --local_epochs 5 --rounds 150

# FedProx — 3 seeds
& $py "$root/02_fl_dynamics.py" --algorithm fedprox --seed 42  --local_epochs 5 --rounds 150
& $py "$root/02_fl_dynamics.py" --algorithm fedprox --seed 123 --local_epochs 5 --rounds 150
& $py "$root/02_fl_dynamics.py" --algorithm fedprox --seed 7   --local_epochs 5 --rounds 150

# FedRep — 2 seeds remaining
& $py "$root/02_fl_dynamics.py" --algorithm fedrep --seed 123 --local_epochs 5 --rounds 150
& $py "$root/02_fl_dynamics.py" --algorithm fedrep --seed 7   --local_epochs 5 --rounds 150

# Ditto — 3 seeds
& $py "$root/02_fl_dynamics.py" --algorithm ditto --seed 42  --local_epochs 5 --rounds 150
& $py "$root/02_fl_dynamics.py" --algorithm ditto --seed 123 --local_epochs 5 --rounds 150
& $py "$root/02_fl_dynamics.py" --algorithm ditto --seed 7   --local_epochs 5 --rounds 150

# FedProto — 3 seeds
& $py "$root/02_fl_dynamics.py" --algorithm fedproto --seed 42  --local_epochs 5 --rounds 150
& $py "$root/02_fl_dynamics.py" --algorithm fedproto --seed 123 --local_epochs 5 --rounds 150
& $py "$root/02_fl_dynamics.py" --algorithm fedproto --seed 7   --local_epochs 5 --rounds 150

Write-Output "==== Phase 1 done: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===="

# ---------- Phase 1 aggregate + figures ----------
& $py "$root/06_aggregate.py" --seeds 42 123 7
& $py "$root/07_make_figures.py"

# ---------- Phase 2: 18 codebook stacking runs ----------
foreach ($seed in @(42, 123, 7)) {
    foreach ($cell in @("V6-Dyn-A_centralised", "V6-Dyn-B-FedAvg", "V6-Dyn-B-FedProx", "V6-Dyn-B-FedRep", "V6-Dyn-B-Ditto", "V6-Dyn-B-FedProto")) {
        & $py "$root/08_codebook_stacking.py" --seed $seed --cell $cell
    }
}

Write-Output "==== Phase 2 codebook done: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===="

# ---------- Phase 2 aggregate + F6 ----------
& $py "$root/09_aggregate_codebook.py" --seeds 42 123 7
& $py "$root/10_make_codebook_figure.py"

Write-Output "==== v08 full sweep finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===="
