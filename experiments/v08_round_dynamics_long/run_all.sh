#!/usr/bin/env bash
# v08 Phase 1 (default, lambda=0.3) + Phase 2 (codebook stacking) full sweep.
# Continue-on-error so partial failures don't kill the chain; aggregators
# robustly scan whatever cells exist.

PY='C:\Users\HOME\JW\Research Docs\FL_Peak_Project\.venv\Scripts\python.exe'
ROOT='C:\Users\HOME\JW\Research Docs\FL_Peak_Project\experiments\v08_round_dynamics_long'

echo "==== v08 full sweep started: $(date '+%Y-%m-%d %H:%M:%S') ===="

# ---------- Phase 1: 16 remaining runs ----------
# Centralised — 2 seeds remaining (seed 42 already done)
"$PY" "$ROOT\\01_centralised.py" --seed 123 --epochs 40
"$PY" "$ROOT\\01_centralised.py" --seed 7   --epochs 40

# FedAvg — 3 seeds
"$PY" "$ROOT\\02_fl_dynamics.py" --algorithm fedavg --seed 42  --local_epochs 5 --rounds 150
"$PY" "$ROOT\\02_fl_dynamics.py" --algorithm fedavg --seed 123 --local_epochs 5 --rounds 150
"$PY" "$ROOT\\02_fl_dynamics.py" --algorithm fedavg --seed 7   --local_epochs 5 --rounds 150

# FedProx — 3 seeds
"$PY" "$ROOT\\02_fl_dynamics.py" --algorithm fedprox --seed 42  --local_epochs 5 --rounds 150
"$PY" "$ROOT\\02_fl_dynamics.py" --algorithm fedprox --seed 123 --local_epochs 5 --rounds 150
"$PY" "$ROOT\\02_fl_dynamics.py" --algorithm fedprox --seed 7   --local_epochs 5 --rounds 150

# FedRep — 2 seeds remaining (seed 42 already done)
"$PY" "$ROOT\\02_fl_dynamics.py" --algorithm fedrep --seed 123 --local_epochs 5 --rounds 150
"$PY" "$ROOT\\02_fl_dynamics.py" --algorithm fedrep --seed 7   --local_epochs 5 --rounds 150

# Ditto — 2 seeds remaining (seed 42 already done)
"$PY" "$ROOT\\02_fl_dynamics.py" --algorithm ditto --seed 123 --local_epochs 5 --rounds 150
"$PY" "$ROOT\\02_fl_dynamics.py" --algorithm ditto --seed 7   --local_epochs 5 --rounds 150

# FedProto — 3 seeds
"$PY" "$ROOT\\02_fl_dynamics.py" --algorithm fedproto --seed 42  --local_epochs 5 --rounds 150
"$PY" "$ROOT\\02_fl_dynamics.py" --algorithm fedproto --seed 123 --local_epochs 5 --rounds 150
"$PY" "$ROOT\\02_fl_dynamics.py" --algorithm fedproto --seed 7   --local_epochs 5 --rounds 150

echo "==== Phase 1 done: $(date '+%Y-%m-%d %H:%M:%S') ===="

# ---------- Phase 1 aggregate + figures ----------
"$PY" "$ROOT\\06_aggregate.py" --seeds 42 123 7
"$PY" "$ROOT\\07_make_figures.py"

# ---------- Phase 2: 18 codebook stacking runs ----------
for seed in 42 123 7; do
  for cell in V6-Dyn-A_centralised V6-Dyn-B-FedAvg V6-Dyn-B-FedProx V6-Dyn-B-FedRep V6-Dyn-B-Ditto V6-Dyn-B-FedProto; do
    "$PY" "$ROOT\\08_codebook_stacking.py" --seed "$seed" --cell "$cell"
  done
done

echo "==== Phase 2 codebook done: $(date '+%Y-%m-%d %H:%M:%S') ===="

# ---------- Phase 2 aggregate + F6 ----------
"$PY" "$ROOT\\09_aggregate_codebook.py" --seeds 42 123 7
"$PY" "$ROOT\\10_make_codebook_figure.py"

echo "==== v08 full sweep finished: $(date '+%Y-%m-%d %H:%M:%S') ===="
