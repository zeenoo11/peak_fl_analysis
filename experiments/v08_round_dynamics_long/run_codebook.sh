#!/usr/bin/env bash
# v08 Phase 2 codebook stacking — 17 cells × seeds where final_state_dict.pt exists.
# FedProto seed 7 is excluded (still training); will run separately after it finishes.

PY='C:\Users\HOME\JW\Research Docs\FL_Peak_Project\.venv\Scripts\python.exe'
ROOT='C:\Users\HOME\JW\Research Docs\FL_Peak_Project\experiments\v08_round_dynamics_long'

echo "==== Phase 2 codebook (17 of 18 runs) started: $(date '+%Y-%m-%d %H:%M:%S') ===="

for seed in 42 123 7; do
  for cell in V6-Dyn-A_centralised V6-Dyn-B-FedAvg V6-Dyn-B-FedProx V6-Dyn-B-FedRep V6-Dyn-B-Ditto; do
    "$PY" "$ROOT\\08_codebook_stacking.py" --seed "$seed" --cell "$cell"
  done
done

# FedProto seeds 42 + 123 only (seed 7 still training)
"$PY" "$ROOT\\08_codebook_stacking.py" --seed 42  --cell V6-Dyn-B-FedProto
"$PY" "$ROOT\\08_codebook_stacking.py" --seed 123 --cell V6-Dyn-B-FedProto

echo "==== Phase 2 codebook done (17 of 18): $(date '+%Y-%m-%d %H:%M:%S') ===="
