#!/usr/bin/env bash
set -euo pipefail

PY=/home/maxon/workspace/CTF/.venv/bin/python
HERE=/home/maxon/workspace/PhysicsRelationCTF/sdwpf
cd "$HERE"

for SPEC in \
  "final_original_100 cuda:0" \
  "final_physics_100 cuda:1" \
  "final_original_50 cuda:0" \
  "final_physics_50 cuda:1" \
  "tune_original_20 cuda:0" \
  "tune_physics_farm_rowmax_b2_20 cuda:1" \
  "final_original_10 cuda:0" \
  "final_physics_10 cuda:1"; do
  read -r RUN DEVICE <<< "$SPEC"
  "$PY" evaluate_run.py \
    --run-dir "/home/maxon/workspace/PhysicsRelationCTF/runs/sdwpf_36x6/$RUN" \
    --device "$DEVICE" --save-predictions
done

"$PY" aggregate_results.py
"$PY" make_figures.py
