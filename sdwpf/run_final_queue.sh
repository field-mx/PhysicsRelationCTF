#!/usr/bin/env bash
set -euo pipefail

PY=/home/maxon/workspace/CTF/.venv/bin/python
HERE=/home/maxon/workspace/PhysicsRelationCTF/sdwpf
ROLE=${1:?original or physics}
DEVICE=${2:?cuda device}
cd "$HERE"

for FRACTION in 100 50 10; do
  if [[ "$ROLE" == original ]]; then
    "$PY" train.py \
      --model original --fraction "$FRACTION" \
      --run-name "final_original_${FRACTION}" --device "$DEVICE" \
      --batch-size 16 --epochs 12 --patience 3
  elif [[ "$ROLE" == physics ]]; then
    "$PY" train.py \
      --model physics --fraction "$FRACTION" \
      --run-name "final_physics_${FRACTION}" --device "$DEVICE" \
      --batch-size 16 --epochs 12 --patience 3 \
      --kw 0.075 --beta-init 2.0 --direction-mode farm \
      --bias-normalization rowmax
  else
    echo "unknown role: $ROLE" >&2
    exit 2
  fi
done
