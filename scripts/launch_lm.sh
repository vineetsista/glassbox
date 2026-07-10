#!/usr/bin/env bash
# Launch (or resume) the tier-S LM training run under nohup.
# Usage: scripts/launch_lm.sh [max_steps]
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS="${1:-6000}"
THREADS="${2:-8}"
RUN=runs/lm_s

mkdir -p "$RUN"
OMP_NUM_THREADS="$THREADS" nohup .venv/bin/python -m train.train_lm \
    --data data/tinystories.bin --run "$RUN" --resume --max-steps "$STEPS" \
    --threads "$THREADS" --compile --ckpt-every 500 \
    >> "$RUN/train.log" 2>&1 &
echo "launched pid $! -> $RUN/train.log"
