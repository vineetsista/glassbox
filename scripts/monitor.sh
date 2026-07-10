#!/usr/bin/env bash
# One-glance status of all long-running jobs. Usage: scripts/monitor.sh
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=== processes ==="
pgrep -af "train.train_lm|grok.train_grok|sae.train_sae|cache_activations" || echo "(none running)"

for run in runs/*/; do
    log="$run/log.csv"
    [ -f "$log" ] || continue
    echo "=== $run ==="
    tail -1 "$log"
done

echo "=== disk ==="
du -sh data runs 2>/dev/null
