#!/usr/bin/env bash
# Runs the full post-LM pipeline once runs/lm_s finishes 6000 steps:
#   activation cache -> SAE training -> dashboards -> GBX export ->
#   sample gallery -> loss plot -> site build.
# Idempotent-ish: each stage skips if its output already exists.
# Log: runs/pipeline.log
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
THREADS=10
export OMP_NUM_THREADS=$THREADS

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for LM run to reach 6000 steps..."
while true; do
    last=$(tail -1 runs/lm_s/log.csv 2>/dev/null | cut -d, -f1 || echo 0)
    if [ "${last:-0}" = "6000" ] || grep -q "training complete" runs/lm_s/train.log 2>/dev/null; then
        break
    fi
    if ! pgrep -f "train.train_lm" > /dev/null; then
        log "LM trainer not running and not finished (last step: $last) - aborting"
        exit 1
    fi
    sleep 300
done
log "LM training complete (step $(tail -1 runs/lm_s/log.csv | cut -d, -f1))"

if [ ! -f data/acts_l2/meta.json ]; then
    log "caching 20M activations..."
    $PY -m sae.cache_activations --ckpt runs/lm_s/latest.pt --data data/tinystories.bin \
        --out data/acts_l2 --n-tokens 20000000 --threads $THREADS
fi

if [ ! -f runs/sae_l2/latest.pt ] || [ "$(tail -1 runs/sae_l2/log.csv 2>/dev/null | cut -d, -f1)" != "20000" ]; then
    log "training SAE (20k steps)..."
    $PY -m sae.train_sae --acts data/acts_l2 --run runs/sae_l2 --resume --threads $THREADS
fi

log "baking dashboards..."
$PY -m sae.dashboards --sae runs/sae_l2/latest.pt --acts data/acts_l2 \
    --lm runs/lm_s/latest.pt --tokenizer data/tokenizer.json \
    --out web/public/assets/features --scan-tokens 4000000 --threads $THREADS

log "exporting model.gbx..."
$PY scripts/export_gbx.py --ckpt runs/lm_s/latest.pt --tokenizer data/tokenizer.json \
    --sae runs/sae_l2/latest.pt --out web/public/assets/model.gbx

log "sample gallery..."
$PY scripts/make_gallery.py --ckpt runs/lm_s/latest.pt --tokenizer data/tokenizer.json \
    --out docs/SAMPLES.md

log "copying wasm engine + building site..."
cp engine/build-wasm/glassbox.js engine/build-wasm/glassbox.wasm web/public/assets/
(cd web && npm run build)

log "engine benchmark on the real model..."
./engine/build/engine_cli bench --gbx web/public/assets/model.gbx --steps 300 | tee runs/engine_bench.txt

log "PIPELINE COMPLETE"
