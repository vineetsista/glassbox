#!/usr/bin/env bash
# Assemble all web assets and build the static site into web/dist.
# Prereqs (each produced by earlier pipeline stages; missing ones are skipped
# with a loud warning so the site still builds with partial exhibits):
#   runs/grok/ckpts/          -> grok.grokpack (python -m grok.export_web)
#   runs/lm_s/latest.pt       -> model.gbx    (scripts/export_gbx.py)
#   runs/sae_l2/latest.pt     -> baked into model.gbx + features/ JSON
#   engine/build-wasm/        -> glassbox.js/.wasm (emcmake)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
ASSETS=web/public/assets
mkdir -p "$ASSETS"

echo "== grokpack =="
if [ -d runs/grok/ckpts ] && [ -n "$(ls runs/grok/ckpts 2>/dev/null)" ]; then
    [ -f "$ASSETS/grok.grokpack" ] || $PY -m grok.export_web --run runs/grok --out "$ASSETS/grok.grokpack"
else
    echo "WARNING: no grok checkpoints; exhibit 1 will show an error"
fi

echo "== model.gbx =="
if [ -f runs/lm_s/latest.pt ]; then
    if [ -f runs/sae_l2/latest.pt ]; then
        $PY scripts/export_gbx.py --ckpt runs/lm_s/latest.pt --tokenizer data/tokenizer.json \
            --sae runs/sae_l2/latest.pt --out "$ASSETS/model.gbx"
    else
        echo "WARNING: no SAE checkpoint; exporting model without steering support"
        $PY scripts/export_gbx.py --ckpt runs/lm_s/latest.pt --tokenizer data/tokenizer.json \
            --out "$ASSETS/model.gbx"
    fi
else
    echo "WARNING: no LM checkpoint; exhibits 2/3 will show an error"
fi

echo "== wasm engine =="
if [ -f engine/build-wasm/glassbox.js ]; then
    cp engine/build-wasm/glassbox.js engine/build-wasm/glassbox.wasm "$ASSETS/"
else
    echo "WARNING: wasm engine not built (emcmake cmake -B build-wasm && ninja)"
fi

echo "== feature dashboards =="
if [ ! -f "$ASSETS/features/index.json" ]; then
    echo "WARNING: dashboards not baked (python -m sae.dashboards); exhibit 2 degraded"
fi

echo "== vite build =="
cd web
npm run build
echo "site in web/dist ($(du -sh dist | cut -f1))"
