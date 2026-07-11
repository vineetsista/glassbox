# Phase 5 — web app

Date: 2026-07-09 (shell complete; assets land as pipeline runs finish)

## Done
- React + Vite + TS, strict tsc, zero backend, zero chart libraries: canvas
  renderers written by hand (line chart, bars, heatmap) in src/charts.ts.
- Exhibit 1 "Watch it grok": .grokpack parser (DataView + hand-rolled fp16
  decode), scrub slider + play mode over every checkpoint; accuracy curves,
  restricted/excluded progress measures (log scale), embedding DFT spectrum
  with key-freq highlights, embedding cosine-similarity heatmap, attention
  probe heatmap.
- Exhibit 2 "Look inside": feature browser over baked JSON dashboards
  (sortable, filterable), per-feature max-activating examples with token
  heat, activation histogram, logit-lens tables, plus LIVE probing: type any
  text, the WASM engine reports the feature's activation on every token.
- Exhibit 3 "Steer it": same-seed baseline vs steered generation side by
  side, feature search, per-feature multiplier sliders (0 = ablate), live
  active-feature readout during generation.
- Engine wrapper (src/engine.ts) around the Emscripten module; app degrades
  gracefully (each exhibit explains exactly which pipeline stage is missing).
- Build: 53KB gzipped JS. scripts/build_site.sh assembles all assets with
  loud warnings; scripts/dev.sh for local dev.

## Pending (blocked on training runs)
- Real grok.grokpack, model.gbx, features/ JSON; then a manual browser pass
  (Playwright smoke via webapp tooling) and screenshots for the README.
