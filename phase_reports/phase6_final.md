# Phase 6 — integration, verification, release

Date: 2026-07-11

## Done
- CI on GitHub Actions: python (ruff+mypy+pytest), native (clang-format
  check, build, ctest), wasm (emsdk build), web (tsc+vite). All green from
  the first remote run.
- End-to-end browser verification with Playwright against the BUILT site:
  - exhibit 1: grokpack loads, scrubber works, no console errors
  - exhibit 2: 1536-feature browser + live WASM probe works
  - exhibit 3: steering produces different same-seed generations; engine
    loads, generates, exposes live features; no console errors
- Two release blockers found and fixed during verification:
  1. dynamic import of glassbox.js resolved relative to the bundled module
     (404 in production builds) -> resolve against document.baseURI
  2. _malloc/_free not exported by Emscripten by default -> -sEXPORTED_FUNCTIONS
- BENCHMARKS.md fully populated with measured numbers (no placeholders).
- docs/SAMPLES.md: unedited fixed-seed gallery from the final checkpoint.
- Site deployed to GitHub Pages from web/dist (gh-pages branch).

## Verification chain (what proves what)
- Python tests (21) pin tokenizer/model/SAE semantics to fixtures.
- C++ tests (12 cases, 750 assertions) pin the engine to those same fixtures:
  logits 1.8e-07 vs contract 1e-3.
- The deployed site serves a GBX exported from the exact checkpoint whose
  metrics are in BENCHMARKS.md; the browser runs the parity-tested engine.
- Playwright smoke tests exercised the real user paths on the built assets.

## Not done / cut (with reasons)
- OPTIONAL auto-interp feature labels: cut - no API budget assumed, and
  hand-labeling 1536 features would violate the honesty bar (unverifiable).
- Tier-M model, multi-seed runs: no GPU, one-laptop compute budget (D001,
  D005). All reported numbers are single-seed point estimates.
- WASM threads/workers: single-threaded engine is fast enough (>=340 tok/s
  in-browser for a 2.6M model); complexity not justified.
