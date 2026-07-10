# GLASSBOX

An interactive interpretability instrument. Small transformers trained from scratch,
their internal features extracted with sparse autoencoders, and a zero-backend browser
app where you can watch grokking unfold, inspect live feature activations, and steer
model behavior by ablating or amplifying individual features. Inference runs client-side
on a from-scratch C++ engine compiled to WASM SIMD.

> STATUS: under construction. This README is finalized in the last phase; until then it
> tracks what actually exists. Nothing here is aspirational — see phase_reports/.

## Layout

- `train/` — from-scratch GPT (RoPE, RMSNorm, hooks) + byte-level BPE trainer (PyTorch as substrate only)
- `grok/` — modular-arithmetic grokking specimen + Fourier analysis
- `sae/` — activation caching, top-k sparse autoencoder, feature dashboards
- `engine/` — C++20 inference engine, zero deps, native CLI + WASM SIMD build
- `web/` — React+Vite+TS app, three exhibits
- `scripts/` — launchers, exporters, build scripts
- `docs/` — METHODS, GBX_FORMAT, WALKTHROUGH, INTERVIEW_DRILL, BENCHMARKS

## Honesty

Every number in these docs traces to a script and a seed. Models here are tiny and
CPU-trained (see DECISIONS.md D001); samples and metrics are shown as-is, including the
mediocre ones. See the Limitations section (added with the final report).
