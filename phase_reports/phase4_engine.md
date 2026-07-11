# Phase 4 — C++ engine

Date: 2026-07-09

## Done
- C++20, zero third-party deps (Catch2 tests only), -Wall -Wextra -Wpedantic
  -Werror, clang-format enforced.
- Hand-rolled: JSON parser (json.hpp), GBX loader (gbx.hpp), byte-level BPE
  runtime with a scanner replicating the Python pre-tokenizer (tokenizer.hpp),
  fp32 kernels (ops.hpp), KV-cached GPT forward (model.hpp), top-k SAE with
  steering (sae.hpp), xorshift128+ sampler (sampler.hpp), CLI (main.cpp),
  WASM C API (wasm_api.cpp).
- GBX format designed + documented (docs/GBX_FORMAT.md): one bundle carries
  model + tokenizer + SAE; exporter in scripts/export_gbx.py.
- Steering semantics: delta edits in SAE feature space at the hook layer,
  reconstruction error preserved; ablate = x0, amplify > x1; reversibility
  tested.

## Measured (engine/build, Release, this laptop)
- 12 Catch2 test cases / 750 assertions green.
- Logit parity vs PyTorch on committed fixtures: max |diff| = 1.8e-07
  (contract: <= 1e-3). Also exposed as `engine_cli parity`.
- Tokenizer parity: exact ids on all fixture strings (ASCII, accented, CJK).
- SAE encode/decode parity: <= 1e-4 vs fixture.
- WASM SIMD build (emcc -msimd128): glassbox.wasm 93KB + 12KB JS glue.

## Known divergence (documented in METHODS.md)
- C++ \s is ASCII-only in the pre-tokenizer; Python matches Unicode
  whitespace. Boundaries can differ on exotic whitespace; round-trip exact in
  both. TinyStories is ASCII-dominant; fixtures pass exactly.

## Deferred to final phase
- tokens/sec benchmarks on the real tier-S model (BENCHMARKS.md) — needs the
  trained checkpoint.
