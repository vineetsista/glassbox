# BENCHMARKS

Every number measured on the build machine — i7-1250U (9W, 2P+8E, 12
threads), 7.6GiB RAM visible to WSL2, no GPU — unless marked CI. Sources in
parentheses; regenerate with the named script/log.

## Language model training (runs/lm_s/log.csv)

| metric | value |
|---|---|
| parameters | 2,557,632 (1,771,200 non-embedding) |
| tokens trained | 49,152,000 (6000 steps x 32 x 256) — ~0.47 epoch |
| final train loss | 1.906 (cross-entropy, nats/token) |
| final val loss | 1.921 (fixed-seed heldout batches) |
| training throughput | 966-3203 tok/s (torch.compile, 8 threads; range reflects grok-run contention then solo) |
| wall clock | ~2 days elapsed incl. laptop sleep; ~6h of actual compute |

Train/val curves overlap throughout (runs/lm_s/loss.png): at half an epoch
the model never sees data twice, so there is nothing to overfit. Loss was
still falling at step 6000 — the model is compute-bound, not data-bound.

## Tokenizer (tokenizer_train.log)

| metric | value |
|---|---|
| BPE training (50MB, vocab 4096) | 46 s |
| corpus packing (419MB -> 105M tokens, 8 workers) | 160 s (0.66 M tok/s) |
| compression on in-domain demo text | 53 chars -> 13 tokens (~4.1 chars/token) |

## Grokking specimen (runs/grok/log.csv, docs/figures/grok/findings.json)

| metric | value |
|---|---|
| memorization (train acc 100%) | step ~200 |
| grokked (test acc 99.97%) | step 3000 |
| final (step 7600) | train loss 1.2e-5, test loss 5.4e-5 |
| key frequencies | {9, 27, 40, 43, 45} |
| embedding power in key freqs | 90.6% of non-DC |
| neurons single-frequency (>85% power) | 90.0% of 512 |
| restricted test loss (key freqs only) | 1.1e-5 (vs full 5.4e-5) |
| excluded train loss (key freqs removed) | 10.7 |
| training rate | ~1.1 it/s full-batch (12769 x 3) at 4 threads |

## SAE (runs/sae_l2/log.csv)

| metric | value |
|---|---|
| training data | 20,000,000 cached fp16 activations (blocks.2.hook_resid_post) |
| steps | 20,000 x batch 4096 (~82M samples, ~4.1 epochs over the cache) |
| final FVU | 0.0976 |
| dead features (no fire in 500 steps) | 0 of 1536, throughout |
| L0 | 32 structurally (top-k) |
| mean active-feature activation | 1.68 |
| training throughput | ~15,000 activations/s (10 threads) |
| activation caching | 20M tokens in ~1h42m (~4,500 tok/s inference) |

## C++ engine

| metric | value |
|---|---|
| logit parity vs PyTorch (contract 1e-3) | max abs diff 1.8e-07 (engine_tests / engine_cli parity) |
| tokenizer id parity on fixtures | exact |
| SAE encode/decode parity | <= 1e-4 |
| native generation, tier-S model (1 thread) | 1,398 tok/s (runs/engine_bench.txt) |
| in-browser WASM generation (headless chromium) | >= 340 tok/s (240 tokens in <=0.7s; lower bound, poll-interval limited) |
| wasm binary size | 92,812 B (+12,365 B JS glue) |
| GBX bundle (model+tokenizer+SAE) | 12.6 MB fp32 |

## Web

| metric | value |
|---|---|
| app bundle | 164.8 kB JS (53.3 kB gzip), 4.1 kB CSS |
| grok scrubber data | 2.3 MB (77 checkpoints, fp16) |
| CI (GitHub runners) | python + native + wasm + web all green, first run |
