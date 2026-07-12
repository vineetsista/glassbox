# Phase 3 — sparse autoencoder

Date: 2026-07-09 (code), 2026-07-11 (run complete)

## Done
- 20M activations cached from blocks.2.hook_resid_post (fp16 memmap, 7.7GB)
  at ~4,500 tok/s inference; window offsets stored for context recovery.
- Top-k SAE (1536 features, k=32) trained 20k steps: **final FVU 0.0976,
  zero dead features at any point** (resampling machinery never needed to
  fire in the real run — verified separately that it works).
- Dashboards baked for all 1536 features: true argmax windows over a
  4M-token scan, log-bin histograms, direct-path logit lens, ~12MB of
  static JSON.
- Full pipeline ran unattended via scripts/post_lm_pipeline.sh (self-waits
  for the LM, then chains cache -> SAE -> dashboards -> exports -> site).

## Observations (from the baked index, checkable in exhibit 2)
- Feature density spans ~4 orders of magnitude; the k=32 budget concentrates
  on names, dialogue markers, story-position and syntax features — as
  expected at 2.6M params (see Limitations in README).
- Steering with these features produces visible, seed-stable story changes
  (verified in-browser via Playwright: amplifying a Lily-associated feature
  x4 turned a Benny/forest story into one where Lily speaks).

## Honest notes
- FVU 0.098 means ~10% of residual-stream variance is NOT captured by 32
  active features per token; steering preserves that error term untouched.
- The dashboard scan covers 4M of the 20M cached tokens (compute budget);
  max-activating examples are exact for the scanned sample, not the corpus.
