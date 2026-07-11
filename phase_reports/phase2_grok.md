# Phase 2 — grokking specimen

Date: 2026-07-09 (training in flight; export/analysis after run completes)

## Done
- grok/model.py: 1-layer transformer per Nanda et al. (d128, 4 heads, ReLU
  d_mlp 512, learned pos emb, no LayerNorm, untied unembed), shares the
  HookPoint system with the LM.
- grok/train_grok.py: full-batch AdamW, lr 1e-3, wd 1.0, betas (.9,.98),
  30% train fraction (seed 999), checkpoint every 100 steps (fp16 archive +
  metrics + atomic resume ckpt).
- grok/fourier.py: embedding DFT norms, key-frequency detection, logit-grid
  2D-DFT restricted/excluded losses (our exact definitions in METHODS.md
  section 3), per-neuron frequency attribution.
- grok/analyze.py: figures + findings.json from OUR weights, no hand-waving.
- grok/export_web.py: .grokpack binary for the browser scrubber (per-ckpt
  metrics, embedding fp16, DFT spectrum, attention probe rows).
- tests/test_grok.py: planted-frequency recovery, restricted==full on purely
  periodic logits, split determinism.

## Observed so far (runs/grok/log.csv — the numbers, not a narrative)
- step 200: train acc 100.0%, test acc 5.7% (memorization complete)
- step 1000: test acc 12.9%; test loss rising (18.1) — classic overfit phase
- step 2000: test acc 25.7%, test loss falling — transition begins
- step 3000: test acc 99.97%
- step 3300: test acc 100.0%
- run continues for the weight-decay cleanup tail; will be stopped at 10k
  (decision D006) since the full arc is captured.
