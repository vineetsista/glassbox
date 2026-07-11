# Phase 2 — grokking specimen

Date: 2026-07-09, closed 2026-07-11 (run stopped at 7600 steps, D006)

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

## Observed (runs/grok/log.csv — the numbers, not a narrative)
- step 200: train acc 100.0%, test acc 5.7% (memorization complete)
- step 1000: test acc 12.9%; test loss rising (18.1) — classic overfit phase
- step 2000: test acc 25.7%, test loss falling — transition begins
- step 3000: test acc 99.97%; step 3300: 100.0%
- stopped at 7600 (D006): train loss 1.2e-5, test loss 5.4e-5, both acc 100%

## Fourier circuit verification (docs/figures/grok/findings.json)
- key frequencies (from final embedding DFT): {9, 27, 40, 43, 45}
- 90.6% of non-DC embedding power in those 5 of 56 frequencies
- 90.0% of the 512 MLP neurons put >85% of non-DC power on one frequency;
  cluster sizes: f9->94, f40->139, f43->136, f45->143. Honest wrinkle:
  f27 is in the embedding top-5 but dominates no neuron cluster.
- restricted test loss (key-freq logit components only): 1.1e-5 — BETTER
  than the full model's 5.4e-5. excluded train loss (key freqs removed):
  10.7. The periodic circuit is not just present; it is the whole model.
- exports: 2.3MB grok.grokpack (77 ckpts) for the web scrubber; four
  figures under docs/figures/grok/.
