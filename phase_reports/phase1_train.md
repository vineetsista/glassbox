# Phase 1 — tokenizer, model, data, training loop

Date: 2026-07-09

## Done
- Byte-level BPE from scratch (train/bpe.py): word-frequency trainer with
  incremental pair counts. Vocab 4096 trained on 50MB TinyStories in 46s.
  Property tests: exact round-trip on random unicode (5 planes), whitespace
  torture, >2x compression on in-domain text.
- Hand-written GPT (train/gpt.py): RoPE, pre-norm RMSNorm, GELU-tanh MLP,
  tied embeddings, no biases. 2,557,632 params at tier S.
- First-class hook system: named HookPoints at every interesting tensor;
  observe/intervene; hooks() context manager; run_with_cache(). Later shared
  with the grok model via the HookedModel mixin.
- Data: TinyStories 419MB -> 105,028,360 packed uint16 tokens.
- Training loop: AdamW + cosine w/ warmup, grad clip, CSV+PNG logging, atomic
  checkpoint/resume with RNG state, nohup launcher + monitor script,
  torch.compile flag (1.4x on this CPU).
- Tests: 21 python tests green (bpe, gpt fixtures/causality/hook semantics/
  RoPE properties, sae units), fixtures pinned for CI.

## Incidents (all fixed, logged in DECISIONS.md)
- D004: first data pack split stories at paragraph blank lines -> EOT
  mid-story. Refetched with explicit sentinel; tokenizer unaffected.
- ruff caught a literal NBSP in a test string (violates ASCII-only rule).
- mypy caught a real bug: GrokModel initially lacked the hook utilities it
  was being called with (would have crashed at runtime).

## Run (completed 2026-07-11)
- runs/lm_s: 6000 steps x 8192 tok = 49M tokens (~0.47 epoch), lr 3e-3
  cosine, seed 1337, torch.compile, 8 threads.
- Final: train loss 1.906, val loss 1.921 (fixed-seed heldout batches).
  Curves overlap throughout — at half an epoch nothing repeats, so nothing
  overfits; loss still falling at the end (compute-bound). loss.png archived
  to docs/figures/lm_loss.png.
- Throughput 966-3203 tok/s depending on core contention with the grok run
  (D005/D006); ~6h of real compute spread over 2 days of laptop sleep.
