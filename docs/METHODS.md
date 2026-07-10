# METHODS

What was built, precisely, and where each claim comes from. Anything with a
number in the README traces to a script named here plus a seed recorded in the
relevant `config.json`.

## 1. Language model (train/)

Decoder-only transformer, written by hand on top of raw PyTorch ops
(`train/gpt.py`): RoPE (interleaved-pair rotation, base 10000), pre-norm
RMSNorm (eps 1e-5), GELU (tanh approximation) MLP at 4x width, tied
input/output embeddings, no biases anywhere, dropout 0.

Tier S (DECISIONS.md D001/D005): d_model 192, 4 layers, 6 heads, d_mlp 768,
ctx 256, vocab 4096 -> 2,557,632 parameters (1,771,200 non-embedding).

Hook system: every interesting intermediate value flows through a named
`HookPoint` (identity `nn.Module`). Hook fns observe or rewrite the value;
`hooks()` is a context manager guaranteeing removal; `run_with_cache()`
captures activations. The C++ engine reimplements the same intervention
point for SAE steering (section 5). Names are listed in `train/gpt.py`.

Training (`train/train_lm.py`): AdamW (betas 0.9/0.95, wd 0.1), cosine
schedule with 500-step warmup decaying to 10% of peak lr 3e-3, grad clip 1.0,
batch 32 x 256 tokens, fp32 on CPU, `torch.compile`. Checkpoints are atomic
(tmp+rename) and carry optimizer + RNG state; runs resume exactly.

Data: TinyStories (roneneldan/TinyStories via `datasets`, download only),
419MB of train text, stories separated by an explicit sentinel (D004), packed
to a flat uint16 binary of 105,028,360 tokens. Validation = held-out tail 1%
of the pack plus the official validation split for the text corpus.

## 2. Tokenizer (train/bpe.py)

Byte-level BPE trained from scratch on 50MB of TinyStories, vocab 4096
(256 bytes + 3839 merges + `<|endoftext|>`). Trainer uses word-frequency
BPE with incremental pair-count updates (46s for 50MB). Pre-tokenizer is a
GPT-2-flavored regex restricted to stdlib `re` (ASCII letter/digit classes).

Round-trip is exact on arbitrary unicode by construction (byte fallback);
property-tested on random codepoints in `tests/test_bpe.py`.

Known divergence, C++ runtime (`engine/src/tokenizer.hpp`): Python `\s`
matches Unicode whitespace (U+00A0 etc.); the C++ scanner treats only ASCII
whitespace as whitespace, so exotic whitespace pre-tokenizes differently.
Round-trip remains exact in both; only token boundaries can differ, and only
on text far outside the training distribution. Parity on committed fixtures
(ASCII + accented + CJK samples) is exact (`engine/tests/test_tokenizer.cpp`).

## 3. Grokking specimen (grok/)

Per Power et al. 2022 / Nanda et al. 2023: modular addition mod p=113,
sequences `[a, b, =]`, 1-layer transformer (d_model 128, 4 heads, d_mlp 512,
ReLU, learned positional embeddings, no LayerNorm, untied unembedding),
full-batch AdamW lr 1e-3, betas (0.9, 0.98), weight decay 1.0, train fraction
30% (seed 999). Checkpoint every 100 steps with metrics; fp16 archived
weights.

Observed (runs/grok/log.csv): train accuracy 100% by step ~200 while test
accuracy sat below 10%; test accuracy reached 99.97% by step 3000 and 100.0%
by step 3300. Exact curves regenerate from the CSV; figures from
`python -m grok.analyze`.

### Progress measures — our exact definitions

Let `L[a,b,:]` be the logit grid over all p^2 inputs (`grok/fourier.py`).
Take the 2D DFT of `L` over the (a, b) axes. Let K = the 5 frequencies with
the largest embedding-DFT norm in the FINAL checkpoint (held fixed across the
scrub). Build the component mask M = { (fa, fb) : |fa| in K u {0} and |fb| in
K u {0} }, minus the pure-DC term.

- restricted_test = cross-entropy of `idft(spec * (M + DC))` on test pairs
- excluded_train  = cross-entropy of `idft(spec * (1 - M))` on train pairs

These are *inspired by* Nanda et al.'s restricted/excluded loss but are
defined on the logit spectrum rather than on internal Fourier components;
they are coarser but model-agnostic. Interpretation is the same: restricted
tracking full test loss shows the periodic circuit forming; excluded loss
rising above train loss shows the memorization residue dissolving.

Neuron frequency attribution: activation of each MLP neuron over the (a, b)
grid -> 2D DFT -> power attributed to k = max(|fa|, |fb|); a neuron "is tuned
to k" if most non-DC power lands on one k. Numbers in
`docs/figures/grok/findings.json`.

## 4. Sparse autoencoder (sae/)

Top-k SAE (`sae/model.py`): 8x expansion (1536 features on d_in 192), k=32,
unit-norm decoder columns (renormalized every step), encoder initialized to
decoder transpose, biases zero. Trained with Adam lr 3e-4 on fp16-memmapped
`blocks.2.hook_resid_post` activations (the hook layer is recorded in the GBX
header). Dead features (no fire in 500 steps) are resampled every 2000 steps
toward high-reconstruction-error inputs with Adam state reset.

Metrics reported (runs/sae_l2/log.csv): FVU (fraction of variance
unexplained), dead-feature count, mean activation. L0 is structurally k=32.
No cherry-picking: the log CSV is the record.

Dashboards (`sae/dashboards.py`): max-activating examples are the true argmax
windows over the scanned sample; histograms use fixed log bins 1e-2..1e2.

Logit lens: `W_dec[:, f] @ W_U` (tied embedding). This is a DIRECT-PATH
approximation: it ignores blocks after the hook layer and the final RMSNorm's
per-example scale. It answers "what does this direction push toward if read
out directly", not "what is this feature's total causal effect".

## 5. C++ engine (engine/)

C++20, zero third-party dependencies (Catch2 in tests only), -Wall -Wextra
-Wpedantic -Werror. Hand-rolled: JSON parser, GBX loader, BPE runtime,
matvec/rmsnorm/rope/softmax/gelu kernels, KV-cached forward pass, top-k SAE,
xorshift128+ sampler. WASM build: Emscripten, `-msimd128`, no threads.

Parity contract: logits within 1e-3 (fp32) of PyTorch on committed fixture
vectors. Measured: max |diff| = 1.8e-07 (`engine/tests/test_parity.cpp`,
also `engine_cli parity`). Tokenizer and SAE parity are exact to 1e-4.

Steering: with z = enc(resid_post) at the hook layer, for each edited feature
f with multiplier m, resid += (m - 1) * z_f * W_dec[:, f]; a non-firing
feature with m > 1 is added as m * W_dec[:, f]. The SAE reconstruction error
term is untouched (we edit deltas, never replace the stream with a
reconstruction). Ablation = m 0. Reversibility is tested.

## 6. Determinism

Seeds: LM 1337, grok 999, SAE 4242, fixtures 20260709/7/31337/11, eval
sampling 0, C++ sampler seeded explicitly. Eval prompts are frozen in
`train/sample.py`. CI never trains; fixtures pin behavior.
