# INTERVIEW DRILL

Questions a sharp interviewer would ask about GLASSBOX, with the honest
answers this repo supports. Rule: never claim beyond what a script measured.

## Architecture / training

**Why RMSNorm + RoPE + tied embeddings for the LM?**
Pre-norm RMSNorm is the stability workhorse for small transformers (one gain
vector, no mean subtraction — also trivial to reimplement bit-compatibly in
C++). RoPE gives relative positions without a learned table and its identity
at position 0 makes the KV-cache implementation clean. Tied embeddings cut
~30% of parameters at this scale and give the SAE logit lens a natural W_U.

**Why is the grok model different (no LayerNorm, learned pos emb, untied)?**
Faithfulness to the setup where the Fourier-circuit result is established
(Nanda et al.). No LayerNorm keeps activations linear-ish in the embeddings
so DFT analysis is clean; positions are only 3 tokens so a learned table is
simplest.

**Your model is 2.6M params. Why should anyone care?**
Because every claim is checkable end to end on one laptop. The pipeline —
hooks, SAE, steering, parity-tested engine — is the same shape as
frontier-scale interp tooling; the scale is the price of honesty
(DECISIONS.md D005 documents the hardware and the sizing math).

**Where does the loss curve saturate and why?**
See runs/lm_s/loss.png and BENCHMARKS.md for the actual numbers. 49M tokens
is ~0.5 epoch of our pack; the model is data-rich but compute-starved at
this size — bigger d_model would keep improving faster than more epochs.

## Grokking

**What exactly is grokking?**
Delayed generalization: train accuracy hits 100% thousands of steps before
test accuracy moves. Ours: memorized by step ~200, grokked (test 99.97%) at
step 3000 (log.csv is the receipt).

**How do you KNOW your model learned the Fourier algorithm and isn't just
generalizing some other way?**
Three independent measurements on our weights (grok/analyze.py):
(1) embedding DFT power concentrates on ~5 frequencies; (2) MLP neurons'
activation grids over (a, b) are single-frequency plaids — the majority of
neurons put most non-DC power on one key frequency; (3) reconstructing the
logits from ONLY key-frequency components preserves test loss (restricted),
while deleting them destroys it (excluded). Numbers in findings.json.

**Your restricted/excluded losses aren't Nanda's exact ones.**
Correct, and METHODS.md section 3 says so: ours are defined on the 2D DFT of
the logit grid — coarser, model-agnostic, same interpretation. I can walk
through the derivation of why cos(w(a+b-c)) structure implies logit spectrum
sparsity.

**Why does weight decay matter?**
It is the selection pressure: the memorizing solution has large norm, the
periodic solution is compact. wd=1.0 makes the transition fast (ours grokked
in ~3k steps; with small wd it can take 10x longer or stall). The falling
weight-norm trace through the transition is in the scrubber.

## SAE

**Why top-k instead of L1?**
No L1 coefficient to tune, structurally exact L0=k, no shrinkage on active
features. The cost: k is a hard prior on per-token feature count, and dead
features need explicit resampling (we do; counts logged).

**What is FVU and what did you get?**
Fraction of variance unexplained on reconstruction. See runs/sae_l2/log.csv
and BENCHMARKS.md — the log IS the reported number, no cherry-picking.

**Is your logit lens causal?**
No — direct-path only (W_dec @ W_U), it ignores everything after the hook
layer. It is labeled as such in the UI and METHODS.md. Causal evidence comes
from the steering exhibit, which is an actual intervention.

**Why preserve the SAE error term when steering?**
Replacing the stream with the reconstruction injects the SAE's error
everywhere and degrades the model even with no edits. Delta-editing
(x += (m-1) z_f d_f) touches only the feature you asked about — so observed
changes are attributable to that feature.

## Engine

**Why write the inference engine from scratch?**
The point of the project: demonstrate the whole stack is understood. Also
enables the zero-backend demo — a 93KB WASM binary + 12MB weights runs
everything client-side.

**How do you know the C++ is correct?**
Logit parity against PyTorch on committed fixture vectors: contract 1e-3,
measured 1.8e-07 max abs diff, in CI. Plus exact tokenizer-id parity and SAE
parity. The fixture never retrains in CI, so drift in either implementation
fails the build.

**KV cache: what's stored and what's the cost?**
Per layer, the post-RoPE keys and values for every past position:
2 x n_layers x ctx x d_model floats (4 x 256 x 192 -> 786k floats ~ 3MB).
Each new token is O(pos) attention + O(d^2) matvecs.

**Sources of Python/C++ divergence you accepted?**
(1) summation order in fp32 (harmless, 1e-7-scale); (2) unicode-whitespace
pre-tokenization (documented; unreachable for TinyStories text);
(3) top-k tie-breaking in the SAE (measure zero). All in METHODS.md.

## Process

**What broke during the build?**
The honest list: paragraph-vs-story separator bug packed EOT mid-story
(caught by doc-count sanity check, D004); GrokModel missing hook utilities
(caught by mypy before runtime); an NBSP in a test string (caught by ruff,
ASCII rule); PyTorch thread oversubscription causing 10x slowdowns (fixed
with explicit thread budgets, D005); a self-matching pkill that killed its
own launcher shell. All in DECISIONS.md / phase reports.
