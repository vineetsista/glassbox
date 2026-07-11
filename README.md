# GLASSBOX

An interactive interpretability instrument. Small transformers trained from
scratch, their internal features extracted with a sparse autoencoder, and a
zero-backend browser app where you can watch grokking unfold, inspect live
feature activations, and steer model behavior by ablating or amplifying
individual features. Inference runs client-side on a from-scratch C++20
engine compiled to WASM SIMD.

Everything here was built and measured on one 9W laptop CPU. That constraint
is documented, not hidden: every number below traces to a script, a seed,
and a log file in this repo.

## The three exhibits

1. **Watch it grok** — a 1-layer transformer learns `(a + b) mod 113` from
   30% of all pairs. Scrub through 77 dense checkpoints: it memorizes by
   step 200 (train 100%, test 6%), then at ~step 3000 test accuracy snaps to
   100% as the lookup table is replaced by trigonometry. The embedding DFT
   collapses onto 5 frequencies in front of you.
2. **Look inside** — browse all 1536 features of a top-k SAE trained on the
   residual stream of a TinyStories model: max-activating contexts,
   activation histograms, logit-lens readouts — and probe any feature live
   on your own text, in your browser.
3. **Steer it** — generate stories with the same seed twice: once as
   trained, once with your feature edits applied to the residual stream
   mid-network. Ablate a feature (x0) or amplify it (x4-x10) and watch the
   story bend.

## What "from scratch" means here

- **Transformer** (`train/gpt.py`): hand-written attention, RoPE, RMSNorm,
  hook system. PyTorch is used only as the numerical substrate (tensors,
  autograd, optimizer). No HuggingFace, no TransformerLens, no nanoGPT
  copy-paste.
- **Tokenizer** (`train/bpe.py`): byte-level BPE trainer and codec, written
  from scratch; vocab 4096 trained on 50MB of TinyStories in 46s. Exact
  round-trip on arbitrary unicode, property-tested.
- **SAE** (`sae/`): top-k sparse autoencoder (8x expansion, k=32, unit-norm
  decoder, dead-feature resampling), trained on 20M cached activations.
- **Engine** (`engine/`): C++20, zero third-party dependencies (Catch2 for
  tests only) — hand-rolled JSON parser, custom GBX weight format, BPE
  runtime, fp32 kernels, KV cache, SAE steering, deterministic sampler.
  Logit parity vs PyTorch: contract 1e-3, **measured max diff 1.8e-07**.
- **Web** (`web/`): React+Vite+TS, zero chart libraries (hand-rolled canvas
  renderers), zero backend. The WASM engine is 93KB; the model bundle 12.6MB.

## Headline results (all regenerable; see docs/BENCHMARKS.md)

- **Grokking, verified mechanistically on our weights**
  (`python -m grok.analyze`): key frequencies {9, 27, 40, 43, 45} hold 90.6%
  of embedding power; 90% of MLP neurons are single-frequency; keeping only
  key-frequency logit components gives test loss 1.1e-5 (the full model:
  5.4e-5), removing them gives 10.7. The periodic circuit isn't just present
  — it is the model.
- **LM**: 2.56M params, 49M tokens (~0.5 epoch), final val loss 1.921. The
  loss was still falling when the budget ran out; docs/figures/lm_loss.png
  is the honest curve.
- **SAE**: metrics in `runs/sae_l2/log.csv` and phase_reports/phase3_sae.md
  — FVU, dead counts, and density come straight from the training log.

## Reproduce

```bash
# WSL2/Linux, Python 3.11+, g++ 13+, CMake 3.22+, Node 20+, emsdk
python -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements-dev.txt

.venv/bin/python scripts/fetch_tinystories.py            # or fetch_gutenberg_kids.py
.venv/bin/python scripts/train_tokenizer.py --corpus data/tinystories_train.txt
.venv/bin/python scripts/pack_data.py --corpus data/tinystories_train.txt \
    --tokenizer data/tokenizer.json --out data/tinystories.bin

bash scripts/launch_lm.sh 6000 8                          # ~6h on a laptop CPU
.venv/bin/python -m grok.train_grok --run runs/grok --steps 8000
bash scripts/post_lm_pipeline.sh                          # everything downstream

cmake -S engine -B engine/build -G Ninja && ninja -C engine/build && ctest --test-dir engine/build
emcmake cmake -S engine -B engine/build-wasm -G Ninja && ninja -C engine/build-wasm
bash scripts/build_site.sh                                # -> web/dist
```

Seeds: LM 1337, grok 999, SAE 4242 (all in the configs). CI runs tests on
committed fixtures only — training never runs in CI.

## Repo map

`train/` LM + BPE + hooks - `grok/` grokking specimen + Fourier analysis -
`sae/` cache/train/dashboards - `engine/` C++ engine - `web/` the app -
`scripts/` launchers and exporters - `docs/` METHODS, GBX_FORMAT,
WALKTHROUGH, INTERVIEW_DRILL, BENCHMARKS, SAMPLES - `phase_reports/` +
`DECISIONS.md` the build log.

## Limitations — read this before being impressed

- **The LM is tiny and undertrained.** 2.56M params, half an epoch, val loss
  1.92. It writes grammatical, TinyStories-flavored text with frequent
  logical derailments (docs/SAMPLES.md shows unedited samples — the honest
  gallery). Nothing here says anything about frontier models except by
  methodological analogy.
- **SAE features at this scale are low-level.** Expect token/phrase/position
  features, names, and dialogue markers — not abstract concepts. Some
  features are dense or uninterpretable; the dashboards show them anyway.
- **Steering is a nudge, not a dial.** Effects are real and seed-reproducible
  but noisy; ablating a feature often has subtler effects than amplifying.
  The logit-lens panel is a direct-path approximation and is labeled as such.
- **Our grokking progress measures are simplified** (2D DFT of the logit
  grid, not Nanda et al.'s internal-component decomposition) — coarser but
  model-agnostic; definitions in METHODS.md section 3.
- **One seed each.** Compute budget bought one LM run, one grok run, one SAE
  run. No variance bars anywhere; treat every number as a point estimate.
- **C++/Python tokenizer divergence on exotic unicode whitespace**
  (documented in METHODS.md section 2); irrelevant for the training
  distribution, but it exists.
