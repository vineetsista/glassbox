# WALKTHROUGH — a guided ten minutes with GLASSBOX

Serve the site (`scripts/build_site.sh`, then any static server on
`web/dist`, e.g. `python -m http.server -d web/dist 8080`).

## Exhibit 1 — Watch it grok (2 min)

Drag the slider to step ~200. Train accuracy is 100%, test is under 10%: the
network has memorized 3,830 addition facts and understands nothing. Look at
the embedding spectrum: white noise across all 56 frequencies.

Now press play. Around step 2000-3000, test accuracy snaps upward to 100%.
Watch the spectrum at the same time: it collapses onto ~5 pink spikes. Those
are the key frequencies; the network has quietly replaced its lookup table
with trigonometry — cos/sin waves over Z_113, multiplied in attention,
read out with the identity cos(w(a+b-c)).

Two subtler things to show off:
- The "restricted" curve (teal, log plot) starts tracking test loss BEFORE
  accuracy moves: the general circuit grows inside the memorizing network,
  invisible to top-line metrics. This is why interpretability people care
  about progress measures.
- The weight-norm badge falls through the transition: weight decay is the
  pressure that makes the compact (general) solution win.

## Exhibit 2 — Look inside (4 min)

You are looking at every feature of a top-k sparse autoencoder trained on
the residual stream of a from-scratch TinyStories model. Sort by density.
Click around. Typical finds at this scale: features for names, for quoted
dialogue, for "the start of a story", for specific tokens after specific
contexts.

For any feature: green highlights show where in real stories it fires
strongest; the histogram shows how often and how hard; the logit-lens table
shows what it pushes the model to say next (direct path only — honest label).

Then use the live probe: type your own sentence and watch the feature light
up (or refuse to). This runs the actual model in your browser through the
C++/WASM engine — there is no server.

## Exhibit 3 — Steer it (4 min)

Pick a feature from exhibit 2 that fires on something recognizable (a name,
dialogue, an emotion). Add it as an edit. Generate: left is the model as
trained, right is the same seed with your edit applied to the residual
stream mid-network.

- Amplify (x4-x10): the concept starts intruding into the story.
- Ablate (x0): the concept gets avoided or degraded.

Expectations to set honestly: this is a 2.6M-parameter model trained for one
laptop-night. Its stories are simple and its features are correspondingly
low-level. Steering effects are real and repeatable (fixed seed!) but they
are nudges, not mind control. That gap between "the feature clearly encodes
X" and "editing it cleanly controls X" is itself the honest lesson — the
same gap exists in frontier-scale interpretability.

## If they ask "what am I actually running?"

A hand-written C++20 transformer engine compiled to WASM SIMD, loading a
12MB GBX bundle (weights + BPE tokenizer + SAE) and doing fp32 KV-cached
inference, logit-parity-tested against PyTorch to 1e-3 (measured 1.8e-07).
