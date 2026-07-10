# DECISIONS.md

Running log of engineering decisions. Newest at the bottom. Every entry: date, decision, why.

## 2026-07-09 — D001: Hardware detection and MODEL_TIER

Detected inside WSL2 (Ubuntu 24.04.3, kernel 6.6.87.2-microsoft-standard-WSL2):

- CPU: 12 logical cores (nproc)
- RAM: 7.6 GiB total visible to WSL, ~6.7 GiB available
- GPU: none (`nvidia-smi` not present / no NVIDIA GPU passthrough)

Decision: **MODEL_TIER = S** per the build brief.
LM config: d_model 192, 4 layers, ctx 256, ~2-4M params.
All long training runs launch via nohup with checkpoint/resume; we report honest
CPU-scale results rather than pretending to GPU-scale quality.

## 2026-07-09 — D002: Toolchain

- g++ 13.3.0 (meets C++20 requirement; clang not installed, not needed)
- CMake 3.28.3, Ninja 1.11.1
- Python 3.12.3 (brief asks 3.11+; 3.12 OK) in a venv at .venv/
- Node 20.20.2, npm 10.8.2
- Emscripten: not present -> install emsdk under ~/tools/emsdk (outside repo)
- `gh` CLI: not installed and no auth available -> repo stays local; push
  instructions go in the final report per brief section 1.

## 2026-07-09 — D003: File authoring path

Repo lives in the WSL ext4 filesystem (~/projects/glassbox), never /mnt/c or
OneDrive, per brief. Files are authored through the \\wsl.localhost UNC bridge;
all builds/tests/training run natively inside WSL.

## 2026-07-09 — D004: Story separator sentinel

First pack of TinyStories used blank lines as story separators, but stories
contain internal blank lines between paragraphs -> 2.47M "docs" from 465k
stories, i.e. <|endoftext|> was landing mid-story. Fixed by separating stories
with an explicit sentinel line ("<|endofstory|>") at fetch time; pack_data
splits on it and never tokenizes it. The BPE tokenizer (trained on clean story
text before the sentinel existed) is unaffected and was NOT retrained;
train_tokenizer.py now strips the sentinel for future regeneration.

## 2026-07-09 — D005: Compute recalibration after measuring the actual CPU

The CPU is an i7-1250U: a 9W ultralight part, 2 P-cores + 8 E-cores. Measured
LM training throughput (tier S, batch 32, ctx 256): ~680 tok/s eager /
~970 tok/s with torch.compile at 6 threads while grok holds 4 threads.
Decisions:
- torch.compile added to train_lm (--compile, ~1.4x; hooks unused in training,
  checkpoints saved from the uncompiled module so keys stay clean).
- Attention builds its causal mask once as a buffer instead of per forward.
- Thread split while both jobs run: grok 4, LM 8 (oversubscribing 12 threads
  with 2 PyTorch processes caused ~10x slowdowns from spin contention).
- LM run sized honestly for this machine: 6000 steps x 8192 tok = 49M tokens
  (~0.5 epoch of the 105M-token pack), cosine completing at step 6000. This is
  a deliberate scale-down from the notional 60k-step config; the README will
  report the model as what it is: a ~2.6M-param CPU-trained TinyStories model.
- Every number above is from timed runs on this box, not estimates.
