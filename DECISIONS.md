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

## 2026-07-09 — D004: Story separator sentinel

First pack of TinyStories used blank lines as story separators, but stories
contain internal blank lines between paragraphs -> 2.47M "docs" from 465k
stories, i.e. <|endoftext|> was landing mid-story. Fixed by separating stories
with an explicit sentinel line ("<|endofstory|>") at fetch time; pack_data
splits on it and never tokenizes it. The BPE tokenizer (trained on clean story
text before the sentinel existed) is unaffected and was NOT retrained;
train_tokenizer.py now strips the sentinel for future regeneration.

## 2026-07-09 — D003: File authoring path

Repo lives in the WSL ext4 filesystem (~/projects/glassbox), never /mnt/c or
OneDrive, per brief. Files are authored through the \\wsl.localhost UNC bridge;
all builds/tests/training run natively inside WSL.
