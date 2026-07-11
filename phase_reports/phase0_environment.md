# Phase 0 — environment + scaffold

Date: 2026-07-09

## Done
- Hardware detect: WSL2 Ubuntu 24.04, i7-1250U (2P+8E, 12 threads), 7.6GiB RAM
  visible, no GPU -> MODEL_TIER=S (DECISIONS.md D001, D005).
- Toolchain: g++ 13.3, CMake 3.28, Ninja 1.11, Python 3.12 venv (torch
  2.13.0+cpu), Node 20.20, emsdk (emcc 6.0.2) installed to ~/tools/emsdk.
- Repo at ~/projects/glassbox (WSL ext4, never /mnt/c), git from commit one.
- ruff + mypy configured (pyproject.toml), both clean from the first commit.
- `gh` CLI unavailable (no sudo password for apt); push instructions deferred
  to the final report.

## Surprises
- Bash tool quoting through wsl.exe eats `$var` in double quotes; everything
  runs via single-quoted `bash -lc` or script files.
- No sudo -> everything user-space; nothing actually blocked on it.
