"""Token packing and loading.

Documents are tokenized, joined with <|endoftext|>, and packed into one flat
uint16 binary (vocab 4096 fits comfortably). Training samples are random
ctx_len+1 windows; a held-out tail of the file serves as validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def pack_tokens(token_stream: list[list[int]], eot_id: int, out_path: str | Path) -> int:
    """Write docs as one uint16 file, each doc followed by EOT. Returns token count."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out_path, "wb") as f:
        for ids in token_stream:
            arr = np.array([*ids, eot_id], dtype=np.uint16)
            arr.tofile(f)
            total += len(arr)
    return total


class PackedDataset:
    """Random-window sampler over a memmapped token file. Deterministic given seed."""

    def __init__(self, path: str | Path, ctx_len: int, val_frac: float = 0.01) -> None:
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.ctx_len = ctx_len
        n = len(self.tokens)
        self.val_start = int(n * (1.0 - val_frac))
        meta_path = Path(str(path) + ".json")
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    def batch(
        self, batch_size: int, rng: np.random.Generator, split: str = "train"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lo, hi = (0, self.val_start) if split == "train" else (self.val_start, len(self.tokens))
        starts = rng.integers(lo, hi - self.ctx_len - 1, size=batch_size)
        rows = np.stack([self.tokens[s : s + self.ctx_len + 1] for s in starts]).astype(np.int64)
        chunk = torch.from_numpy(rows)
        return chunk[:, :-1], chunk[:, 1:]
