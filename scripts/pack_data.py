"""Tokenize a text corpus into the flat uint16 binary used for training.

Parallelized across processes; each doc is encoded independently and joined
with <|endoftext|>. Writes <out>.json with provenance metadata.

    python scripts/pack_data.py --corpus data/tinystories_train.txt \
        --tokenizer data/tokenizer.json --out data/tinystories.bin
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.bpe import BPETokenizer

_tok: BPETokenizer | None = None
_tok_path: str = ""


def _init(tok_path: str) -> None:
    global _tok
    _tok = BPETokenizer.load(tok_path)


def _encode_doc(doc: str) -> list[int]:
    assert _tok is not None
    return _tok.encode(doc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    text = Path(args.corpus).read_text(encoding="utf-8")
    sep = "\n<|endofstory|>\n" if "<|endofstory|>" in text else "\n\n"
    docs = [d.strip() for d in text.split(sep) if d.strip()]
    print(f"{len(docs)} docs (sep={sep!r})")

    tok = BPETokenizer.load(args.tokenizer)
    t0 = time.time()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with Pool(args.workers, initializer=_init, initargs=(args.tokenizer,)) as pool, open(
        out_path, "wb"
    ) as f:
        for ids in pool.imap(_encode_doc, docs, chunksize=256):
            arr = np.array([*ids, tok.eot_id], dtype=np.uint16)
            arr.tofile(f)
            total += len(arr)

    meta = {
        "corpus": args.corpus,
        "tokenizer": args.tokenizer,
        "n_docs": len(docs),
        "n_tokens": total,
        "dtype": "uint16",
    }
    Path(str(out_path) + ".json").write_text(json.dumps(meta, indent=2))
    dt = time.time() - t0
    print(f"{total:,} tokens in {dt:.0f}s ({total / dt / 1e6:.2f} M tok/s)")


if __name__ == "__main__":
    main()
