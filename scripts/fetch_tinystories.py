"""Download TinyStories via `datasets` and write a plain-text corpus.

Output: data/tinystories_train.txt capped at --max-mb, plus
data/tinystories_val.txt from the validation split. Stories contain internal
blank lines, so they are separated by an explicit sentinel line (SEP) rather
than by blank lines. `datasets` is allowed purely for download (brief sec. 2).

Fallback if HF is unreachable: scripts/fetch_gutenberg_kids.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

SEP = "\n<|endofstory|>\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-mb", type=int, default=400, help="cap on train text size")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    from datasets import load_dataset

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    budget = args.max_mb * 1024 * 1024
    written = 0
    n_docs = 0
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    with open(out_dir / "tinystories_train.txt", "w", encoding="utf-8") as f:
        for row in ds:
            text = row["text"].strip()
            if not text:
                continue
            chunk = text + SEP
            f.write(chunk)
            written += len(chunk.encode("utf-8"))
            n_docs += 1
            if written >= budget:
                break
    print(f"train: {n_docs} docs, {written / 1e6:.1f} MB")

    val = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    vw = 0
    with open(out_dir / "tinystories_val.txt", "w", encoding="utf-8") as f:
        for row in val:
            text = row["text"].strip()
            if not text:
                continue
            chunk = text + SEP
            f.write(chunk)
            vw += len(chunk.encode("utf-8"))
            if vw >= 20 * 1024 * 1024:
                break
    print(f"val: {vw / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
