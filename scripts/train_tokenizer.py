"""Train the byte-level BPE tokenizer on <=50MB of TinyStories text.

    python scripts/train_tokenizer.py --corpus data/tinystories_train.txt \
        --out data/tokenizer.json --train-mb 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.bpe import train_bpe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="data/tokenizer.json")
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--train-mb", type=int, default=50)
    args = ap.parse_args()

    budget = args.train_mb * 1024 * 1024
    with open(args.corpus, encoding="utf-8") as f:
        text = f.read(budget)
    # avoid a torn utf-8 char / word at the cut point
    text = text[: text.rfind("\n")]
    print(f"training BPE on {len(text.encode('utf-8')) / 1e6:.1f} MB, vocab {args.vocab_size}")

    t0 = time.time()
    tok = train_bpe(text, vocab_size=args.vocab_size, verbose=True)
    tok.save(args.out)
    print(f"done in {time.time() - t0:.0f}s -> {args.out}")

    demo = "Once upon a time, there was a little girl named Lily."
    ids = tok.encode(demo)
    print(f"demo: {len(demo)} chars -> {len(ids)} tokens")
    assert tok.decode(ids) == demo


if __name__ == "__main__":
    main()
