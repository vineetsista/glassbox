"""Fallback corpus builder: public-domain children's literature from Project
Gutenberg, used only if HuggingFace is unreachable (brief section 3.1).

Downloads a fixed list of book IDs over plain HTTPS, strips Gutenberg headers
and footers, and writes data/gutenberg_kids_train.txt in the same blank-line-
separated format as fetch_tinystories.py.
"""

from __future__ import annotations

import argparse
import re
import urllib.request
from pathlib import Path

# Well-known public-domain children's books (Gutenberg IDs).
BOOK_IDS = [
    11,     # Alice in Wonderland
    16,     # Peter Pan
    55,     # Wizard of Oz
    113,    # The Secret Garden
    236,    # The Jungle Book
    271,    # Black Beauty
    289,    # The Wind in the Willows
    501,    # Just So Stories
    902,    # The Happy Prince
    1874,   # English Fairy Tales
    2591,   # Grimms' Fairy Tales
    7439,   # A Child's Garden of Verses? (kept if fetch succeeds)
    16389,  # The Enchanted Castle
    17034,  # The Water-Babies
    19993,  # The Velveteen Rabbit
]

_START = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.S | re.I)
_END = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG.*", re.S | re.I)


def fetch_book(book_id: int) -> str | None:
    for url in (
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
    ):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                raw = r.read().decode("utf-8", errors="replace")
            body = _START.split(raw, maxsplit=1)[-1]
            body = _END.split(body, maxsplit=1)[0]
            return body.strip()
        except Exception:
            continue
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(out_dir / "gutenberg_kids_train.txt", "w", encoding="utf-8") as f:
        for bid in BOOK_IDS:
            body = fetch_book(bid)
            if body is None:
                print(f"  book {bid}: FAILED, skipping")
                continue
            # paragraphs -> blank-line-separated docs of a few paragraphs each
            paras = [p.strip() for p in body.split("\n\n") if p.strip()]
            for i in range(0, len(paras), 4):
                doc = " ".join(" ".join(p.split()) for p in paras[i : i + 4])
                f.write(doc + "\n\n")
                total += len(doc)
            print(f"  book {bid}: ok ({len(body) / 1e6:.1f} MB)")
    print(f"total {total / 1e6:.1f} MB -> gutenberg_kids_train.txt")


if __name__ == "__main__":
    main()
