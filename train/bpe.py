"""Byte-level BPE, written from scratch.

Trainer and codec. No external tokenizer libraries. Ids 0..255 are raw bytes,
ids 256..vocab_size-2 are learned merges in rank order, and the final id is the
<|endoftext|> document separator. Byte-level means any str round-trips exactly:
encode -> decode is the identity on arbitrary unicode.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# GPT-2-flavored pre-tokenizer, restricted to stdlib `re`. Contractions, letter
# runs, digit runs, punctuation runs (with leading space), whitespace runs.
# Non-ASCII bytes land in the "other" class; correctness never depends on the
# split because every chunk is encoded from its raw utf-8 bytes.
_PRETOKEN_RE = re.compile(
    r"'(?:[sdmt]|ll|ve|re)"
    r"| ?[A-Za-z]+"
    r"| ?[0-9]+"
    r"| ?[^\sA-Za-z0-9]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def pretokenize(text: str) -> list[str]:
    return _PRETOKEN_RE.findall(text)


@dataclass
class BPETokenizer:
    """Codec over a learned merge list. Load with .load(), build with train_bpe()."""

    merges: list[tuple[int, int]]  # rank order; merge i produces id 256+i
    vocab_size: int

    _ranks: dict[tuple[int, int], int] = field(init=False, repr=False)
    _token_bytes: list[bytes] = field(init=False, repr=False)
    _cache: dict[str, list[int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        assert self.vocab_size == 256 + len(self.merges) + 1, "vocab = bytes + merges + EOT"
        self._ranks = {pair: i for i, pair in enumerate(self.merges)}
        toks: list[bytes] = [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            toks.append(toks[a] + toks[b])
        toks.append(b"<|endoftext|>")  # display only; never produced by encode()
        self._token_bytes = toks
        self._cache = {}

    @property
    def eot_id(self) -> int:
        return self.vocab_size - 1

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            best_rank = None
            best_i = -1
            for i in range(len(ids) - 1):
                r = self._ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            ids[best_i : best_i + 2] = [256 + best_rank]
        if len(self._cache) < 1_000_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in pretokenize(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: list[int]) -> str:
        tb = self._token_bytes
        data = b"".join(tb[i] for i in ids if i != self.eot_id)
        return data.decode("utf-8", errors="replace")

    def token_str(self, tid: int) -> str:
        """Human-readable form of one token (lossy; for dashboards)."""
        return self._token_bytes[tid].decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        obj = {
            "format": "glassbox-bpe-v1",
            "vocab_size": self.vocab_size,
            "merges": [[a, b] for a, b in self.merges],
        }
        Path(path).write_text(json.dumps(obj), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        assert obj["format"] == "glassbox-bpe-v1"
        merges = [(int(a), int(b)) for a, b in obj["merges"]]
        return cls(merges=merges, vocab_size=obj["vocab_size"])


def train_bpe(text: str, vocab_size: int, verbose: bool = False) -> BPETokenizer:
    """Learn merges from `text`. Classic word-frequency BPE with incremental
    pair-count updates: pre-tokenize, count unique chunks, merge the most
    frequent adjacent id pair until we have vocab_size - 257 merges.
    """
    n_merges = vocab_size - 257
    assert n_merges > 0

    chunk_freq = Counter(pretokenize(text))
    # words: list of (ids, freq)
    words: list[list[int]] = []
    freqs: list[int] = []
    for chunk, f in chunk_freq.items():
        words.append(list(chunk.encode("utf-8")))
        freqs.append(f)

    # pair -> total count; pair -> set of word indices containing it
    pair_counts: Counter[tuple[int, int]] = Counter()
    pair_words: dict[tuple[int, int], set[int]] = {}
    for wi, ids in enumerate(words):
        f = freqs[wi]
        for a, b in zip(ids, ids[1:]):
            pair_counts[(a, b)] += f
            pair_words.setdefault((a, b), set()).add(wi)

    merges: list[tuple[int, int]] = []
    for rank in range(n_merges):
        if not pair_counts:
            break
        # deterministic tie-break: highest count, then lowest pair ids
        best = min(pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        new_id = 256 + rank
        merges.append(best)
        affected = pair_words.pop(best, set())
        pair_counts.pop(best, None)
        for wi in affected:
            ids = words[wi]
            f = freqs[wi]
            i = 0
            while i < len(ids) - 1:
                if ids[i] == best[0] and ids[i + 1] == best[1]:
                    if i > 0:
                        _dec(pair_counts, pair_words, (ids[i - 1], ids[i]), f, wi, words)
                    if i + 2 < len(ids):
                        _dec(pair_counts, pair_words, (ids[i + 1], ids[i + 2]), f, wi, words)
                    ids[i : i + 2] = [new_id]
                    if i > 0:
                        _inc(pair_counts, pair_words, (ids[i - 1], ids[i]), f, wi)
                    if i + 1 < len(ids):
                        _inc(pair_counts, pair_words, (ids[i], ids[i + 1]), f, wi)
                else:
                    i += 1
        if verbose and (rank + 1) % 500 == 0:
            print(f"  merge {rank + 1}/{n_merges}")

    return BPETokenizer(merges=merges, vocab_size=256 + len(merges) + 1)


def _dec(
    pc: Counter[tuple[int, int]],
    pw: dict[tuple[int, int], set[int]],
    pair: tuple[int, int],
    f: int,
    wi: int,
    words: list[list[int]],
) -> None:
    pc[pair] -= f
    if pc[pair] <= 0:
        del pc[pair]
        pw.pop(pair, None)
    else:
        ids = words[wi]
        # keep the reverse index conservative: drop wi only if pair vanished from word
        s = pw.get(pair)
        if s is not None and not _contains_pair(ids, pair):
            s.discard(wi)


def _inc(
    pc: Counter[tuple[int, int]],
    pw: dict[tuple[int, int], set[int]],
    pair: tuple[int, int],
    f: int,
    wi: int,
) -> None:
    pc[pair] += f
    pw.setdefault(pair, set()).add(wi)


def _contains_pair(ids: list[int], pair: tuple[int, int]) -> bool:
    return any(a == pair[0] and b == pair[1] for a, b in zip(ids, ids[1:]))
