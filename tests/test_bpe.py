"""Tokenizer tests, including the round-trip property test on random unicode."""

import random

import pytest
from train.bpe import BPETokenizer, pretokenize, train_bpe

TRAIN_TEXT = (
    "Once upon a time there was a little girl named Lily. "
    "She liked to play in the park with her dog. "
    "The dog was big and red and very happy. "
) * 50


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return train_bpe(TRAIN_TEXT, vocab_size=400)


def test_roundtrip_ascii(tok: BPETokenizer) -> None:
    for s in ["hello world", "Once upon a time", "", "a", "  leading spaces", "trailing  "]:
        assert tok.decode(tok.encode(s)) == s


def test_roundtrip_random_unicode(tok: BPETokenizer) -> None:
    rng = random.Random(42)
    planes = [(0x20, 0x7E), (0xA0, 0x2FF), (0x370, 0x3FF), (0x4E00, 0x4EFF), (0x1F600, 0x1F64F)]
    for _ in range(200):
        chars = []
        for _ in range(rng.randint(0, 80)):
            lo, hi = rng.choice(planes)
            chars.append(chr(rng.randint(lo, hi)))
        s = "".join(chars)
        assert tok.decode(tok.encode(s)) == s


def test_roundtrip_whitespace_torture(tok: BPETokenizer) -> None:
    for s in ["\n\n\n", "\t \t", "a\nb\r\nc", " \u00a0 ", "\r\r"]:
        assert tok.decode(tok.encode(s)) == s


def test_compression_on_domain_text(tok: BPETokenizer) -> None:
    s = "The little girl played with the happy dog in the park."
    ids = tok.encode(s)
    assert len(ids) < len(s) / 2, "trained BPE should compress in-domain text >2x"


def test_pretokenize_reconstructs() -> None:
    for s in ["Hello, world! It's 42 degrees.", "a  b   c", "don't stop"]:
        assert "".join(pretokenize(s)) == s


def test_encode_deterministic(tok: BPETokenizer) -> None:
    s = "The dog was big and red."
    assert tok.encode(s) == tok.encode(s)


def test_eot_never_produced(tok: BPETokenizer) -> None:
    ids = tok.encode("some text <|endoftext|> more text")
    assert tok.eot_id not in ids


def test_save_load_identical(tok: BPETokenizer, tmp_path) -> None:
    p = tmp_path / "tok.json"
    tok.save(p)
    tok2 = BPETokenizer.load(p)
    s = "Once upon a time, there was a dog."
    assert tok.encode(s) == tok2.encode(s)
