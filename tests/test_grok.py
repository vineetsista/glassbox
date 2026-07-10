"""Unit tests for the grok model and Fourier helpers (fast, CI-safe)."""

import torch
from grok.fourier import (
    embedding_dft_norms,
    key_frequencies,
    restricted_excluded_losses,
)
from grok.model import GrokConfig, GrokModel, all_pairs_dataset, train_test_split

SMALL = GrokConfig(p=11, d_model=32, n_heads=2, d_mlp=64)


def test_all_pairs_shapes_and_labels() -> None:
    tokens, labels = all_pairs_dataset(SMALL.p)
    assert tokens.shape == (121, 3)
    assert (tokens[:, 2] == SMALL.p).all()
    i = 5 * SMALL.p + 7
    assert tokens[i, 0] == 5 and tokens[i, 1] == 7
    assert labels[i] == (5 + 7) % SMALL.p


def test_split_deterministic_and_disjoint() -> None:
    tr1, te1 = train_test_split(SMALL.p, 0.3, seed=1)
    tr2, _te2 = train_test_split(SMALL.p, 0.3, seed=1)
    assert (tr1 == tr2).all()
    assert not (tr1 & te1).any()
    assert (tr1 | te1).all()
    assert abs(tr1.float().mean().item() - 0.3) < 0.01


def test_model_forward_shape_and_determinism() -> None:
    torch.manual_seed(0)
    model = GrokModel(SMALL)
    tokens, _ = all_pairs_dataset(SMALL.p)
    with torch.no_grad():
        a = model(tokens[:16])
        b = model(tokens[:16])
    assert a.shape == (16, SMALL.vocab_size)
    assert torch.allclose(a, b)


def test_embedding_dft_finds_planted_frequency() -> None:
    p = 23
    d = 8
    k_true = 4
    t = torch.arange(p).float()
    w_e = torch.zeros(p, d)
    w_e[:, 0] = torch.cos(2 * torch.pi * k_true * t / p)
    w_e[:, 1] = torch.sin(2 * torch.pi * k_true * t / p)
    w_e += 0.01 * torch.randn(p, d, generator=torch.Generator().manual_seed(0))
    norms = embedding_dft_norms(w_e, p)
    assert int(norms[1:].argmax()) + 1 == k_true
    assert key_frequencies(w_e, p, top_k=1) == [k_true]


def test_restricted_excluded_on_perfect_periodic_logits() -> None:
    """Logits built purely from key-frequency waves: restricted loss must equal
    full loss, and excluding the key freqs must destroy performance."""
    p = 11
    kf = [2]
    a = torch.arange(p)[:, None, None].float()
    b = torch.arange(p)[None, :, None].float()
    c = torch.arange(p)[None, None, :].float()
    w = 2 * torch.pi * kf[0] / p
    logits = 5.0 * torch.cos(w * (a + b - c))  # maximized at c = a+b mod p
    train_mask, test_mask = train_test_split(p, 0.3, seed=0)
    out = restricted_excluded_losses(logits, kf, train_mask, test_mask)
    assert abs(out["restricted_test"] - out["full_test"]) < 1e-4
    assert out["excluded_train"] > out["full_train"] + 1.0
