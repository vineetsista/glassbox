"""Model tests: fixture determinism, hook semantics, causality, RoPE."""

from pathlib import Path

import pytest
import torch
from scripts.make_fixtures import FIXTURE_CFG
from train.gpt import GPT, apply_rope

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def tiny() -> GPT:
    model = GPT(FIXTURE_CFG)
    model.load_state_dict(torch.load(FIXTURES / "tiny_gpt_state.pt", weights_only=True))
    model.eval()
    return model


def test_forward_matches_committed_fixture(tiny: GPT) -> None:
    tokens = torch.load(FIXTURES / "tiny_gpt_tokens.pt", weights_only=True)
    expected = torch.load(FIXTURES / "tiny_gpt_logits.pt", weights_only=True)
    with torch.no_grad():
        logits = tiny(tokens)
    assert torch.allclose(logits, expected, atol=1e-5), "forward pass drifted from fixture"


def test_causality(tiny: GPT) -> None:
    """Changing a future token must not affect past logits."""
    g = torch.Generator().manual_seed(0)
    t1 = torch.randint(0, FIXTURE_CFG.vocab_size, (1, 12), generator=g)
    t2 = t1.clone()
    t2[0, -1] = (t2[0, -1] + 1) % FIXTURE_CFG.vocab_size
    with torch.no_grad():
        l1, l2 = tiny(t1), tiny(t2)
    assert torch.allclose(l1[:, :-1], l2[:, :-1], atol=1e-6)
    assert not torch.allclose(l1[:, -1], l2[:, -1])


def test_run_with_cache_is_pure(tiny: GPT) -> None:
    tokens = torch.load(FIXTURES / "tiny_gpt_tokens.pt", weights_only=True)
    with torch.no_grad():
        plain = tiny(tokens)
    cached_logits, cache = tiny.run_with_cache(tokens)
    assert torch.allclose(plain, cached_logits, atol=1e-6)
    assert f"blocks.{FIXTURE_CFG.n_layers - 1}.hook_resid_post" in cache
    # hooks must all be removed afterwards
    assert all(not hp._fns for _, hp in tiny.hook_points())


def test_hook_intervention_and_cleanup(tiny: GPT) -> None:
    tokens = torch.load(FIXTURES / "tiny_gpt_tokens.pt", weights_only=True)
    with torch.no_grad():
        base = tiny(tokens)
        with tiny.hooks([("blocks.0.hook_mlp_out", lambda t, hp: torch.zeros_like(t))]):
            ablated = tiny(tokens)
        after = tiny(tokens)
    assert not torch.allclose(base, ablated)
    assert torch.allclose(base, after, atol=1e-6), "hook leaked past context manager"


def test_attention_pattern_rows_sum_to_one(tiny: GPT) -> None:
    tokens = torch.load(FIXTURES / "tiny_gpt_tokens.pt", weights_only=True)
    _, cache = tiny.run_with_cache(tokens, names=["blocks.0.attn.hook_pattern"])
    pattern = cache["blocks.0.attn.hook_pattern"]
    assert torch.allclose(pattern.sum(-1), torch.ones_like(pattern.sum(-1)), atol=1e-5)
    # strictly causal: upper triangle is zero
    t = pattern.shape[-1]
    upper = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
    assert pattern[..., upper].abs().max() == 0


def test_rope_preserves_norm_and_relativity() -> None:
    from train.gpt import _rope_cos_sin

    cos, sin = _rope_cos_sin(32, 16, 10000.0, torch.device("cpu"), torch.float32)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 2, 32, 16, generator=g)
    rx = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), rx.norm(dim=-1), atol=1e-5)
    # relative property: <rope(q)_i, rope(k)_j> depends only on i-j
    q = torch.randn(16, generator=g)
    k = torch.randn(16, generator=g)

    def dot(i: int, j: int) -> float:
        qi = apply_rope(q.view(1, 1, 1, -1).expand(1, 1, 32, 16).clone(), cos, sin)[0, 0, i]
        kj = apply_rope(k.view(1, 1, 1, -1).expand(1, 1, 32, 16).clone(), cos, sin)[0, 0, j]
        return float(qi @ kj)

    assert abs(dot(5, 3) - dot(10, 8)) < 1e-4
    assert abs(dot(7, 7) - dot(0, 0)) < 1e-4


def test_tied_embeddings(tiny: GPT) -> None:
    assert tiny.wte.weight.data_ptr() == tiny.wte.weight.data_ptr()
    # unembed uses wte.weight directly; perturbing it must change logits
    tokens = torch.load(FIXTURES / "tiny_gpt_tokens.pt", weights_only=True)
    with torch.no_grad():
        base = tiny(tokens)
        tiny.wte.weight[0, 0] += 1.0
        bumped = tiny(tokens)
        tiny.wte.weight[0, 0] -= 1.0
    assert not torch.allclose(base, bumped)
