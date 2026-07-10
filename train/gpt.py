"""From-scratch decoder-only transformer with a first-class hook system.

Architecture: RoPE, pre-norm RMSNorm, GELU MLP, tied embeddings, no biases,
dropout 0. PyTorch is used only as the numerical substrate (tensors, autograd);
attention, RoPE, norms, and the hook machinery are written here by hand.

Hook points (names as exposed by GPT.hook_points()):
    hook_embed
    blocks.{i}.hook_resid_pre
    blocks.{i}.attn.hook_q | hook_k | hook_v          [B, H, T, d_head]
    blocks.{i}.attn.hook_pattern                       [B, H, T, T] post-softmax
    blocks.{i}.attn.hook_z                             [B, H, T, d_head]
    blocks.{i}.hook_attn_out
    blocks.{i}.hook_resid_mid
    blocks.{i}.hook_mlp_act                            [B, T, d_mlp] post-GELU
    blocks.{i}.hook_mlp_out
    blocks.{i}.hook_resid_post
    hook_final_norm

A hook fn has signature fn(tensor, hook) -> tensor | None. Returning a tensor
replaces the value (interventions); returning None observes only.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig

HookFn = Callable[[torch.Tensor, "HookPoint"], torch.Tensor | None]


class HookPoint(nn.Module):
    """Identity module that hook fns can observe or rewrite in place."""

    def __init__(self) -> None:
        super().__init__()
        self.name: str = ""  # filled in by GPT.__init__
        self._fns: list[HookFn] = []

    def add(self, fn: HookFn) -> None:
        self._fns.append(fn)

    def clear(self) -> None:
        self._fns.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fn in self._fns:
            out = fn(x, self)
            if out is not None:
                x = out
        return x


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def _rope_cos_sin(
    ctx_len: int, d_head: int, base: float, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    t = torch.arange(ctx_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # [T, d_head/2]
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, d_head]; rotate consecutive even/odd pairs."""
    t = x.shape[-2]
    cos, sin = cos[:t], sin[:t]  # [T, d_head/2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.w_q = nn.Linear(d, d, bias=False)
        self.w_k = nn.Linear(d, d, bias=False)
        self.w_v = nn.Linear(d, d, bias=False)
        self.w_o = nn.Linear(d, d, bias=False)
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_pattern = HookPoint()
        self.hook_z = HookPoint()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h, dh = self.cfg.n_heads, self.cfg.d_head

        def split(v: torch.Tensor) -> torch.Tensor:
            return v.view(b, t, h, dh).transpose(1, 2)  # [B, H, T, dh]

        q = self.hook_q(apply_rope(split(self.w_q(x)), cos, sin))
        k = self.hook_k(apply_rope(split(self.w_k(x)), cos, sin))
        v = self.hook_v(split(self.w_v(x)))

        scores = q @ k.transpose(-2, -1) / math.sqrt(dh)  # [B, H, T, T]
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        pattern = self.hook_pattern(F.softmax(scores, dim=-1))
        z = self.hook_z(pattern @ v)  # [B, H, T, dh]
        z = z.transpose(1, 2).reshape(b, t, d)
        return self.w_o(z)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.w_in = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.w_out = nn.Linear(cfg.d_mlp, cfg.d_model, bias=False)
        self.hook_mlp_act = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act = self.hook_mlp_act(F.gelu(self.w_in(x), approximate="tanh"))
        return self.w_out(act)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.mlp = MLP(cfg)
        self.hook_resid_pre = HookPoint()
        self.hook_attn_out = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = self.hook_resid_pre(x)
        x = self.hook_resid_mid(x + self.hook_attn_out(self.attn(self.norm1(x), cos, sin)))
        x = self.hook_resid_post(x + self.hook_mlp_out(self.mlp(self.norm2(x))))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.hook_embed = HookPoint()
        self.hook_final_norm = HookPoint()

        cos, sin = _rope_cos_sin(
            cfg.ctx_len, cfg.d_head, cfg.rope_base, torch.device("cpu"), torch.float32
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        for module in self.blocks:  # residual-stream projections get depth-scaled init
            block = cast(Block, module)
            for w in (block.attn.w_o.weight, block.mlp.w_out.weight):
                nn.init.normal_(w, std=0.02 / math.sqrt(2 * cfg.n_layers))

        for name, module in self.named_modules():
            if isinstance(module, HookPoint):
                module.name = name

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B, T] -> logits [B, T, vocab]. Tied unembedding."""
        assert tokens.shape[1] <= self.cfg.ctx_len
        x = self.hook_embed(self.wte(tokens))
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.hook_final_norm(self.final_norm(x))
        return x @ self.wte.weight.T

    # ---- hook utilities -------------------------------------------------

    def hook_points(self) -> Iterator[tuple[str, HookPoint]]:
        for name, module in self.named_modules():
            if isinstance(module, HookPoint):
                yield name, module

    def clear_hooks(self) -> None:
        for _, hp in self.hook_points():
            hp.clear()

    @contextmanager
    def hooks(self, fns: list[tuple[str, HookFn]]) -> Iterator[None]:
        """Temporarily attach (hook_name, fn) pairs."""
        by_name = dict(self.hook_points())
        added: list[tuple[HookPoint, HookFn]] = []
        try:
            for name, fn in fns:
                hp = by_name[name]
                hp.add(fn)
                added.append((hp, fn))
            yield
        finally:
            for hp, fn in added:
                hp._fns.remove(fn)

    @torch.no_grad()
    def run_with_cache(
        self, tokens: torch.Tensor, names: list[str] | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass capturing activations. `names=None` captures everything."""
        cache: dict[str, torch.Tensor] = {}

        def make_fn(name: str) -> HookFn:
            def fn(x: torch.Tensor, hp: HookPoint) -> None:
                cache[name] = x.detach().clone()

            return fn

        wanted = [n for n, _ in self.hook_points() if names is None or n in names]
        with self.hooks([(n, make_fn(n)) for n in wanted]):
            logits = self.forward(tokens)
        return logits, cache

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel()
        return n
