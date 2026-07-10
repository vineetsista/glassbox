"""The grokking specimen: a 1-layer transformer for modular addition.

Setup follows Power et al. 2022 / Nanda et al. 2023: inputs are sequences
[a, b, =] over vocab {0..p-1, '='}, target is (a + b) mod p at the last
position. Learned positional embeddings, ReLU MLP, NO LayerNorm (keeps the
Fourier analysis clean), separate unembedding.

Reuses the HookPoint system from train.gpt — same observe/intervene contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from train.gpt import HookPoint


@dataclass
class GrokConfig:
    p: int = 113
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 512
    seq_len: int = 3  # a b =

    @property
    def vocab_size(self) -> int:
        return self.p + 1  # numbers + '='

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


class GrokModel(nn.Module):
    def __init__(self, cfg: GrokConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, h, dh = cfg.d_model, cfg.n_heads, cfg.d_head
        self.wte = nn.Embedding(cfg.vocab_size, d)
        self.wpe = nn.Embedding(cfg.seq_len, d)
        self.w_q = nn.Linear(d, d, bias=False)
        self.w_k = nn.Linear(d, d, bias=False)
        self.w_v = nn.Linear(d, d, bias=False)
        self.w_o = nn.Linear(d, d, bias=False)
        self.w_in = nn.Linear(d, cfg.d_mlp, bias=False)
        self.w_out = nn.Linear(cfg.d_mlp, d, bias=False)
        self.unembed = nn.Linear(d, cfg.vocab_size, bias=False)

        self.hook_embed = HookPoint()
        self.hook_pattern = HookPoint()   # [B, H, T, T]
        self.hook_z = HookPoint()         # [B, H, T, d_head]
        self.hook_resid_mid = HookPoint() # [B, T, d_model]
        self.hook_mlp_act = HookPoint()   # [B, T, d_mlp] post-ReLU
        self.hook_resid_post = HookPoint()

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=1.0 / math.sqrt(d))
        for name, module in self.named_modules():
            if isinstance(module, HookPoint):
                module.name = name

        self._h, self._dh = h, dh

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B, 3] -> logits [B, vocab] at the final position."""
        b, t = tokens.shape
        h, dh = self._h, self._dh
        pos = torch.arange(t, device=tokens.device)
        x = self.hook_embed(self.wte(tokens) + self.wpe(pos))

        def split(v: torch.Tensor) -> torch.Tensor:
            return v.view(b, t, h, dh).transpose(1, 2)

        q, k, v = split(self.w_q(x)), split(self.w_k(x)), split(self.w_v(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(dh)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        pattern = self.hook_pattern(F.softmax(scores, dim=-1))
        z = self.hook_z(pattern @ v)
        attn_out = self.w_o(z.transpose(1, 2).reshape(b, t, -1))
        x = self.hook_resid_mid(x + attn_out)
        act = self.hook_mlp_act(F.relu(self.w_in(x)))
        x = self.hook_resid_post(x + self.w_out(act))
        return self.unembed(x[:, -1, :])


def all_pairs_dataset(p: int, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Every (a, b) pair: tokens [p*p, 3] with '=' = p, labels [p*p]."""
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    eq = torch.full_like(a, p)
    tokens = torch.stack([a, b, eq], dim=1).to(device)
    labels = ((a + b) % p).to(device)
    return tokens, labels


def train_test_split(
    p: int, train_frac: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic boolean masks over the p*p pairs."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(p * p, generator=g)
    n_train = int(train_frac * p * p)
    train_mask = torch.zeros(p * p, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    return train_mask, ~train_mask
