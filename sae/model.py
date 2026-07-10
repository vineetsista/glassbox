"""Top-k sparse autoencoder, written from scratch.

    z    = topk(relu(W_enc (x - b_dec) + b_enc), k)
    xhat = W_dec z + b_dec

Decoder columns are constrained to unit norm (renormalized after each
optimizer step). Sparsity is structural (exactly k active features per
token), so the interesting metrics are FVU and dead-feature count, not L0.

Dead features are revived by resampling (Anthropic's recipe, adapted):
periodically, features that have not fired in `dead_steps` batches get their
decoder column pointed at a high-reconstruction-error input, encoder row
scaled to match, and their Adam state reset.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SAEConfig:
    d_in: int = 192
    expansion: int = 8
    k: int = 32

    @property
    def n_features(self) -> int:
        return self.d_in * self.expansion


class TopKSAE(nn.Module):
    def __init__(self, cfg: SAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        f, d = cfg.n_features, cfg.d_in
        self.w_enc = nn.Parameter(torch.empty(f, d))
        self.b_enc = nn.Parameter(torch.zeros(f))
        self.w_dec = nn.Parameter(torch.empty(d, f))
        self.b_dec = nn.Parameter(torch.zeros(d))
        # init decoder to random unit columns, encoder to its transpose
        nn.init.normal_(self.w_dec)
        with torch.no_grad():
            self.w_dec.div_(self.w_dec.norm(dim=0, keepdim=True))
            self.w_enc.copy_(self.w_dec.T)

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-top-k feature activations: relu(W_enc (x - b_dec) + b_enc)."""
        return torch.relu((x - self.b_dec) @ self.w_enc.T + self.b_enc)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse activations: only the top k per row survive."""
        pre = self.encode_pre(x)
        vals, idx = pre.topk(self.cfg.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, idx, vals)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.w_dec.T + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    @torch.no_grad()
    def renorm_decoder(self) -> None:
        self.w_dec.div_(self.w_dec.norm(dim=0, keepdim=True).clamp_min(1e-8))


def fvu(x: torch.Tensor, xhat: torch.Tensor) -> torch.Tensor:
    """Fraction of variance unexplained (0 = perfect, 1 = no better than mean)."""
    resid = (x - xhat).pow(2).sum()
    total = (x - x.mean(dim=0, keepdim=True)).pow(2).sum()
    return resid / total.clamp_min(1e-9)
