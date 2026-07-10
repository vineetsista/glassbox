"""Model and training configs. Tier S is the shipped default (DECISIONS.md D001)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 6
    d_mlp: int = 768  # 4 * d_model
    ctx_len: int = 256
    rope_base: float = 10000.0
    rms_eps: float = 1e-5

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


TIER_S = GPTConfig()

# Tier M kept for reference; not used on this hardware (no GPU).
TIER_M = GPTConfig(d_model=384, n_layers=6, n_heads=6, d_mlp=1536, ctx_len=512)


@dataclass
class TrainConfig:
    seed: int = 1337
    batch_size: int = 32
    grad_accum: int = 1
    lr: float = 3e-3  # small models tolerate high lr
    lr_min_frac: float = 0.1
    warmup_steps: int = 500
    max_steps: int = 60_000
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    eval_every: int = 500
    eval_batches: int = 20
    ckpt_every: int = 2000
    log_every: int = 50
