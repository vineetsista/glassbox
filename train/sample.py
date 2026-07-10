"""Deterministic sampling from a GPT checkpoint. Used for the honest gallery."""

from __future__ import annotations

import torch

from .bpe import BPETokenizer
from .gpt import GPT

# Fixed prompts for the sample gallery; never change these after first publication,
# so galleries across checkpoints are comparable.
EVAL_PROMPTS: list[str] = [
    "Once upon a time",
    "The little dog",
    "Tom and Lily went to the",
    "One day, a girl named",
    "The big red ball",
]


@torch.no_grad()
def generate(
    model: GPT,
    tok: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int = 40,
    seed: int = 0,
) -> str:
    device = next(model.parameters()).device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ids = tok.encode(prompt)
    for _ in range(max_new_tokens):
        window = ids[-model.cfg.ctx_len :]
        x = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(x)[0, -1].float().cpu()
        if temperature <= 0:
            next_id = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k > 0:
                thresh = torch.topk(logits, top_k).values[-1]
                logits[logits < thresh] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, 1, generator=gen))
        if next_id == tok.eot_id:
            break
        ids.append(next_id)
    return tok.decode(ids)
