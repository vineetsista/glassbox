"""Generate committed test fixtures: a tiny fixed-seed GPT, input tokens, and
expected logits. These pin forward-pass behavior in CI (training never runs in
CI) and later serve as the PyTorch side of the C++ logit-parity test.

    python scripts/make_fixtures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.config import GPTConfig
from train.gpt import GPT

FIXTURE_CFG = GPTConfig(
    vocab_size=128, d_model=32, n_layers=2, n_heads=4, d_mlp=128, ctx_len=16
)


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(20260709)
    model = GPT(FIXTURE_CFG)
    model.eval()

    gen = torch.Generator().manual_seed(7)
    tokens = torch.randint(0, FIXTURE_CFG.vocab_size, (2, 16), generator=gen)
    with torch.no_grad():
        logits = model(tokens)

    torch.save(model.state_dict(), out / "tiny_gpt_state.pt")
    torch.save(tokens, out / "tiny_gpt_tokens.pt")
    torch.save(logits, out / "tiny_gpt_logits.pt")
    print(f"fixtures written to {out}")
    print(f"logits checksum: {logits.double().abs().sum().item():.6f}")


if __name__ == "__main__":
    main()
