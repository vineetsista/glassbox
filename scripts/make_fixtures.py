"""Generate committed test fixtures pinning behavior across Python and C++:

- tiny fixed-seed GPT: state dict + input tokens + expected logits (.pt, for
  Python CI) and the same as raw binaries + GBX (for C++ parity tests)
- tiny BPE tokenizer trained on fixed text + encodings of fixed strings
- tiny SAE: state + expected sparse codes for fixed inputs

Training never runs in CI; these fixtures are the contract.

    python scripts/make_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sae.model import SAEConfig, TopKSAE
from scripts.export_gbx import export_gbx
from train.bpe import train_bpe
from train.config import GPTConfig
from train.gpt import GPT

FIXTURE_CFG = GPTConfig(
    vocab_size=300, d_model=32, n_layers=2, n_heads=4, d_mlp=128, ctx_len=16
)

FIXTURE_SAE_CFG = SAEConfig(d_in=32, expansion=4, k=8)

TOKENIZER_TEXT = (
    "Once upon a time there was a little girl named Lily. "
    "She liked to play in the park with her dog. "
    "The dog was big and red and very happy. "
) * 60

ENCODE_SAMPLES = [
    "Once upon a time, the dog played happily.",
    "hello world",
    "cafe éèê unicode こんにちは",
    "  spaces\t\ttabs\n\nnewlines  ",
]


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)

    # --- tiny GPT -----------------------------------------------------
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
    tokens.numpy().astype("<u2").tofile(out / "tiny_gpt_tokens.u16")
    logits.numpy().astype("<f4").tofile(out / "tiny_gpt_logits.f32")

    # --- tiny tokenizer -------------------------------------------------
    tok = train_bpe(TOKENIZER_TEXT, vocab_size=FIXTURE_CFG.vocab_size)
    tok.save(out / "tiny_tokenizer.json")
    enc = {s: tok.encode(s) for s in ENCODE_SAMPLES}
    (out / "tiny_encodings.json").write_text(json.dumps(enc, ensure_ascii=True))

    # --- tiny SAE -------------------------------------------------------
    torch.manual_seed(31337)
    sae = TopKSAE(FIXTURE_SAE_CFG)
    sae.eval()
    gen2 = torch.Generator().manual_seed(11)
    x = torch.randn(4, FIXTURE_SAE_CFG.d_in, generator=gen2)
    with torch.no_grad():
        z = sae.encode(x)
        xhat = sae.decode(z)
    torch.save(sae.state_dict(), out / "tiny_sae_state.pt")
    x.numpy().astype("<f4").tofile(out / "tiny_sae_input.f32")
    z.numpy().astype("<f4").tofile(out / "tiny_sae_z.f32")
    xhat.numpy().astype("<f4").tofile(out / "tiny_sae_xhat.f32")

    # --- GBX bundle (model + tokenizer + sae) for the C++ engine --------
    export_gbx(
        model, tok, out / "tiny_model.gbx",
        sae_state=sae.state_dict(),
        sae_cfg={"d_in": 32, "expansion": 4, "k": 8},
        sae_hook_layer=1,
    )

    print(f"fixtures written to {out}")
    print(f"logits checksum: {logits.double().abs().sum().item():.6f}")


if __name__ == "__main__":
    main()
