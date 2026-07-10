"""Export a trained LM (+tokenizer, +optional SAE) into the GBX weight format
consumed by the C++ engine (native and WASM). See docs/GBX_FORMAT.md.

Layout (little-endian):
    bytes 0..3   magic "GBX1"
    u32          header_len (bytes of JSON)
    bytes        header JSON (utf-8)
    bytes        zero padding to 64-byte alignment
    bytes        tensor data, each tensor 64-byte aligned, fp32 row-major

Header JSON:
    {"config": {...GPTConfig...},
     "tokenizer": {"merges": [[a,b],...], "vocab_size": N},
     "sae": {"d_in":..,"k":..,"n_features":..} | null,
     "tensors": [{"name","shape","offset","nbytes"}...]}

    python scripts/export_gbx.py --ckpt runs/lm_s/latest.pt \
        --tokenizer data/tokenizer.json --sae runs/sae_l2/latest.pt \
        --out web/public/assets/model.gbx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.bpe import BPETokenizer  # noqa: E402
from train.config import GPTConfig  # noqa: E402
from train.gpt import GPT  # noqa: E402

ALIGN = 64


def export_gbx(
    model: GPT,
    tok: BPETokenizer,
    out_path: str | Path,
    sae_state: dict | None = None,
    sae_cfg: dict | None = None,
    sae_hook_layer: int = 2,
) -> None:
    cfg = model.cfg
    tensors: list[tuple[str, np.ndarray]] = []

    sd = {k: v.detach().float().numpy() for k, v in model.state_dict().items()}
    tensors.append(("wte", sd["wte.weight"]))
    for i in range(cfg.n_layers):
        pre = f"blocks.{i}."
        tensors.append((f"blocks.{i}.norm1", sd[pre + "norm1.weight"]))
        tensors.append((f"blocks.{i}.w_q", sd[pre + "attn.w_q.weight"]))
        tensors.append((f"blocks.{i}.w_k", sd[pre + "attn.w_k.weight"]))
        tensors.append((f"blocks.{i}.w_v", sd[pre + "attn.w_v.weight"]))
        tensors.append((f"blocks.{i}.w_o", sd[pre + "attn.w_o.weight"]))
        tensors.append((f"blocks.{i}.norm2", sd[pre + "norm2.weight"]))
        tensors.append((f"blocks.{i}.w_in", sd[pre + "mlp.w_in.weight"]))
        tensors.append((f"blocks.{i}.w_out", sd[pre + "mlp.w_out.weight"]))
    tensors.append(("final_norm", sd["final_norm.weight"]))

    sae_header = None
    if sae_state is not None:
        assert sae_cfg is not None
        for name in ("w_enc", "b_enc", "w_dec", "b_dec"):
            tensors.append((f"sae.{name}", sae_state[name].detach().float().numpy()))
        sae_header = {
            "d_in": sae_cfg["d_in"],
            "expansion": sae_cfg["expansion"],
            "k": sae_cfg["k"],
            "n_features": sae_cfg["d_in"] * sae_cfg["expansion"],
            "hook_layer": sae_hook_layer,
        }

    manifest = []
    offset = 0
    blobs: list[bytes] = []
    for name, arr in tensors:
        arr = np.ascontiguousarray(arr, dtype="<f4")
        pad = (-offset) % ALIGN
        offset += pad
        blobs.append(b"\x00" * pad + arr.tobytes())
        manifest.append({
            "name": name,
            "shape": list(arr.shape),
            "offset": offset,
            "nbytes": arr.nbytes,
        })
        offset += arr.nbytes

    header = {
        "config": {
            "vocab_size": cfg.vocab_size,
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "d_mlp": cfg.d_mlp,
            "ctx_len": cfg.ctx_len,
            "rope_base": cfg.rope_base,
            "rms_eps": cfg.rms_eps,
        },
        "tokenizer": {"merges": [[a, b] for a, b in tok.merges], "vocab_size": tok.vocab_size},
        "sae": sae_header,
        "tensors": manifest,
    }
    hj = json.dumps(header).encode("utf-8")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(b"GBX1")
        f.write(np.uint32(len(hj)).tobytes())
        f.write(hj)
        base = 8 + len(hj)
        f.write(b"\x00" * ((-base) % ALIGN))
        for blob in blobs:
            f.write(blob)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, "
          f"{len(manifest)} tensors)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--sae", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ck["cfg"]))
    model.load_state_dict(ck["model"])
    model.eval()
    tok = BPETokenizer.load(args.tokenizer)

    sae_state = sae_cfg = None
    if args.sae:
        sck = torch.load(args.sae, map_location="cpu", weights_only=False)
        sae_state = sck["sae"]
        sae_cfg = sck["cfg"]

    export_gbx(model, tok, args.out, sae_state, sae_cfg)


if __name__ == "__main__":
    main()
