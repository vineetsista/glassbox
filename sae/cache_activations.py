"""Stream the packed corpus through the trained LM and memmap one layer's
resid_post activations to disk in fp16, along with the window index so
dashboards can recover token context.

Layout on disk (out dir):
    acts.f16      [n_tokens_cached, d_model] fp16, row-major
    windows.u64   [n_windows] start offsets into the token pack
    meta.json     hook name, ckpt, counts, provenance

    python -m sae.cache_activations --ckpt runs/lm_s/latest.pt \
        --data data/tinystories.bin --out data/acts_l2 --n-tokens 20000000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from train.config import GPTConfig
from train.gpt import GPT

HOOK = "blocks.2.hook_resid_post"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-tokens", type=int, default=20_000_000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--hook", default=HOOK)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    tokens = np.memmap(args.data, dtype=np.uint16, mode="r")
    ctx = cfg.ctx_len
    n_windows = args.n_tokens // ctx
    # deterministic disjoint windows from the START of the pack (training
    # sampled uniform random windows over the first 99%; overlap is expected
    # and harmless - the SAE learns the distribution, not held-out data)
    starts = np.arange(n_windows, dtype=np.uint64) * ctx
    assert int(starts[-1]) + ctx + 1 < len(tokens), "pack too small for requested tokens"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    acts = np.memmap(out / "acts.f16", dtype=np.float16, mode="w+",
                     shape=(n_windows * ctx, cfg.d_model))
    starts.tofile(out / "windows.u64")

    grabbed: list[torch.Tensor] = []

    def grab(x: torch.Tensor, hp: object) -> None:
        grabbed.append(x.detach())

    bs = args.batch_size
    t0 = time.time()
    with torch.no_grad(), model.hooks([(args.hook, grab)]):
        for i in range(0, n_windows, bs):
            batch_starts = starts[i : i + bs]
            rows = np.stack([tokens[s : s + ctx] for s in batch_starts]).astype(np.int64)
            grabbed.clear()
            model(torch.from_numpy(rows))
            a = grabbed[0].reshape(-1, cfg.d_model).to(torch.float16).numpy()
            acts[i * ctx : i * ctx + a.shape[0]] = a
            if (i // bs) % 50 == 0 and i > 0:
                done = i * ctx
                rate = done / (time.time() - t0)
                eta = (n_windows * ctx - done) / rate / 60
                print(f"{done / 1e6:.1f}M/{n_windows * ctx / 1e6:.0f}M tokens "
                      f"({rate:.0f} tok/s, eta {eta:.0f} min)", flush=True)

    acts.flush()
    meta = {
        "hook": args.hook,
        "ckpt": args.ckpt,
        "ckpt_step": ck["step"],
        "data": args.data,
        "n_tokens": n_windows * ctx,
        "n_windows": n_windows,
        "ctx_len": ctx,
        "d_model": cfg.d_model,
        "dtype": "float16",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"cached {n_windows * ctx / 1e6:.1f}M activations to {out}")


if __name__ == "__main__":
    main()
