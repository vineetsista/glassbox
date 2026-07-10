"""Bake static feature dashboards from a trained SAE + cached activations.

Per feature JSON (web/public/assets/features/f{ID}.json):
    - density: fraction of sampled tokens where the feature is active post-topk
    - hist: log-binned activation histogram (positive activations)
    - logit_lens: top promoted / suppressed tokens via W_dec[:, f] @ W_U
      (direct-path approximation: ignores blocks after the hook layer and the
      final RMSNorm's per-example scale; documented in METHODS.md)
    - examples: top max-activating windows with token strings and per-token
      feature activations for highlighting (deduped by window)

Plus index.json: per-feature summary (density, max act, top tokens preview).

    python -m sae.dashboards --sae runs/sae_l2/latest.pt --acts data/acts_l2 \
        --lm runs/lm_s/latest.pt --tokenizer data/tokenizer.json \
        --out web/public/assets/features
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from train.bpe import BPETokenizer
from train.config import GPTConfig
from train.gpt import GPT

from .model import SAEConfig, TopKSAE

N_EXAMPLES = 12
CTX_BEFORE = 24
CTX_AFTER = 8
HIST_BINS = 32
HIST_LO, HIST_HI = 1e-2, 1e2  # log-spaced absolute bins; activations outside clip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--lm", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scan-tokens", type=int, default=4_000_000)
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    acts_dir = Path(args.acts)
    meta = json.loads((acts_dir / "meta.json").read_text())
    n_tok, d = meta["n_tokens"], meta["d_model"]
    ctx = meta["ctx_len"]
    acts = np.memmap(acts_dir / "acts.f16", dtype=np.float16, mode="r", shape=(n_tok, d))
    starts = np.fromfile(acts_dir / "windows.u64", dtype=np.uint64)
    tokens = np.memmap(meta["data"], dtype=np.uint16, mode="r")

    ck = torch.load(args.sae, map_location="cpu", weights_only=False)
    cfg = SAEConfig(**ck["cfg"])
    sae = TopKSAE(cfg)
    sae.load_state_dict(ck["sae"])
    sae.eval()
    n_feat = cfg.n_features

    lm_ck = torch.load(args.lm, map_location="cpu", weights_only=False)
    gpt = GPT(GPTConfig(**lm_ck["cfg"]))
    gpt.load_state_dict(lm_ck["model"])
    tok = BPETokenizer.load(args.tokenizer)

    n_windows = min(meta["n_windows"], args.scan_tokens // ctx)
    print(f"scanning {n_windows} windows ({n_windows * ctx / 1e6:.1f}M tokens)")

    # pass 1: per-window max activation and position for every feature,
    # plus histograms / densities from the same scan
    win_max = np.zeros((n_windows, n_feat), dtype=np.float16)
    win_pos = np.zeros((n_windows, n_feat), dtype=np.uint16)
    hist = np.zeros((n_feat, HIST_BINS), dtype=np.int64)
    fire_count = np.zeros(n_feat, dtype=np.int64)
    bin_edges = np.logspace(np.log10(HIST_LO), np.log10(HIST_HI), HIST_BINS + 1)

    bs = 16  # windows per batch
    with torch.no_grad():
        for i in range(0, n_windows, bs):
            nb = min(bs, n_windows - i)
            x = torch.from_numpy(
                acts[i * ctx : (i + nb) * ctx].astype(np.float32)
            )
            z = sae.encode(x).view(nb, ctx, n_feat)
            m, p = z.max(dim=1)
            win_max[i : i + nb] = m.numpy().astype(np.float16)
            win_pos[i : i + nb] = p.numpy().astype(np.uint16)
            zpos = z[z > 0]
            fire_count += (z > 0).sum(dim=(0, 1)).numpy()
            if zpos.numel():
                # histogram per feature: bucket indices from log bins
                idx = np.clip(
                    np.searchsorted(bin_edges, z.numpy().reshape(-1, n_feat), side="right") - 1,
                    0, HIST_BINS - 1,
                )
                mask = z.numpy().reshape(-1, n_feat) > 0
                for b in range(HIST_BINS):
                    hist[:, b] += ((idx == b) & mask).sum(axis=0)
            if (i // bs) % 100 == 0 and i:
                print(f"  scan {i}/{n_windows}", flush=True)

    total_scanned = n_windows * ctx
    density = fire_count / total_scanned

    # logit lens: decoder direction -> unembed (direct path)
    with torch.no_grad():
        w_u = gpt.wte.weight  # [vocab, d]
        lens = (sae.w_dec.T @ w_u.T).float()  # [n_feat, vocab]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    index = []
    with torch.no_grad():
        for f in range(n_feat):
            top_w = np.argsort(win_max[:, f].astype(np.float32))[::-1][:N_EXAMPLES]
            examples = []
            for w in top_w:
                if win_max[w, f] <= 0:
                    break
                pos = int(win_pos[w, f])
                lo = max(0, pos - CTX_BEFORE)
                hi = min(ctx, pos + CTX_AFTER + 1)
                row = torch.from_numpy(
                    acts[int(w) * ctx + lo : int(w) * ctx + hi].astype(np.float32)
                )
                z_slice = sae.encode(row)[:, f]
                s = int(starts[w])
                ids = tokens[s + lo : s + hi].tolist()
                examples.append({
                    "tokens": [tok.token_str(t) for t in ids],
                    "acts": [round(float(a), 4) for a in z_slice],
                    "max_pos": pos - lo,
                    "max_act": round(float(win_max[w, f]), 4),
                })
            pos_lens = lens[f].topk(10)
            neg_lens = (-lens[f]).topk(10)
            promoted = [[tok.token_str(int(i)), round(float(v), 3)]
                        for v, i in zip(pos_lens.values, pos_lens.indices, strict=True)]
            feature = {
                "id": f,
                "density": float(density[f]),
                "max_act": float(win_max[:, f].astype(np.float32).max()),
                "hist": {"edges": [round(float(e), 4) for e in bin_edges],
                         "counts": hist[f].tolist()},
                "logit_lens": {
                    "promoted": promoted,
                    "suppressed": [[tok.token_str(int(i)), round(float(v), 3)]
                                   for v, i in zip(neg_lens.values, neg_lens.indices, strict=True)],
                },
                "examples": examples,
            }
            (out / f"f{f}.json").write_text(json.dumps(feature))
            index.append({
                "id": f,
                "density": round(float(density[f]), 6),
                "max_act": round(float(win_max[:, f].astype(np.float32).max()), 3),
                "top_tokens": [t for t, _ in promoted[:4]],
                "n_examples": len(examples),
            })
            if (f + 1) % 200 == 0:
                print(f"  baked {f + 1}/{n_feat}", flush=True)

    (out / "index.json").write_text(json.dumps({
        "n_features": n_feat,
        "k": cfg.k,
        "hook": meta["hook"],
        "scanned_tokens": total_scanned,
        "features": index,
    }))
    print(f"dashboards -> {out}")


if __name__ == "__main__":
    main()
