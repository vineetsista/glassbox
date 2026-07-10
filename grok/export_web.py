"""Pack every grok checkpoint into one binary for the web scrubber (.grokpack)
plus a per-checkpoint progress CSV for analysis figures.

Format (little-endian):
    magic   "GRPK"
    u32     version = 1
    u32     header_len
    bytes   header JSON (utf-8)
    then n_ckpts fixed-size records as described by header.record_fields.

Per record:
    u32     step
    f32 x 8 train_loss, train_acc, test_loss, test_acc,
            restricted_test, excluded_train, full_test, wnorm
    f32 x (p//2+1)  embedding DFT norms
    f16 x (p+1)*d   token embeddings (numbers + '=')
    f16 x n_probe*heads*3  attention row of the '=' position on probe pairs

    python -m grok.export_web --run runs/grok --out web/public/assets/grok.grokpack
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path

import numpy as np
import torch
from train.gpt import HookPoint

from .fourier import (
    embedding_dft_norms,
    key_frequencies,
    logit_grid,
    restricted_excluded_losses,
)
from .model import GrokConfig, GrokModel, train_test_split

N_PROBE = 16


def probe_pairs(p: int, n: int = N_PROBE) -> torch.Tensor:
    g = torch.Generator().manual_seed(123)
    a = torch.randint(0, p, (n,), generator=g)
    b = torch.randint(0, p, (n,), generator=g)
    eq = torch.full((n,), p)
    return torch.stack([a, b, eq], dim=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/grok")
    ap.add_argument("--out", default="web/public/assets/grok.grokpack")
    args = ap.parse_args()

    run_dir = Path(args.run)
    run_cfg = json.loads((run_dir / "config.json").read_text())
    cfg = GrokConfig(**run_cfg["cfg"])
    p = cfg.p

    ckpt_files = sorted((run_dir / "ckpts").glob("step*.pt"))
    assert ckpt_files, "no checkpoints found"
    print(f"{len(ckpt_files)} checkpoints")

    model = GrokModel(cfg)
    model.eval()

    def load_into(model: GrokModel, f: Path) -> tuple[int, dict[str, float]]:
        ck = torch.load(f, map_location="cpu", weights_only=False)
        model.load_state_dict({k: v.float() for k, v in ck["model_fp16"].items()})
        return ck["step"], ck["metrics"]

    # key frequencies come from the FINAL model and are held fixed across the scrub
    final_step, _ = load_into(model, ckpt_files[-1])
    kf = key_frequencies(model.wte.weight[:p].detach(), p)
    print(f"key frequencies (from final ckpt step {final_step}): {kf}")

    train_mask, test_mask = train_test_split(p, run_cfg["train_frac"], run_cfg["seed"])
    probes = probe_pairs(p)

    pat: list[torch.Tensor] = []

    def grab_pattern(x: torch.Tensor, hp: HookPoint) -> None:
        pat.append(x[:, :, -1, :].detach())  # '=' row: [B, H, 3]

    header = {
        "format": "grokpack",
        "p": p,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "n_ckpts": len(ckpt_files),
        "key_freqs": kf,
        "n_freq": p // 2 + 1,
        "n_probe": N_PROBE,
        "probe_pairs": probes[:, :2].tolist(),
        "train_frac": run_cfg["train_frac"],
        "record_fields": [
            {"name": "step", "dtype": "u32", "count": 1},
            {"name": "metrics", "dtype": "f32", "count": 8,
             "labels": ["train_loss", "train_acc", "test_loss", "test_acc",
                        "restricted_test", "excluded_train", "full_test", "wnorm"]},
            {"name": "emb_dft", "dtype": "f32", "count": p // 2 + 1},
            {"name": "emb", "dtype": "f16", "count": (p + 1) * cfg.d_model},
            {"name": "attn_probe", "dtype": "f16", "count": N_PROBE * cfg.n_heads * 3},
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "progress.csv"

    with open(out_path, "wb") as f, open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["step", "train_loss", "train_acc", "test_loss", "test_acc",
                         "restricted_test", "excluded_train", "full_test", "wnorm"])
        hj = json.dumps(header).encode("utf-8")
        f.write(b"GRPK" + struct.pack("<II", 1, len(hj)) + hj)

        for i, cf_path in enumerate(ckpt_files):
            step, metrics = load_into(model, cf_path)
            with torch.no_grad():
                logits = logit_grid(model)
                rex = restricted_excluded_losses(logits, kf, train_mask, test_mask)
                dft = embedding_dft_norms(model.wte.weight[:p].detach(), p)
                wnorm = float(
                    torch.stack([w.pow(2).sum() for w in model.parameters()]).sum().sqrt()
                )
                pat.clear()
                with model.hooks([("hook_pattern", grab_pattern)]):
                    model(probes)

            row = [metrics["train_loss"], metrics["train_acc"], metrics["test_loss"],
                   metrics["test_acc"], rex["restricted_test"], rex["excluded_train"],
                   rex["full_test"], wnorm]
            writer.writerow([step, *[f"{v:.6f}" for v in row]])

            f.write(struct.pack("<I", step))
            f.write(np.array(row, dtype="<f4").tobytes())
            f.write(dft.numpy().astype("<f4").tobytes())
            f.write(model.wte.weight.detach().numpy().astype("<f2").tobytes())
            f.write(pat[0].numpy().astype("<f2").tobytes())
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(ckpt_files)}")

    size = out_path.stat().st_size
    print(f"wrote {out_path} ({size / 1e6:.1f} MB) and {csv_path}")


if __name__ == "__main__":
    main()
