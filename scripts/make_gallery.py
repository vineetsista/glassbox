"""Generate the honest sample gallery: fixed prompts, fixed seeds, no
cherry-picking. Output is a markdown file with every sample exactly as
generated (plus the checkpoint step and val loss for context).

    python scripts/make_gallery.py --ckpt runs/lm_s/latest.pt \
        --tokenizer data/tokenizer.json --out docs/SAMPLES.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.bpe import BPETokenizer
from train.config import GPTConfig
from train.gpt import GPT
from train.sample import EVAL_PROMPTS, generate

SEEDS = [0, 1, 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", default="docs/SAMPLES.md")
    ap.add_argument("--max-new", type=int, default=140)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    torch.set_num_threads(8)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ck["cfg"]))
    model.load_state_dict(ck["model"])
    model.eval()
    tok = BPETokenizer.load(args.tokenizer)

    lines = [
        "# Sample gallery (unedited)",
        "",
        f"Checkpoint: `{args.ckpt}` at step {ck['step']}. "
        f"Temperature {args.temperature}, top-k 40, fixed prompts and seeds "
        f"(train/sample.py). Every sample below is exactly what the model wrote; "
        "nothing was regenerated or selected. This model has ~2.6M parameters and "
        "was trained on a laptop CPU - judge it accordingly.",
        "",
    ]
    for prompt in EVAL_PROMPTS:
        lines.append(f"## `{prompt}`")
        lines.append("")
        for seed in SEEDS:
            text = generate(
                model, tok, prompt,
                max_new_tokens=args.max_new,
                temperature=args.temperature,
                seed=seed,
            )
            lines.append(f"**seed {seed}:** {text}")
            lines.append("")
        print(f"done: {prompt!r}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"gallery -> {args.out}")


if __name__ == "__main__":
    main()
