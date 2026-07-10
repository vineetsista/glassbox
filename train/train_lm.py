"""LM training loop: AdamW, cosine schedule with warmup, grad clip, CSV+PNG
logging, checkpoint/resume. Run as a module:

    python -m train.train_lm --data data/tinystories.bin --run runs/lm_s
    python -m train.train_lm --data data/tinystories.bin --run runs/lm_s --resume

Designed to survive nohup + kill -9: every checkpoint is atomic (tmp+rename)
and carries optimizer state, RNG state, and step counter.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import TIER_S, TrainConfig
from .data import PackedDataset
from .gpt import GPT


def lr_at(step: int, tc: TrainConfig) -> float:
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    frac = (step - tc.warmup_steps) / max(1, tc.max_steps - tc.warmup_steps)
    frac = min(1.0, frac)
    cos = 0.5 * (1 + math.cos(math.pi * frac))
    return tc.lr * (tc.lr_min_frac + (1 - tc.lr_min_frac) * cos)


def save_ckpt(
    path: Path, model: GPT, opt: torch.optim.Optimizer, step: int, rng_state: object
) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "step": step,
            "rng": rng_state,
            "cfg": asdict(model.cfg),
        },
        tmp,
    )
    os.replace(tmp, path)


@torch.no_grad()
def eval_loss(model: torch.nn.Module, ds: PackedDataset, tc: TrainConfig, seed: int) -> float:
    model.eval()
    rng = np.random.default_rng(seed)  # fixed eval batches every time
    losses = []
    for _ in range(tc.eval_batches):
        x, y = ds.batch(tc.batch_size, rng, split="val")
        logits = model(x)
        losses.append(F.cross_entropy(logits.view(-1, logits.shape[-1]), y.reshape(-1)).item())
    model.train()
    return float(np.mean(losses))


def plot_curves(run_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, train_l, val_steps, val_l = [], [], [], []
    with open(run_dir / "log.csv") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            train_l.append(float(row["train_loss"]))
            if row["val_loss"]:
                val_steps.append(int(row["step"]))
                val_l.append(float(row["val_loss"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, train_l, label="train", alpha=0.7)
    if val_steps:
        ax.plot(val_steps, val_l, label="val", marker="o", ms=3)
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title(run_dir.name)
    fig.tight_layout()
    fig.savefig(run_dir / "loss.png", dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--compile", action="store_true", help="torch.compile the step fn (~1.4x on CPU)")
    ap.add_argument("--ckpt-every", type=int, default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads or max(1, (os.cpu_count() or 4) - 2))

    run_dir = Path(args.run)
    run_dir.mkdir(parents=True, exist_ok=True)
    tc = TrainConfig()
    if args.max_steps:
        tc.max_steps = args.max_steps
    if args.batch_size:
        tc.batch_size = args.batch_size
    if args.ckpt_every:
        tc.ckpt_every = args.ckpt_every
    cfg = TIER_S

    torch.manual_seed(tc.seed)
    model = GPT(cfg)
    opt = torch.optim.AdamW(
        model.parameters(), lr=tc.lr, betas=tc.betas, weight_decay=tc.weight_decay
    )
    rng = np.random.default_rng(tc.seed)
    start_step = 0

    ckpt_path = run_dir / "latest.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        rng.bit_generator.state = ck["rng"]
        print(f"resumed from step {start_step}")
    else:
        (run_dir / "config.json").write_text(
            json.dumps({"model": asdict(cfg), "train": asdict(tc)}, indent=2)
        )

    ds = PackedDataset(args.data, cfg.ctx_len)
    print(f"model params: {model.num_params():,} | tokens in dataset: {len(ds.tokens):,}")

    # forward through the compiled wrapper; checkpoints always save from `model`
    # so state_dict keys stay clean of _orig_mod prefixes
    step_model: torch.nn.Module = torch.compile(model) if args.compile else model  # type: ignore[assignment]

    log_path = run_dir / "log.csv"
    new_log = not log_path.exists()
    log_f = open(log_path, "a", newline="")  # noqa: SIM115 - lives for the whole run
    logger = csv.writer(log_f)
    if new_log:
        logger.writerow(["step", "train_loss", "val_loss", "lr", "tokens_per_sec"])

    model.train()
    t0 = time.time()
    tokens_since = 0
    running_loss = 0.0
    running_n = 0

    for step in range(start_step, tc.max_steps):
        lr = lr_at(step, tc)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        for _ in range(tc.grad_accum):
            x, y = ds.batch(tc.batch_size, rng)
            logits = step_model(x)
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.reshape(-1))
            (loss / tc.grad_accum).backward()
            tokens_since += x.numel()
            running_loss += loss.item()
            running_n += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()

        if (step + 1) % tc.log_every == 0:
            dt = time.time() - t0
            tps = tokens_since / dt if dt > 0 else 0.0
            val = ""
            if (step + 1) % tc.eval_every == 0:
                val = f"{eval_loss(step_model, ds, tc, seed=999):.4f}"
            avg = running_loss / max(1, running_n)
            logger.writerow([step + 1, f"{avg:.4f}", val, f"{lr:.2e}", f"{tps:.0f}"])
            log_f.flush()
            print(f"step {step + 1} loss {avg:.4f} val [{val}] lr {lr:.2e} tok/s {tps:.0f}")
            running_loss, running_n = 0.0, 0
            t0 = time.time()
            tokens_since = 0

        if (step + 1) % tc.ckpt_every == 0 or (step + 1) == tc.max_steps:
            save_ckpt(ckpt_path, model, opt, step + 1, rng.bit_generator.state)
            save_ckpt(run_dir / f"step{step + 1:06d}.pt", model, opt, step + 1, rng.bit_generator.state)
            try:
                plot_curves(run_dir)
            except Exception as e:  # plotting must never kill training
                print(f"plot failed: {e}")

    log_f.close()
    print("training complete")


if __name__ == "__main__":
    main()
