"""Train the grokking specimen: full-batch AdamW, weight decay 1.0, mod-113
addition, 30% train fraction. Dense checkpointing every 100 steps for the web
scrubber and post-hoc analysis.

    python -m grok.train_grok --run runs/grok --steps 30000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import GrokConfig, GrokModel, all_pairs_dataset, train_test_split

SEED = 999
TRAIN_FRAC = 0.3
LR = 1e-3
WD = 1.0
BETAS = (0.9, 0.98)
CKPT_EVERY = 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/grok")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads or max(1, (os.cpu_count() or 4) - 2))
    torch.manual_seed(SEED)

    run_dir = Path(args.run)
    (run_dir / "ckpts").mkdir(parents=True, exist_ok=True)

    cfg = GrokConfig()
    model = GrokModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=BETAS)

    tokens, labels = all_pairs_dataset(cfg.p)
    train_mask, test_mask = train_test_split(cfg.p, TRAIN_FRAC, SEED)
    x_tr, y_tr = tokens[train_mask], labels[train_mask]
    x_te, y_te = tokens[test_mask], labels[test_mask]

    start = 0
    latest = run_dir / "latest.pt"
    if args.resume and latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        print(f"resumed at step {start}")
    else:
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "cfg": asdict(cfg),
                    "seed": SEED,
                    "train_frac": TRAIN_FRAC,
                    "lr": LR,
                    "wd": WD,
                    "betas": BETAS,
                    "ckpt_every": CKPT_EVERY,
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                },
                indent=2,
            )
        )

    log_path = run_dir / "log.csv"
    log_f = open(log_path, "a", newline="")  # noqa: SIM115 - lives for the whole run
    logger = csv.writer(log_f)
    if start == 0:
        logger.writerow(["step", "train_loss", "train_acc", "test_loss", "test_acc"])

    t0 = time.time()
    for step in range(start, args.steps):
        model.train()
        logits = model(x_tr)
        loss = F.cross_entropy(logits, y_tr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step + 1) % CKPT_EVERY == 0 or step == 0:
            model.eval()
            with torch.no_grad():
                tr_logits = model(x_tr)
                te_logits = model(x_te)
                tr_loss = F.cross_entropy(tr_logits, y_tr).item()
                te_loss = F.cross_entropy(te_logits, y_te).item()
                tr_acc = (tr_logits.argmax(-1) == y_tr).float().mean().item()
                te_acc = (te_logits.argmax(-1) == y_te).float().mean().item()
            logger.writerow(
                [step + 1, f"{tr_loss:.6f}", f"{tr_acc:.4f}", f"{te_loss:.6f}", f"{te_acc:.4f}"]
            )
            log_f.flush()
            sd_fp16 = {k: v.half() for k, v in model.state_dict().items()}
            torch.save(
                {"model_fp16": sd_fp16, "step": step + 1,
                 "metrics": {"train_loss": tr_loss, "train_acc": tr_acc,
                             "test_loss": te_loss, "test_acc": te_acc}},
                run_dir / "ckpts" / f"step{step + 1:06d}.pt",
            )
            tmp = run_dir / "latest.tmp"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1}, tmp)
            os.replace(tmp, latest)
            if (step + 1) % 1000 == 0 or step == 0:
                rate = (step + 1 - start) / (time.time() - t0)
                print(
                    f"step {step + 1} train {tr_loss:.4f}/{tr_acc:.3f} "
                    f"test {te_loss:.4f}/{te_acc:.3f} ({rate:.1f} it/s)"
                )

    log_f.close()
    print("grok training complete")


if __name__ == "__main__":
    main()
