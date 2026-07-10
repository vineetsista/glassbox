"""Train the top-k SAE on cached activations.

Optimizer: Adam (no weight decay). Decoder columns renormalized to unit norm
after every step. Dead features (no fire in `dead_window` steps) are resampled
every `resample_every` steps toward high-error inputs. Metrics logged to CSV:
FVU, dead count, mean/max pre-topk activation. Checkpoint/resume like the LM.

    python -m sae.train_sae --acts data/acts_l2 --run runs/sae_l2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .model import SAEConfig, TopKSAE, fvu


@dataclass
class SAETrainConfig:
    seed: int = 4242
    batch_size: int = 4096
    lr: float = 3e-4
    max_steps: int = 20000
    dead_window: int = 500       # steps without firing -> dead
    resample_every: int = 2000
    log_every: int = 50
    ckpt_every: int = 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    tc = SAETrainConfig()
    if args.max_steps:
        tc.max_steps = args.max_steps

    acts_dir = Path(args.acts)
    meta = json.loads((acts_dir / "meta.json").read_text())
    n, d = meta["n_tokens"], meta["d_model"]
    acts = np.memmap(acts_dir / "acts.f16", dtype=np.float16, mode="r", shape=(n, d))

    run_dir = Path(args.run)
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(tc.seed)
    cfg = SAEConfig(d_in=d)
    sae = TopKSAE(cfg)
    opt = torch.optim.Adam(sae.parameters(), lr=tc.lr)
    rng = np.random.default_rng(tc.seed)
    start = 0
    fires = torch.zeros(cfg.n_features, dtype=torch.long)  # steps since last fire

    latest = run_dir / "latest.pt"
    if args.resume and latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        sae.load_state_dict(ck["sae"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        fires = ck["fires"]
        rng.bit_generator.state = ck["rng"]
        print(f"resumed at step {start}")
    else:
        (run_dir / "config.json").write_text(
            json.dumps({"sae": asdict(cfg), "train": asdict(tc), "acts_meta": meta}, indent=2)
        )

    log_path = run_dir / "log.csv"
    log_f = open(log_path, "a", newline="")  # noqa: SIM115 - lives for the whole run
    logger = csv.writer(log_f)
    if start == 0:
        logger.writerow(["step", "fvu", "dead", "mean_act", "tokens_per_sec"])

    def sample_batch() -> torch.Tensor:
        idx = rng.integers(0, n, size=tc.batch_size)
        return torch.from_numpy(acts[np.sort(idx)].astype(np.float32))

    t0 = time.time()
    seen = 0
    for step in range(start, tc.max_steps):
        x = sample_batch()
        xhat, z = sae(x)
        loss = fvu(x, xhat)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sae.renorm_decoder()

        fired = (z > 0).any(dim=0)
        fires[fired] = 0
        fires[~fired] += 1
        seen += tc.batch_size

        if (step + 1) % tc.resample_every == 0:
            dead = fires >= tc.dead_window
            n_dead = int(dead.sum())
            if n_dead:
                with torch.no_grad():
                    err = (x - xhat).pow(2).sum(dim=1)
                    worst = err.topk(min(n_dead, x.shape[0])).indices
                    for j, feat in enumerate(torch.where(dead)[0][: len(worst)]):
                        v = x[worst[j]] - sae.b_dec
                        v = v / v.norm().clamp_min(1e-8)
                        sae.w_dec[:, feat] = v
                        sae.w_enc[feat] = v * 0.2
                        sae.b_enc[feat] = 0.0
                    # reset Adam state for touched params
                    for p in (sae.w_dec, sae.w_enc, sae.b_enc):
                        st = opt.state.get(p)
                        if st:
                            st["exp_avg"].zero_()
                            st["exp_avg_sq"].zero_()
                fires[dead] = 0
                print(f"step {step + 1}: resampled {n_dead} dead features")

        if (step + 1) % tc.log_every == 0:
            dt = time.time() - t0
            dead_now = int((fires >= tc.dead_window).sum())
            row = [step + 1, f"{float(loss):.5f}", dead_now,
                   f"{float(z[z > 0].mean()):.4f}", f"{seen / dt:.0f}"]
            logger.writerow(row)
            log_f.flush()
            if (step + 1) % (tc.log_every * 10) == 0:
                print(f"step {step + 1} fvu {float(loss):.4f} dead {dead_now} "
                      f"({seen / dt:.0f} tok/s)", flush=True)
            t0 = time.time()
            seen = 0

        if (step + 1) % tc.ckpt_every == 0 or (step + 1) == tc.max_steps:
            tmp = run_dir / "latest.tmp"
            torch.save({"sae": sae.state_dict(), "opt": opt.state_dict(), "step": step + 1,
                        "fires": fires, "rng": rng.bit_generator.state,
                        "cfg": asdict(cfg)}, tmp)
            os.replace(tmp, latest)

    log_f.close()
    print("sae training complete")


if __name__ == "__main__":
    main()
