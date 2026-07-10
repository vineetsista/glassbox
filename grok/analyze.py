"""Notebook-as-script: verify the Fourier multiplication circuit on OUR
trained grok weights and emit figures + findings.json. Nothing here is
hand-waved: every claim in the README's grokking section comes from a number
this script computed.

    python -m grok.analyze --run runs/grok --out docs/figures/grok
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .fourier import (
    embedding_dft_norms,
    key_frequencies,
    logit_grid,
    neuron_freq_clustering,
    restricted_excluded_losses,
)
from .model import GrokConfig, GrokModel, train_test_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/grok")
    ap.add_argument("--out", default="docs/figures/grok")
    args = ap.parse_args()

    run_dir = Path(args.run)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cfg = json.loads((run_dir / "config.json").read_text())
    cfg = GrokConfig(**run_cfg["cfg"])
    p = cfg.p

    final = sorted((run_dir / "ckpts").glob("step*.pt"))[-1]
    ck = torch.load(final, map_location="cpu", weights_only=False)
    model = GrokModel(cfg)
    model.load_state_dict({k: v.float() for k, v in ck["model_fp16"].items()})
    model.eval()
    step = ck["step"]
    print(f"analyzing final checkpoint step {step}")

    findings: dict = {"final_step": step, "final_metrics": ck["metrics"]}

    # 1) embedding is sparse in the Fourier basis
    w_e = model.wte.weight[:p].detach()
    dft = embedding_dft_norms(w_e, p)
    kf = key_frequencies(w_e, p)
    findings["key_freqs"] = kf
    total_power = float(dft[1:].pow(2).sum())
    key_power = float(dft[kf].pow(2).sum())
    findings["emb_power_frac_in_key_freqs"] = key_power / total_power

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(dft)), dft.numpy())
    for k in kf:
        ax.axvline(k, color="crimson", alpha=0.3)
    ax.set_xlabel("frequency k")
    ax.set_ylabel("|DFT| norm across d_model")
    ax.set_title(f"Embedding DFT (step {step}); key freqs {kf} hold "
                 f"{100 * findings['emb_power_frac_in_key_freqs']:.1f}% of non-DC power")
    fig.tight_layout()
    fig.savefig(out_dir / "emb_dft.png", dpi=120)
    plt.close(fig)

    # 2) MLP neurons cluster on single frequencies
    clus = neuron_freq_clustering(model)
    frac = clus["freq_power_frac"]
    dom = clus["dominant_freq"]
    findings["neurons_frac_gt_085"] = float((frac > 0.85).float().mean())
    findings["neuron_dominant_freq_counts"] = {
        str(k): int((dom == k).sum()) for k in sorted(set(dom.tolist()))
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(frac.numpy(), bins=40)
    axes[0].set_xlabel("fraction of non-DC power in dominant freq")
    axes[0].set_ylabel("neurons")
    axes[0].set_title(f"{100 * findings['neurons_frac_gt_085']:.0f}% of neurons >0.85")
    axes[1].hist(dom.numpy(), bins=range(0, p // 2 + 2))
    axes[1].set_xlabel("dominant frequency")
    axes[1].set_ylabel("neurons")
    axes[1].set_title("dominant frequency per neuron")
    fig.tight_layout()
    fig.savefig(out_dir / "neuron_freqs.png", dpi=120)
    plt.close(fig)

    # 3) example neuron activation heatmaps (top-power neurons)
    from .model import all_pairs_dataset

    tokens, _ = all_pairs_dataset(p)
    acts: list[torch.Tensor] = []

    def grab(x: torch.Tensor, hp: object) -> None:
        acts.append(x[:, -1, :].detach())

    with torch.no_grad(), model.hooks([("hook_mlp_act", grab)]):  # type: ignore[list-item]
        for i in range(0, tokens.shape[0], 12769):
            model(tokens[i : i + 12769])
    grid = torch.cat(acts).view(p, p, -1)
    top_neurons = frac.argsort(descending=True)[:4].tolist()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, ni in zip(axes, top_neurons, strict=False):
        ax.imshow(grid[:, :, ni].numpy(), cmap="RdBu_r")
        ax.set_title(f"neuron {ni} (freq {int(dom[ni])}, frac {float(frac[ni]):.2f})")
        ax.set_xlabel("b")
        ax.set_ylabel("a")
    fig.suptitle("MLP activations over (a, b): periodic plaid = waves in Z_p")
    fig.tight_layout()
    fig.savefig(out_dir / "neuron_heatmaps.png", dpi=120)
    plt.close(fig)

    # 4) restricted / excluded losses on the final model
    train_mask, test_mask = train_test_split(p, run_cfg["train_frac"], run_cfg["seed"])
    with torch.no_grad():
        logits = logit_grid(model)
    rex = restricted_excluded_losses(logits, kf, train_mask, test_mask)
    findings["final_restricted_excluded"] = rex

    # 5) training curves + progress measures over the whole run
    prog = run_dir / "progress.csv"
    if prog.exists():
        rows = list(csv.DictReader(open(prog)))
        steps = [int(r["step"]) for r in rows]
        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        axes[0].plot(steps, [float(r["train_acc"]) for r in rows], label="train acc")
        axes[0].plot(steps, [float(r["test_acc"]) for r in rows], label="test acc")
        axes[0].set_ylabel("accuracy")
        axes[0].legend()
        axes[0].set_title("grokking: memorization then generalization")
        axes[1].plot(steps, [float(r["full_test"]) for r in rows], label="test loss")
        axes[1].plot(steps, [float(r["restricted_test"]) for r in rows],
                     label="restricted test loss (key freqs only)")
        axes[1].plot(steps, [float(r["excluded_train"]) for r in rows],
                     label="excluded train loss (key freqs removed)")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("cross-entropy")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(out_dir / "grokking_curves.png", dpi=120)
        plt.close(fig)
    else:
        print("progress.csv not found - run export_web first for curve figures")

    (out_dir / "findings.json").write_text(json.dumps(findings, indent=2))
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
