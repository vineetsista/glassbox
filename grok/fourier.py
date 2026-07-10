"""Fourier-space analysis helpers for the grokking specimen.

The modular-addition circuit story (Nanda et al. 2023): the network learns
embeddings that are sparse in the Fourier basis over Z_p, attention/MLP
compute products of waves, and the unembedding reads off cos(w(a+b-c)) terms.
These helpers measure that on OUR weights.

Progress measures: our `restricted_loss` keeps only the model's key 2D Fourier
components of the logit grid (plus DC); `excluded_loss` removes them and keeps
everything else. These are inspired by, but not identical to, the definitions
in Nanda et al. — see docs/METHODS.md for the exact functional forms and
caveats. All losses are computed over the full p*p grid or a mask of it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import GrokModel, all_pairs_dataset


@torch.no_grad()
def logit_grid(model: GrokModel, batch_size: int = 12769) -> torch.Tensor:
    """Full logits over every (a, b): returns [p, p, p] (last dim = numeric classes)."""
    p = model.cfg.p
    tokens, _ = all_pairs_dataset(p)
    outs = []
    for i in range(0, tokens.shape[0], batch_size):
        outs.append(model(tokens[i : i + batch_size])[:, :p])
    return torch.cat(outs).view(p, p, p)


def embedding_dft_norms(w_e: torch.Tensor, p: int) -> torch.Tensor:
    """1D DFT of number-token embeddings over the token axis.

    w_e: [p, d] (numeric tokens only). Returns [p//2 + 1] norms: for each
    frequency k, the L2 norm across (cos, sin) components and d_model.
    """
    fourier = torch.fft.rfft(w_e.float(), dim=0)  # [p//2+1, d] complex
    return fourier.abs().pow(2).sum(dim=1).sqrt()


def key_frequencies(w_e: torch.Tensor, p: int, top_k: int = 5) -> list[int]:
    """Frequencies (1..p//2) with the largest embedding DFT norm."""
    norms = embedding_dft_norms(w_e, p)
    norms = norms.clone()
    norms[0] = 0.0  # DC is not a wave
    return sorted(int(i) for i in norms.topk(top_k).indices)


def _freq_mask(p: int, key_freqs: list[int], device: torch.device) -> torch.Tensor:
    """Boolean mask [p, p] over the 2D DFT grid (fftfreq layout) that selects
    components whose |row freq| and |col freq| both lie in key_freqs u {0},
    excluding the pure-DC component (kept separately)."""
    idx = torch.arange(p, device=device)
    folded = torch.minimum(idx, p - idx)  # |frequency| for fft layout
    keep = torch.zeros(p, dtype=torch.bool, device=device)
    keep[0] = True
    for k in key_freqs:
        keep[k] = True
    row_ok = keep[folded]
    mask = row_ok[:, None] & row_ok[None, :]
    mask[0, 0] = False  # DC handled by the caller
    return mask


@torch.no_grad()
def restricted_excluded_losses(
    logits: torch.Tensor,
    key_freqs: list[int],
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> dict[str, float]:
    """Reconstruct the [p, p, p] logit grid keeping (restricted) or removing
    (excluded) the key-frequency 2D Fourier components, then measure CE loss.

    Returns restricted/excluded loss on the test/train split respectively:
    restricted on test (does the periodic structure alone generalize?) and
    excluded on train (does anything beyond the periodic structure still fit
    the training set, i.e. residual memorization?).
    """
    p = logits.shape[0]
    device = logits.device
    labels = ((torch.arange(p)[:, None] + torch.arange(p)[None, :]) % p).to(device)

    spec = torch.fft.fft2(logits.permute(2, 0, 1).float(), dim=(-2, -1))  # [p, p, p]
    mask = _freq_mask(p, key_freqs, device)
    dc = torch.zeros_like(mask)
    dc[0, 0] = True

    spec_restricted = spec * (mask | dc)
    spec_excluded = spec * (~mask)
    restricted = torch.fft.ifft2(spec_restricted, dim=(-2, -1)).real.permute(1, 2, 0)
    excluded = torch.fft.ifft2(spec_excluded, dim=(-2, -1)).real.permute(1, 2, 0)

    def ce(lg: torch.Tensor, m: torch.Tensor) -> float:
        return float(F.cross_entropy(lg.reshape(-1, p)[m.reshape(-1)], labels.reshape(-1)[m.reshape(-1)]))

    return {
        "restricted_test": ce(restricted, test_mask.view(p, p)),
        "excluded_train": ce(excluded, train_mask.view(p, p)),
        "full_test": ce(logits, test_mask.view(p, p)),
        "full_train": ce(logits, train_mask.view(p, p)),
    }


@torch.no_grad()
def neuron_freq_clustering(model: GrokModel) -> dict[str, torch.Tensor]:
    """For each MLP neuron: activation over the (a, b) grid at the final
    position -> 2D DFT -> fraction of (non-DC) power in its dominant frequency.

    Returns {"dominant_freq": [d_mlp] long, "freq_power_frac": [d_mlp] float}.
    """
    p = model.cfg.p
    tokens, _ = all_pairs_dataset(p)
    acts: list[torch.Tensor] = []

    def grab(x: torch.Tensor, hp: object) -> None:
        acts.append(x[:, -1, :].detach())

    with model.hooks([("hook_mlp_act", grab)]):  # type: ignore[list-item]
        for i in range(0, tokens.shape[0], 12769):
            model(tokens[i : i + 12769])
    grid = torch.cat(acts).view(p, p, -1).permute(2, 0, 1).float()  # [d_mlp, p, p]

    spec = torch.fft.fft2(grid, dim=(-2, -1)).abs().pow(2)
    spec[:, 0, 0] = 0.0  # ignore DC
    idx = torch.arange(p)
    folded = torch.minimum(idx, p - idx)
    n_freq = p // 2 + 1
    power = torch.zeros(grid.shape[0], n_freq)
    # attribute each 2D component to max(|fa|, |fb|) — a neuron tuned to
    # frequency k has energy at combinations of {0, k} in both axes
    comb = torch.maximum(folded[:, None].expand(p, p), folded[None, :].expand(p, p))
    for k in range(1, n_freq):
        power[:, k] = spec[:, comb == k].sum(dim=1)
    total = power.sum(dim=1, keepdim=True).clamp_min(1e-9)
    frac, dom = (power / total).max(dim=1)
    return {"dominant_freq": dom, "freq_power_frac": frac}
