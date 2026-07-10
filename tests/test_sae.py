"""Unit tests for the top-k SAE."""

import torch
from sae.model import SAEConfig, TopKSAE, fvu

CFG = SAEConfig(d_in=16, expansion=4, k=3)


def make_sae(seed: int = 0) -> TopKSAE:
    torch.manual_seed(seed)
    return TopKSAE(CFG)


def test_topk_exact_sparsity() -> None:
    sae = make_sae()
    x = torch.randn(64, CFG.d_in)
    z = sae.encode(x)
    active = (z > 0).sum(dim=-1)
    assert (active <= CFG.k).all(), "more than k active features"


def test_decoder_unit_norm_after_renorm() -> None:
    sae = make_sae()
    with torch.no_grad():
        sae.w_dec.mul_(3.7)
    sae.renorm_decoder()
    norms = sae.w_dec.norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_reconstruction_beats_mean_after_few_steps() -> None:
    torch.manual_seed(1)
    sae = make_sae(1)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    # synthetic data with sparse structure the SAE can find
    basis = torch.randn(CFG.d_in, 8)
    for _ in range(300):
        coefs = torch.relu(torch.randn(256, 8)) * (torch.rand(256, 8) < 0.3)
        x = coefs @ basis.T + 0.01 * torch.randn(256, CFG.d_in)
        xhat, _ = sae(x)
        loss = fvu(x, xhat)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sae.renorm_decoder()
    assert float(loss) < 0.5, f"fvu {float(loss)} did not drop below 0.5"


def test_fvu_bounds() -> None:
    x = torch.randn(128, CFG.d_in)
    assert float(fvu(x, x)) < 1e-9
    assert abs(float(fvu(x, x.mean(dim=0, keepdim=True).expand_as(x))) - 1.0) < 1e-5


def test_encode_decode_shapes() -> None:
    sae = make_sae()
    x = torch.randn(5, CFG.d_in)
    xhat, z = sae(x)
    assert xhat.shape == x.shape
    assert z.shape == (5, CFG.n_features)


def test_state_dict_roundtrip(tmp_path) -> None:
    sae = make_sae()
    p = tmp_path / "sae.pt"
    torch.save(sae.state_dict(), p)
    sae2 = TopKSAE(CFG)
    sae2.load_state_dict(torch.load(p, weights_only=True))
    x = torch.randn(4, CFG.d_in)
    assert torch.allclose(sae(x)[0], sae2(x)[0])
