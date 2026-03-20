import pytest
import torch

from models.model import EventVFIConfig, EventVFIModel


def _assert_finite_tensor(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        raise AssertionError(f"Non-finite values detected in {name}")


@pytest.mark.smoke
@pytest.mark.cpu
def test_event_vfi_one_batch_smoke() -> None:
    torch.manual_seed(7)
    dev = torch.device("cpu")

    # Small setup for quick smoke testing.
    b, c, h, w = 2, 3, 32, 32
    num_bins = 5

    cfg = EventVFIConfig(
        num_event_bins=num_bins,
        rgb_in_ch=c,
        rgb_base_ch=8,
        event_base_ch=8,
        encoder_res_blocks=1,
        fusion_use_tau=False,
        fusion_res_blocks=1,
        motion_res_blocks_per_level=1,
        decoder_hidden_ch=24,
        decoder_res_blocks=2,
        use_warped_rgb_features=True,
        clamp_output=True,
    )

    model = EventVFIModel(cfg).to(dev)
    model.train()

    # One fixed synthetic batch.
    I0 = torch.rand(b, c, h, w, device=dev)
    I1 = torch.rand(b, c, h, w, device=dev)
    I2 = torch.rand(b, c, h, w, device=dev)

    E01 = torch.randn(b, num_bins, h, w, device=dev) * 0.05
    E12 = torch.randn(b, num_bins, h, w, device=dev) * 0.05
    E0515 = torch.randn(b, num_bins, h, w, device=dev) * 0.05

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # First forward: shape/finite/warping checks.
    out = model(I0, I1, I2, E01, E12, E0515, return_debug=True)

    assert out["pred_05"].shape == (b, c, h, w)
    assert out["pred_15"].shape == (b, c, h, w)
    assert out["pred_1_cyc"].shape == (b, c, h, w)

    _assert_finite_tensor("pred_05", out["pred_05"])
    _assert_finite_tensor("pred_15", out["pred_15"])
    _assert_finite_tensor("pred_1_cyc", out["pred_1_cyc"])

    warped0 = out["stage_01"]["decoder"]["warped0"]
    warped1 = out["stage_01"]["decoder"]["warped1"]
    assert warped0.shape == (b, c, h, w)
    assert warped1.shape == (b, c, h, w)
    _assert_finite_tensor("stage_01.warped0", warped0)
    _assert_finite_tensor("stage_01.warped1", warped1)

    # Repeated optimization on same batch: loss should trend down.
    losses = []
    for _ in range(6):
        optim.zero_grad(set_to_none=True)
        out = model(I0, I1, I2, E01, E12, E0515, return_debug=False)
        loss_dict = model.compute_cycle_loss(out, I0=I0, I1=I1, I2=I2)
        loss = loss_dict["loss_total"]

        _assert_finite_tensor("loss_total", loss.unsqueeze(0))
        loss.backward()

        # Backward pass finite check.
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                raise AssertionError(f"Non-finite gradient in {name}")

        optim.step()
        losses.append(float(loss.detach().cpu().item()))

    first = losses[0]
    best = min(losses)
    assert best < first, "Loss did not decrease across optimization steps"

