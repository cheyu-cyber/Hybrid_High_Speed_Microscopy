import torch

from utils.metrics import psnr, ssim_metric, summarize_reconstruction_metrics


def test_psnr_identity_is_high():
    x = torch.rand(2, 3, 16, 12)
    value = psnr(x, x)
    assert torch.isfinite(value)
    assert value.item() > 80.0


def test_ssim_metric_identity_is_near_one():
    x = torch.rand(2, 3, 16, 12)
    value = ssim_metric(x, x)
    assert torch.isfinite(value)
    assert 0.99 <= value.item() <= 1.0


def test_summarize_metrics_has_expected_keys():
    x = torch.rand(2, 3, 16, 12)
    y = torch.rand(2, 3, 16, 12)
    out = summarize_reconstruction_metrics(x, y)
    assert set(out.keys()) == {"psnr", "ssim"}
    assert isinstance(out["psnr"], float)
    assert isinstance(out["ssim"], float)
