from __future__ import annotations

import math
from typing import Dict

import torch

from models.loss import ssim_loss


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Compute PSNR in dB for batched tensors in [0, data_range]."""
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")

    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    mse = torch.clamp(mse, min=1e-12)
    psnr_db = 10.0 * torch.log10((data_range ** 2) / mse)
    return psnr_db.mean()


def ssim_metric(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Compute SSIM score from ssim_loss implementation."""
    return 1.0 - ssim_loss(pred, target, data_range=data_range)


def summarize_reconstruction_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Return scalar reconstruction metrics for logging."""
    with torch.no_grad():
        psnr_value = float(psnr(pred, target).item())
        ssim_value = float(ssim_metric(pred, target).item())
    return {
        "psnr": psnr_value,
        "ssim": ssim_value,
    }
