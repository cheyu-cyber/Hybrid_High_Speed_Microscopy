import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# =========================================================
# Basic loss pieces
# =========================================================

def charbonnier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-3,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Robust L1-style loss.

    pred, target: (B, C, H, W)
    mask: optional (B, 1, H, W) or (B, C, H, W)
    """
    diff = torch.sqrt((pred - target) ** 2 + eps ** 2)
    if mask is not None:
        diff = diff * mask
        denom = mask.sum() * pred.shape[1] + 1e-6 if mask.shape[1] == 1 else mask.sum() + 1e-6
        return diff.sum() / denom
    return diff.mean()


def _gaussian_kernel(window_size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel_2d = kernel_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)
    return kernel_2d


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """
    Returns 1 - SSIM averaged over batch and channels.
    Assumes images are in [0, 1].
    """
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")

    b, c, _, _ = pred.shape
    kernel = _gaussian_kernel(window_size, sigma, c, pred.device, pred.dtype)

    mu_x = F.conv2d(pred, kernel, padding=window_size // 2, groups=c)
    mu_y = F.conv2d(target, kernel, padding=window_size // 2, groups=c)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, kernel, padding=window_size // 2, groups=c) - mu_x2
    sigma_y2 = F.conv2d(target * target, kernel, padding=window_size // 2, groups=c) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel, padding=window_size // 2, groups=c) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-8
    )

    return (1.0 - ssim_map).mean()


def image_gradients(x: torch.Tensor):
    """
    Simple forward differences.
    """
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy


def edge_aware_flow_smoothness(
    flow: torch.Tensor,
    image: torch.Tensor,
    alpha: float = 10.0,
) -> torch.Tensor:
    """
    Edge-aware flow smoothness:
    smoother flow in flat image regions,
    less penalty near image edges.
    """
    flow_dx, flow_dy = image_gradients(flow)
    img_dx, img_dy = image_gradients(image)

    # average over channels for RGB
    img_dx_mag = img_dx.abs().mean(dim=1, keepdim=True)
    img_dy_mag = img_dy.abs().mean(dim=1, keepdim=True)

    weight_x = torch.exp(-alpha * img_dx_mag)
    weight_y = torch.exp(-alpha * img_dy_mag)

    loss_x = (flow_dx.abs() * weight_x).mean()
    loss_y = (flow_dy.abs() * weight_y).mean()
    return loss_x + loss_y


# =========================================================
# Optional perceptual loss
# =========================================================

class VGGPerceptualLoss(nn.Module):
    """
    Optional perceptual loss using torchvision VGG16 features.

    This block is optional because some training environments
    prefer not to download torchvision weights automatically.
    """
    def __init__(self, resize_to_224: bool = False) -> None:
        super().__init__()
        try:
            from torchvision import models
            weights = models.VGG16_Weights.IMAGENET1K_V1
            vgg = models.vgg16(weights=weights).features
        except Exception as e:
            raise RuntimeError(
                "Could not load torchvision VGG16 pretrained weights. "
                "Either install torchvision properly or disable perceptual loss."
            ) from e

        # Common shallow-to-mid feature slices
        self.blocks = nn.ModuleList([
            vgg[:4].eval(),    # relu1_2
            vgg[4:9].eval(),   # relu2_2
            vgg[9:16].eval(),  # relu3_3
        ])

        for block in self.blocks:
            for p in block.parameters():
                p.requires_grad = False

        self.resize_to_224 = resize_to_224

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.resize_to_224:
            pred = F.interpolate(pred, size=(224, 224), mode="bilinear", align_corners=False)
            target = F.interpolate(target, size=(224, 224), mode="bilinear", align_corners=False)

        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        loss = 0.0
        x = pred
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss = loss + F.l1_loss(x, y)
        return loss


# =========================================================
# Full self-supervised cycle loss
# =========================================================

class SelfSupervisedVFILoss(nn.Module):
    """
    Main loss for your cycle-consistency setup.

    Expected usage:
      1. predict I_0.5 from (I0, I1, E01)
      2. predict I_1.5 from (I1, I2, E12)
      3. predict I_1^cyc from (I_0.5, I_1.5, E0.5->1.5)
      4. compare I_1^cyc against real I1

    Optional:
      - pseudo-teacher supervision on I_0.5 and I_1.5
      - flow smoothness on stage A / B / C outputs
    """
    def __init__(
        self,
        lambda_char: float = 1.0,
        lambda_ssim: float = 0.2,
        lambda_perc: float = 0.0,
        lambda_smooth: float = 0.01,
        lambda_pseudo: float = 0.0,
        use_perceptual: bool = False,
    ) -> None:
        super().__init__()
        self.lambda_char = lambda_char
        self.lambda_ssim = lambda_ssim
        self.lambda_perc = lambda_perc
        self.lambda_smooth = lambda_smooth
        self.lambda_pseudo = lambda_pseudo

        self.perceptual = VGGPerceptualLoss() if use_perceptual else None

    def reconstruction_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        char = charbonnier_loss(pred, target)
        ssim = ssim_loss(pred, target)

        if self.perceptual is not None and self.lambda_perc > 0:
            perc = self.perceptual(pred, target)
        else:
            perc = pred.new_tensor(0.0)

        total = (
            self.lambda_char * char
            + self.lambda_ssim * ssim
            + self.lambda_perc * perc
        )

        return {
            "recon_total": total,
            "char": char,
            "ssim": ssim,
            "perc": perc,
        }

    def smoothness_loss_from_motion(
        self,
        motion: Optional[Dict[str, torch.Tensor]],
        image_left: Optional[torch.Tensor],
        image_right: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Uses target-time bilateral flows:
          flow_t0 should be smooth relative to left image
          flow_t1 should be smooth relative to right image
        """
        if motion is None or image_left is None or image_right is None:
            return torch.tensor(0.0, device=image_left.device if image_left is not None else "cpu")

        loss0 = edge_aware_flow_smoothness(motion["flow_t0"], image_left)
        loss1 = edge_aware_flow_smoothness(motion["flow_t1"], image_right)
        return loss0 + loss1

    def pseudo_supervision_loss(
        self,
        pred05: Optional[torch.Tensor],
        pred15: Optional[torch.Tensor],
        teacher05: Optional[torch.Tensor],
        teacher15: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (
            pred05 is None or pred15 is None
            or teacher05 is None or teacher15 is None
            or self.lambda_pseudo <= 0
        ):
            if pred05 is not None:
                return pred05.new_tensor(0.0)
            return torch.tensor(0.0)

        loss05 = charbonnier_loss(pred05, teacher05)
        loss15 = charbonnier_loss(pred15, teacher15)
        return loss05 + loss15

    def forward(
        self,
        pred_center_cyc: torch.Tensor,
        gt_center: torch.Tensor,
        motion_01: Optional[Dict[str, torch.Tensor]] = None,
        motion_12: Optional[Dict[str, torch.Tensor]] = None,
        motion_cyc: Optional[Dict[str, torch.Tensor]] = None,
        img0: Optional[torch.Tensor] = None,
        img1: Optional[torch.Tensor] = None,
        img2: Optional[torch.Tensor] = None,
        pred05: Optional[torch.Tensor] = None,
        pred15: Optional[torch.Tensor] = None,
        teacher05: Optional[torch.Tensor] = None,
        teacher15: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # 1. Main cycle reconstruction loss
        recon = self.reconstruction_loss(pred_center_cyc, gt_center)

        # 2. Flow smoothness for each interpolation stage
        smooth_01 = self.smoothness_loss_from_motion(motion_01, img0, img1) if img0 is not None and img1 is not None else pred_center_cyc.new_tensor(0.0)
        smooth_12 = self.smoothness_loss_from_motion(motion_12, img1, img2) if img1 is not None and img2 is not None else pred_center_cyc.new_tensor(0.0)
        smooth_cyc = self.smoothness_loss_from_motion(motion_cyc, pred05, pred15) if pred05 is not None and pred15 is not None else pred_center_cyc.new_tensor(0.0)

        smooth_total = smooth_01 + smooth_12 + smooth_cyc

        # 3. Optional pseudo-supervision
        pseudo = self.pseudo_supervision_loss(pred05, pred15, teacher05, teacher15)

        total = recon["recon_total"]
        total = total + self.lambda_smooth * smooth_total
        total = total + self.lambda_pseudo * pseudo

        return {
            "loss_total": total,
            "loss_recon": recon["recon_total"],
            "loss_char": recon["char"],
            "loss_ssim": recon["ssim"],
            "loss_perc": recon["perc"],
            "loss_smooth": smooth_total,
            "loss_pseudo": pseudo,
        }
