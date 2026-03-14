import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

# ---------------------------------------------------------
# Backward warping
# ---------------------------------------------------------

def make_base_grid(
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Returns base grid in pixel coordinates:
        grid[..., 0] = x
        grid[..., 1] = y
    shape: (B, H, W, 2)
    """
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack([xs, ys], dim=-1)  # (H, W, 2)
    grid = grid.unsqueeze(0).repeat(batch, 1, 1, 1)
    return grid


def pixel_grid_to_normalized(grid_xy: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Convert pixel-coordinate grid to normalized [-1, 1] grid for grid_sample.
    grid_xy: (B, H, W, 2)
    """
    x = grid_xy[..., 0]
    y = grid_xy[..., 1]

    if width > 1:
        x = 2.0 * x / (width - 1) - 1.0
    else:
        x = torch.zeros_like(x)

    if height > 1:
        y = 2.0 * y / (height - 1) - 1.0
    else:
        y = torch.zeros_like(y)

    return torch.stack([x, y], dim=-1)


def backward_warp(
    src: torch.Tensor,
    flow_t_to_s: torch.Tensor,
    padding_mode: str = "border",
) -> torch.Tensor:
    """
    Backward warp.

    Args:
        src:
            source image or feature map, shape (B, C, H, W)
        flow_t_to_s:
            target-to-source flow, shape (B, 2, H, W)
            flow[:, 0] = dx
            flow[:, 1] = dy

    Returns:
        warped source sampled at target grid, shape (B, C, H, W)

    Meaning:
        output(y, x) = src(y + dy, x + dx)
    """
    if src.ndim != 4 or flow_t_to_s.ndim != 4:
        raise ValueError("src and flow_t_to_s must both be 4D tensors")
    if flow_t_to_s.shape[1] != 2:
        raise ValueError("flow_t_to_s must have 2 channels")
    if src.shape[0] != flow_t_to_s.shape[0] or src.shape[-2:] != flow_t_to_s.shape[-2:]:
        raise ValueError("src and flow_t_to_s must have matching batch and spatial size")

    b, _, h, w = src.shape
    base_grid = make_base_grid(b, h, w, src.device, src.dtype)  # (B, H, W, 2)

    flow_xy = flow_t_to_s.permute(0, 2, 3, 1)  # (B, H, W, 2)
    sample_grid = base_grid + flow_xy
    sample_grid = pixel_grid_to_normalized(sample_grid, h, w)

    warped = F.grid_sample(
        src,
        sample_grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )
    return warped


def warp_valid_mask(flow_t_to_s: torch.Tensor) -> torch.Tensor:
    """
    Returns a soft binary valid mask indicating whether warped coordinates
    stay inside the source image bounds.

    shape: (B, 1, H, W)
    """
    if flow_t_to_s.ndim != 4 or flow_t_to_s.shape[1] != 2:
        raise ValueError("flow_t_to_s must have shape (B,2,H,W)")

    b, _, h, w = flow_t_to_s.shape
    base_grid = make_base_grid(b, h, w, flow_t_to_s.device, flow_t_to_s.dtype)
    flow_xy = flow_t_to_s.permute(0, 2, 3, 1)
    sample_grid = base_grid + flow_xy

    x = sample_grid[..., 0]
    y = sample_grid[..., 1]

    valid = (x >= 0) & (x <= (w - 1)) & (y >= 0) & (y <= (h - 1))
    return valid.float().unsqueeze(1)


# ---------------------------------------------------------
# Refinement decoder
# ---------------------------------------------------------

class RefinementDecoder(nn.Module):
    """
    Small residual refinement network.

    Input channels are flexible so you can include:
      - warped RGB frames
      - blended frame
      - masks / confidences / valid masks
      - fused motion feature
      - warped source features
    """
    def __init__(
        self,
        in_ch: int,
        hidden_ch: int = 64,
        num_res_blocks: int = 6,
    ) -> None:
        super().__init__()

        layers = [
            ConvAct(in_ch, hidden_ch, 3, 1, 1),
            ConvAct(hidden_ch, hidden_ch, 3, 1, 1),
        ]
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(hidden_ch))

        self.body = nn.Sequential(*layers)
        self.to_rgb = nn.Conv2d(hidden_ch, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.body(x)
        residual = self.to_rgb(feat)
        return residual


# ---------------------------------------------------------
# Full warp + blend + refine decoder
# ---------------------------------------------------------

class WarpingSynthesisDecoder(nn.Module):
    """
    Inputs:
        I0, I1:            source RGB frames, (B,3,H,W)
        motion: dict with
            flow_t0:       (B,2,H,W)
            flow_t1:       (B,2,H,W)
            mask0:         (B,1,H,W)
            mask1:         (B,1,H,W)
            conf0:         (B,1,H,W)
            conf1:         (B,1,H,W)

        Optional:
            fused_s1:      fused cross-modal feature at full resolution
            rgb_feat0_s1:  RGB feature from I0 at s1
            rgb_feat1_s1:  RGB feature from I1 at s1

    Output dict:
        warped0, warped1
        blend
        residual
        pred
        weights0, weights1
        valid0, valid1
    """
    def __init__(
        self,
        fused_ch: int = 0,
        rgb_feat_ch: int = 0,
        hidden_ch: int = 64,
        num_res_blocks: int = 6,
        use_warped_rgb_features: bool = False,
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        self.use_warped_rgb_features = use_warped_rgb_features
        self.clamp_output = clamp_output

        # Base decoder inputs:
        # warped0(3), warped1(3), blend(3),
        # mask0(1), mask1(1), conf0(1), conf1(1),
        # valid0(1), valid1(1)
        in_ch = 3 + 3 + 3 + 1 + 1 + 1 + 1 + 1 + 1

        if fused_ch > 0:
            in_ch += fused_ch

        if use_warped_rgb_features:
            if rgb_feat_ch <= 0:
                raise ValueError("rgb_feat_ch must be > 0 when use_warped_rgb_features=True")
            in_ch += rgb_feat_ch + rgb_feat_ch

        self.refine = RefinementDecoder(
            in_ch=in_ch,
            hidden_ch=hidden_ch,
            num_res_blocks=num_res_blocks,
        )

    def forward(
        self,
        I0: torch.Tensor,
        I1: torch.Tensor,
        motion: Dict[str, torch.Tensor],
        fused_s1: Optional[torch.Tensor] = None,
        rgb_feat0_s1: Optional[torch.Tensor] = None,
        rgb_feat1_s1: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        flow_t0 = motion["flow_t0"]
        flow_t1 = motion["flow_t1"]
        mask0 = motion["mask0"]
        mask1 = motion["mask1"]
        conf0 = motion["conf0"]
        conf1 = motion["conf1"]

        # 1. Backward warp RGB frames
        warped0 = backward_warp(I0, flow_t0)
        warped1 = backward_warp(I1, flow_t1)

        # 2. Validity masks from flow bounds
        valid0 = warp_valid_mask(flow_t0)
        valid1 = warp_valid_mask(flow_t1)

        # 3. Confidence-aware blending weights
        weights0 = mask0 * conf0 * valid0
        weights1 = mask1 * conf1 * valid1

        denom = weights0 + weights1 + 1e-6
        blend = (weights0 * warped0 + weights1 * warped1) / denom

        # 4. Build refinement input
        parts = [
            warped0,
            warped1,
            blend,
            mask0,
            mask1,
            conf0,
            conf1,
            valid0,
            valid1,
        ]

        if fused_s1 is not None:
            parts.append(fused_s1)

        if self.use_warped_rgb_features:
            if rgb_feat0_s1 is None or rgb_feat1_s1 is None:
                raise ValueError("Need rgb_feat0_s1 and rgb_feat1_s1 when use_warped_rgb_features=True")

            wfeat0 = backward_warp(rgb_feat0_s1, flow_t0)
            wfeat1 = backward_warp(rgb_feat1_s1, flow_t1)
            parts.extend([wfeat0, wfeat1])

        refine_in = torch.cat(parts, dim=1)

        # 5. Residual correction
        residual = self.refine(refine_in)
        pred = blend + residual

        if self.clamp_output:
            pred = pred.clamp(0.0, 1.0)

        return {
            "warped0": warped0,
            "warped1": warped1,
            "valid0": valid0,
            "valid1": valid1,
            "weights0": weights0,
            "weights1": weights1,
            "blend": blend,
            "residual": residual,
            "pred": pred,
        }
