"""Event-guided video frame interpolation model for hybrid high-speed microscopy.

Architecture (inspired by TimeReplayer, CVPR 2022):
  1. Event Voxel Grid   — converts raw events into a fixed-size (B, H, W) tensor
  2. Flow Estimation     — U-Net encoder-decoder predicts bidirectional optical flow
  3. Warping & Synthesis — warps keyframes by estimated flow, then fuses into output

All hyperparameters are read from config.json section "model".
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import load_config
from datasets.event_processing import events_to_voxel_grid


# ---------------------------------------------------------------------------
# 1. Optical-flow backward warping  (torch)
# ---------------------------------------------------------------------------

def warp(image, flow):
    """Backward-warp an image using optical flow.

    Parameters
    ----------
    image : (N, C, H, W) torch.Tensor
    flow  : (N, 2, H, W) torch.Tensor — (dx, dy)

    Returns
    -------
    warped : (N, C, H, W) torch.Tensor
    """
    import torch
    import torch.nn.functional as F

    N, _, H, W = image.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=image.device, dtype=torch.float32),
        torch.arange(W, device=image.device, dtype=torch.float32),
        indexing="ij",
    )
    grid_x = grid_x.unsqueeze(0).expand(N, -1, -1) + flow[:, 0]
    grid_y = grid_y.unsqueeze(0).expand(N, -1, -1) + flow[:, 1]

    # Normalise to [-1, 1] for grid_sample
    grid_x = 2.0 * grid_x / (W - 1) - 1.0
    grid_y = 2.0 * grid_y / (H - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (N, H, W, 2)

    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border",
                         align_corners=True)

# ---------------------------------------------------------------------------
# 2. Flow estimation — lightweight U-Net
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    """Conv -> BatchNorm -> LeakyReLU, repeated twice."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = _ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if sizes don't match exactly
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class FlowEstimator(nn.Module):
    """U-Net that takes an event voxel grid and outputs 2-channel optical flow."""

    def __init__(self, in_channels: int, base_channels: int = 32, depth: int = 4):
        super().__init__()
        ch = base_channels

        # Encoder
        self.enc0 = _ConvBlock(in_channels, ch)
        self.encoders = nn.ModuleList()
        in_ch = ch
        for i in range(depth):
            out_ch = ch * (2 ** (i + 1))
            self.encoders.append(_DownBlock(in_ch, out_ch))
            in_ch = out_ch

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            out_ch = ch * (2 ** i) if i > 0 else ch
            self.decoders.append(_UpBlock(in_ch, out_ch))
            in_ch = out_ch

        # Flow head: 2 channels (dx, dy)
        self.flow_head = nn.Conv2d(ch, 2, 1)

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        voxel : (N, B, H, W) event voxel grid

        Returns
        -------
        flow : (N, 2, H, W) predicted optical flow
        """
        skips = []
        x = self.enc0(voxel)
        skips.append(x)
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        skips.pop()  # bottom level has no skip
        for dec in self.decoders:
            x = dec(x, skips.pop())

        return self.flow_head(x)


# ---------------------------------------------------------------------------
# 3. Synthesis / fusion network
# ---------------------------------------------------------------------------

class SynthesisNet(nn.Module):
    """Fuse two warped frames + event voxel into a final output frame."""

    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()
        # Input: 2 warped frames (each C channels) + voxel (B bins) = 2*C + B
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # Output: blending weight map (1 ch) + residual (C ch)
        )
        self.alpha_head = nn.Sequential(
            nn.Conv2d(base_channels, 1, 1),
            nn.Sigmoid(),
        )
        self.residual_head = nn.Conv2d(base_channels, 0, 1)  # placeholder, set in build

    def build(self, image_channels: int, base_channels: int):
        self.residual_head = nn.Conv2d(base_channels, image_channels, 1)

    def forward(self, warped_0: torch.Tensor, warped_1: torch.Tensor,
                voxel: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        warped_0, warped_1 : (N, C, H, W) warped keyframes
        voxel              : (N, B, H, W) event voxel grid

        Returns
        -------
        output : (N, C, H, W) interpolated frame
        """
        x = torch.cat([warped_0, warped_1, voxel], dim=1)
        feat = self.net(x)
        alpha = self.alpha_head(feat)
        residual = self.residual_head(feat)
        blended = alpha * warped_0 + (1 - alpha) * warped_1
        return blended + residual


# ---------------------------------------------------------------------------
# 5. Full model
# ---------------------------------------------------------------------------

class EventFrameInterpolator(nn.Module):
    """Complete event-guided frame interpolation model.

    Given two keyframes I_0, I_1 and an event voxel grid between them,
    produces an interpolated frame at time t ∈ (0, 1).
    """

    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = load_config("model")

        self.num_bins = int(cfg.num_bins)
        self.image_channels = int(cfg.image_channels)
        base_ch = int(cfg.base_channels)
        depth = int(cfg.unet_depth)

        # Flow estimator: input = voxel grid (B bins) + 2 keyframes
        flow_in = self.num_bins + 2 * self.image_channels
        self.flow_net = FlowEstimator(flow_in, base_channels=base_ch, depth=depth)

        # Synthesis: input = 2 warped frames + voxel
        synth_in = 2 * self.image_channels + self.num_bins
        self.synthesis = SynthesisNet(synth_in, base_channels=base_ch)
        self.synthesis.build(self.image_channels, base_ch)

    def forward(self, frame_0: torch.Tensor, frame_1: torch.Tensor,
                voxel_0_t: torch.Tensor, voxel_t_1: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        frame_0   : (N, C, H, W) keyframe at t=0
        frame_1   : (N, C, H, W) keyframe at t=1
        voxel_0_t : (N, B, H, W) event voxel grid from t=0 to target t
        voxel_t_1 : (N, B, H, W) event voxel grid from target t to t=1

        Returns
        -------
        frame_t : (N, C, H, W) interpolated frame at target time
        """
        # Estimate forward and backward flow
        flow_0_t = self.flow_net(torch.cat([voxel_0_t, frame_0, frame_1], dim=1))
        flow_1_t = self.flow_net(torch.cat([voxel_t_1, frame_1, frame_0], dim=1))

        # Warp keyframes to target time
        warped_0 = warp(frame_0, flow_0_t)
        warped_1 = warp(frame_1, flow_1_t)

        # Fuse
        return self.synthesis(warped_0, warped_1, voxel_0_t)
