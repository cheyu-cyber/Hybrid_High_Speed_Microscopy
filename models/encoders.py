from __future__ import annotations

from pathlib import Path

from pyparsing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import load_config

class EncoderStage(nn.Module):
    """
    One pyramid stage.
    If downsample=True, spatial size is reduced by 2.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_res_blocks: int = 2,
        downsample: bool = True,
    ) -> None:
        super().__init__()

        stride = 2 if downsample else 1
        self.entry = ConvAct(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
        )

        blocks = [ResidualBlock(out_ch) for _ in range(num_res_blocks)]
        self.body = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.entry(x)
        x = self.body(x)
        return x
    
class RGBEncoder(nn.Module):
    """
    Shared RGB encoder for one input frame.

    Input:
        x: (B, 3, H, W)

    Output:
        dict of multiscale features:
            s1: (B, base_ch,     H,   W)
            s2: (B, base_ch*2, H/2, W/2)
            s4: (B, base_ch*4, H/4, W/4)
            s8: (B, base_ch*8, H/8, W/8)
    """
    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 32,
        num_res_blocks_per_stage: int = 2,
    ) -> None:
        super().__init__()

        # Stem at full resolution
        self.stem = nn.Sequential(
            ConvAct(in_ch, base_ch, kernel_size=3, stride=1, padding=1),
            ConvAct(base_ch, base_ch, kernel_size=3, stride=1, padding=1),
        )

        # Full-resolution refinement
        self.stage1 = nn.Sequential(
            ResidualBlock(base_ch),
            ResidualBlock(base_ch),
        )

        # Pyramid stages
        self.stage2 = EncoderStage(
            in_ch=base_ch,
            out_ch=base_ch * 2,
            num_res_blocks=num_res_blocks_per_stage,
            downsample=True,
        )

        self.stage3 = EncoderStage(
            in_ch=base_ch * 2,
            out_ch=base_ch * 4,
            num_res_blocks=num_res_blocks_per_stage,
            downsample=True,
        )

        self.stage4 = EncoderStage(
            in_ch=base_ch * 4,
            out_ch=base_ch * 8,
            num_res_blocks=num_res_blocks_per_stage,
            downsample=True,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got shape {tuple(x.shape)}")
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 RGB channels, got {x.shape[1]}")

        s1 = self.stem(x)
        s1 = self.stage1(s1)

        s2 = self.stage2(s1)
        s4 = self.stage3(s2)
        s8 = self.stage4(s4)

        return {
            "s1": s1,
            "s2": s2,
            "s4": s4,
            "s8": s8,
        }    

class EventEncoder(nn.Module):
    """
    Event encoder for voxel-grid event input.

    Input:
        evt: (B, num_bins, H, W)

    Output:
        {
            "s1": (B, base_ch,     H,   W),
            "s2": (B, base_ch*2, H/2, W/2),
            "s4": (B, base_ch*4, H/4, W/4),
            "s8": (B, base_ch*8, H/8, W/8),
        }
    """
    def __init__(
        self,
        num_bins: int = 9,
        base_ch: int = 32,
        num_res_blocks_per_stage: int = 2,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins

        # Full-resolution stem
        self.stem = nn.Sequential(
            ConvAct(num_bins, base_ch, kernel_size=3, stride=1, padding=1),
            ConvAct(base_ch, base_ch, kernel_size=3, stride=1, padding=1),
        )

        # Full-resolution refinement
        self.stage1 = nn.Sequential(
            ResidualBlock(base_ch),
            ResidualBlock(base_ch),
        )

        # Pyramid stages
        self.stage2 = EncoderStage(
            in_ch=base_ch,
            out_ch=base_ch * 2,
            num_res_blocks=num_res_blocks_per_stage,
            downsample=True,
        )

        self.stage3 = EncoderStage(
            in_ch=base_ch * 2,
            out_ch=base_ch * 4,
            num_res_blocks=num_res_blocks_per_stage,
            downsample=True,
        )

        self.stage4 = EncoderStage(
            in_ch=base_ch * 4,
            out_ch=base_ch * 8,
            num_res_blocks=num_res_blocks_per_stage,
            downsample=True,
        )

    def forward(self, evt: torch.Tensor) -> Dict[str, torch.Tensor]:
        if evt.ndim != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got {tuple(evt.shape)}")
        if evt.shape[1] != self.num_bins:
            raise ValueError(
                f"Expected {self.num_bins} event bins, got {evt.shape[1]}"
            )

        s1 = self.stem(evt)
        s1 = self.stage1(s1)

        s2 = self.stage2(s1)
        s4 = self.stage3(s2)
        s8 = self.stage4(s4)

        return {
            "s1": s1,
            "s2": s2,
            "s4": s4,
            "s8": s8,
        }

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
   
def encode_frame_pair(
    encoder: RGBEncoder,
    I0: torch.Tensor,
    I1: torch.Tensor,
):
    """
    Encode two boundary RGB frames with shared weights.
    """
    feat0 = encoder(I0)
    feat1 = encoder(I1)
    return feat0, feat1




