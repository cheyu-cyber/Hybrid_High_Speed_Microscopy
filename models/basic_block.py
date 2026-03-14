from __future__ import annotations

from pathlib import Path

from pyparsing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import load_config

class ConvAct(nn.Module):
    """
    Conv + activation.
    Kept simple on purpose so it is easy to modify later.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: str = "lrelu",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

        if activation == "lrelu":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif activation == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))
    
class ResidualBlock(nn.Module):
    """
    A small residual block for low-level vision features.
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvAct(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        return x + out