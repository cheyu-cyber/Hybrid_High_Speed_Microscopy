from __future__ import annotations

from pathlib import Path

from pyparsing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import load_config

def upsample_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        x,
        size=ref.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


def upsample_flow_like(flow: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Upsample flow to match ref spatial size, then scale flow magnitude
    to the new pixel grid.

    Example:
        1 pixel at H/8 scale becomes 2 pixels at H/4 scale.
    """
    old_h, old_w = flow.shape[-2:]
    new_h, new_w = ref.shape[-2:]

    up = F.interpolate(flow, size=(new_h, new_w), mode="bilinear", align_corners=False)

    scale_x = float(new_w) / float(old_w)
    scale_y = float(new_h) / float(old_h)

    up[:, 0:1] *= scale_x
    up[:, 1:2] *= scale_y
    return up

class ChannelGate(nn.Module):
    """
    Simple squeeze-excitation style channel gate.
    """
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.mlp(self.pool(x))
        return x * w

class CrossModalFusionBlock(nn.Module):
    """
    Fuse RGB(I0), RGB(I1), and event features at one scale.

    Inputs:
        f0:   (B, rgb_ch, H, W)
        f1:   (B, rgb_ch, H, W)
        fe:   (B, evt_ch, H, W)
        tau:  optional scalar tensor of shape (B,) or (B,1)

    Output:
        fused: (B, out_ch, H, W)

    Design:
        1. project each modality to same hidden size
        2. build a motion cue with |f1 - f0|
        3. concatenate projected features
        4. fuse with 1x1 bottleneck + residual blocks
        5. apply channel gate
    """
    def __init__(
        self,
        rgb_ch: int,
        evt_ch: int,
        out_ch: int,
        use_tau: bool = True,
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.use_tau = use_tau
        self.out_ch = out_ch

        self.rgb0_proj = ConvAct(rgb_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.rgb1_proj = ConvAct(rgb_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.evt_proj = ConvAct(evt_ch, out_ch, kernel_size=1, stride=1, padding=0)

        in_ch = out_ch * 4  # f0, f1, fe, |f1-f0|
        if use_tau:
            in_ch += 1

        self.fuse = nn.Sequential(
            ConvAct(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
            ResidualBlock(out_ch),
            *[ResidualBlock(out_ch) for _ in range(num_res_blocks - 1)],
        )

        self.gate = ChannelGate(out_ch, reduction=8)

    def forward(
        self,
        f0: torch.Tensor,
        f1: torch.Tensor,
        fe: torch.Tensor,
        tau: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if f0.shape[0] != f1.shape[0] or f0.shape[0] != fe.shape[0]:
            raise ValueError("Batch sizes of f0, f1, fe must match")
        if f0.shape[2:] != f1.shape[2:] or f0.shape[2:] != fe.shape[2:]:
            raise ValueError("Spatial sizes of f0, f1, fe must match")

        p0 = self.rgb0_proj(f0)
        p1 = self.rgb1_proj(f1)
        pe = self.evt_proj(fe)

        diff = torch.abs(p1 - p0)

        parts = [p0, p1, pe, diff]

        if self.use_tau:
            if tau is None:
                raise ValueError("tau must be provided when use_tau=True")

            if tau.ndim == 1:
                tau = tau[:, None]
            if tau.ndim != 2 or tau.shape[1] != 1:
                raise ValueError("tau must have shape (B,) or (B,1)")

            b, _, h, w = p0.shape
            tau_map = tau[:, :, None, None].expand(b, 1, h, w)
            parts.append(tau_map)

        x = torch.cat(parts, dim=1)
        x = self.fuse(x)
        x = self.gate(x)
        return x

class PyramidCrossModalFusion(nn.Module):
    """
    Fuse multiscale pyramids from:
        feat0: RGB features from I0
        feat1: RGB features from I1
        evt_feat: event features

    Expected keys:
        "s1", "s2", "s4", "s8"
    """
    def __init__(
        self,
        rgb_channels: Dict[str, int],
        evt_channels: Dict[str, int],
        out_channels: Dict[str, int],
        use_tau: bool = True,
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()

        scales = ["s1", "s2", "s4", "s8"]
        for k in scales:
            if k not in rgb_channels or k not in evt_channels or k not in out_channels:
                raise ValueError(f"Missing channel config for scale {k}")

        self.blocks = nn.ModuleDict({
            k: CrossModalFusionBlock(
                rgb_ch=rgb_channels[k],
                evt_ch=evt_channels[k],
                out_ch=out_channels[k],
                use_tau=use_tau,
                num_res_blocks=num_res_blocks,
            )
            for k in scales
        })

    def forward(
        self,
        feat0: Dict[str, torch.Tensor],
        feat1: Dict[str, torch.Tensor],
        evt_feat: Dict[str, torch.Tensor],
        tau: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        fused = {}
        for k, block in self.blocks.items():
            fused[k] = block(feat0[k], feat1[k], evt_feat[k], tau=tau)
        return fused

# ---------------------------------------------------------
# One coarse-to-fine refinement level
# ---------------------------------------------------------

class MotionHeadLevel(nn.Module):
    """
    One scale of the motion head.

    Inputs:
        fused:      fused cross-modal feature at this scale
        prev_hidden previous hidden state from coarser scale
        prev_pred   previous prediction dict from coarser scale

    Outputs:
        hidden
        pred dict with:
            flow_t0      (B,2,H,W)
            flow_t1      (B,2,H,W)
            mask_logits  (B,2,H,W)
            conf_logits  (B,2,H,W)
    """
    def __init__(
        self,
        fused_ch: int,
        hidden_ch: int,
        num_res_blocks: int = 2,
        use_prev: bool = True,
    ) -> None:
        super().__init__()
        self.use_prev = use_prev
        self.hidden_ch = hidden_ch

        prev_hidden_ch = hidden_ch if use_prev else 0
        prev_pred_ch = 8 if use_prev else 0  # flow_t0(2) + flow_t1(2) + mask_logits(2) + conf_logits(2)

        in_ch = fused_ch + prev_hidden_ch + prev_pred_ch

        layers = [ConvAct(in_ch, hidden_ch, 3, 1, 1)]
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(hidden_ch))
        self.body = nn.Sequential(*layers)

        self.flow_t0_head = nn.Conv2d(hidden_ch, 2, kernel_size=3, stride=1, padding=1)
        self.flow_t1_head = nn.Conv2d(hidden_ch, 2, kernel_size=3, stride=1, padding=1)
        self.mask_head = nn.Conv2d(hidden_ch, 2, kernel_size=3, stride=1, padding=1)
        self.conf_head = nn.Conv2d(hidden_ch, 2, kernel_size=3, stride=1, padding=1)

    def forward(
        self,
        fused: torch.Tensor,
        prev_hidden: Optional[torch.Tensor] = None,
        prev_pred: Optional[Dict[str, torch.Tensor]] = None,
    ):
        parts = [fused]

        if self.use_prev:
            if prev_hidden is None or prev_pred is None:
                raise ValueError("prev_hidden and prev_pred are required when use_prev=True")

            prev_hidden_up = upsample_like(prev_hidden, fused)

            prev_flow_t0_up = upsample_flow_like(prev_pred["flow_t0"], fused)
            prev_flow_t1_up = upsample_flow_like(prev_pred["flow_t1"], fused)
            prev_mask_logits_up = upsample_like(prev_pred["mask_logits"], fused)
            prev_conf_logits_up = upsample_like(prev_pred["conf_logits"], fused)

            prev_pred_cat = torch.cat(
                [prev_flow_t0_up, prev_flow_t1_up, prev_mask_logits_up, prev_conf_logits_up],
                dim=1,
            )

            parts.extend([prev_hidden_up, prev_pred_cat])

        x = torch.cat(parts, dim=1)
        hidden = self.body(x)

        delta_flow_t0 = self.flow_t0_head(hidden)
        delta_flow_t1 = self.flow_t1_head(hidden)
        delta_mask_logits = self.mask_head(hidden)
        delta_conf_logits = self.conf_head(hidden)

        if self.use_prev:
            flow_t0 = prev_flow_t0_up + delta_flow_t0
            flow_t1 = prev_flow_t1_up + delta_flow_t1
            mask_logits = prev_mask_logits_up + delta_mask_logits
            conf_logits = prev_conf_logits_up + delta_conf_logits
        else:
            flow_t0 = delta_flow_t0
            flow_t1 = delta_flow_t1
            mask_logits = delta_mask_logits
            conf_logits = delta_conf_logits

        pred = {
            "flow_t0": flow_t0,
            "flow_t1": flow_t1,
            "mask_logits": mask_logits,
            "conf_logits": conf_logits,
        }
        return hidden, pred


# ---------------------------------------------------------
# Full pyramid motion head
# ---------------------------------------------------------

class MotionHead(nn.Module):
    """
    Coarse-to-fine motion head for fused pyramid features.

    Expected fused pyramid keys:
        "s1", "s2", "s4", "s8"

    Example channel config:
        s1: 32
        s2: 64
        s4: 128
        s8: 256
    """
    def __init__(
        self,
        fused_channels: Dict[str, int],
        hidden_channels: Dict[str, int],
        num_res_blocks_per_level: int = 2,
    ) -> None:
        super().__init__()

        required = ["s1", "s2", "s4", "s8"]
        for k in required:
            if k not in fused_channels or k not in hidden_channels:
                raise ValueError(f"Missing config for scale {k}")

        # coarse -> fine
        self.level_s8 = MotionHeadLevel(
            fused_ch=fused_channels["s8"],
            hidden_ch=hidden_channels["s8"],
            num_res_blocks=num_res_blocks_per_level,
            use_prev=False,
        )
        self.level_s4 = MotionHeadLevel(
            fused_ch=fused_channels["s4"],
            hidden_ch=hidden_channels["s4"],
            num_res_blocks=num_res_blocks_per_level,
            use_prev=True,
        )
        self.level_s2 = MotionHeadLevel(
            fused_ch=fused_channels["s2"],
            hidden_ch=hidden_channels["s2"],
            num_res_blocks=num_res_blocks_per_level,
            use_prev=True,
        )
        self.level_s1 = MotionHeadLevel(
            fused_ch=fused_channels["s1"],
            hidden_ch=hidden_channels["s1"],
            num_res_blocks=num_res_blocks_per_level,
            use_prev=True,
        )

    def forward(self, fused: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # coarse
        h8, p8 = self.level_s8(fused["s8"])

        # refine
        h4, p4 = self.level_s4(fused["s4"], prev_hidden=h8, prev_pred=p8)
        h2, p2 = self.level_s2(fused["s2"], prev_hidden=h4, prev_pred=p4)
        h1, p1 = self.level_s1(fused["s1"], prev_hidden=h2, prev_pred=p2)

        # Convert logits to actual masks / confidences at finest scale.
        # Softmax masks make M0 + M1 = 1, which is a practical version of complementary visibility.
        masks = torch.softmax(p1["mask_logits"], dim=1)
        conf = torch.sigmoid(p1["conf_logits"])

        out = {
            # finest scale predictions for warping at full resolution
            "flow_t0": p1["flow_t0"],              # (B,2,H,W)
            "flow_t1": p1["flow_t1"],              # (B,2,H,W)
            "mask0": masks[:, 0:1],                # (B,1,H,W)
            "mask1": masks[:, 1:2],                # (B,1,H,W)
            "conf0": conf[:, 0:1],                 # (B,1,H,W)
            "conf1": conf[:, 1:2],                 # (B,1,H,W)

            # optional raw logits for losses or debugging
            "mask_logits": p1["mask_logits"],
            "conf_logits": p1["conf_logits"],

            # optional multi-scale predictions for deep supervision
            "pred_s8": p8,
            "pred_s4": p4,
            "pred_s2": p2,
            "pred_s1": p1,
        }
        return out