from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, cast

import torch
import torch.nn as nn

from models.basic_block import ConvAct, ResidualBlock
from models.decoder import WarpingSynthesisDecoder
from models.encoders import EventEncoder, RGBEncoder
from models.loss import SelfSupervisedVFILoss
from models.modal_fusion import MotionHead, PyramidCrossModalFusion
from utils.config import load_config


@dataclass
class EventVFIConfig:
    num_event_bins: int = 9
    rgb_in_ch: int = 3
    rgb_base_ch: int = 32
    event_base_ch: int = 32
    encoder_res_blocks: int = 2
    fusion_use_tau: bool = True
    fusion_res_blocks: int = 2
    motion_res_blocks_per_level: int = 2
    decoder_hidden_ch: int = 64
    decoder_res_blocks: int = 6
    use_warped_rgb_features: bool = True
    clamp_output: bool = True

    @classmethod
    def from_config(cls, section: str = "event_vfi_model") -> "EventVFIConfig":
        """Load an EventVFIConfig from a config.json section via utils/config.py.

        Example:
            cfg = EventVFIConfig.from_config()                         # production
            cfg = EventVFIConfig.from_config("event_vfi_model_smoke")  # small/fast
        """
        raw = load_config(section)
        return cls(
            num_event_bins=int(raw.num_event_bins),
            rgb_in_ch=int(raw.rgb_in_ch),
            rgb_base_ch=int(raw.rgb_base_ch),
            event_base_ch=int(raw.event_base_ch),
            encoder_res_blocks=int(raw.encoder_res_blocks),
            fusion_use_tau=bool(raw.fusion_use_tau),
            fusion_res_blocks=int(raw.fusion_res_blocks),
            motion_res_blocks_per_level=int(raw.motion_res_blocks_per_level),
            decoder_hidden_ch=int(raw.decoder_hidden_ch),
            decoder_res_blocks=int(raw.decoder_res_blocks),
            use_warped_rgb_features=bool(raw.use_warped_rgb_features),
            clamp_output=bool(raw.clamp_output),
        )


class EventVFIModel(nn.Module):
    """End-to-end event-guided VFI model with 3-stage cycle-consistency forward.

    Pipeline:
      RGB encoder -> Event encoder -> Cross-modal fusion -> Motion head -> Warp/synthesis decoder

    Cycle call sequence:
      1) [I0, I1, E01]           -> pred_05    (midpoint I0..I1)
      2) [I1, I2, E12]           -> pred_15    (midpoint I1..I2)
      3) [pred_05, pred_15, E0515] -> pred_1_cyc  (reconstruct I1)

    Loss:  cycle_loss(pred_1_cyc, I1)  --  self-supervised, no GT needed.

    Typical construction:
        # From config.json (production):
        model = EventVFIModel.from_config()

        # From config.json (smoke variant):
        model = EventVFIModel.from_config("event_vfi_model_smoke")

        # Programmatic (tests):
        model = EventVFIModel(EventVFIConfig(rgb_base_ch=8, ...))

    forward() input/output
    ----------------------
    Inputs:
        I0, I1, I2      : (B, 3,        H, W)  boundary RGB frames
        E01, E12, E0515 : (B, num_bins, H, W)  event voxel grids
        tau01/12/0515   : optional (B, 1)       temporal interpolation param
        return_debug    : bool

    Outputs (always present):
        pred_05    : (B, 3, H, W)  interpolated frame between I0 and I1
        pred_15    : (B, 3, H, W)  interpolated frame between I1 and I2
        pred_1_cyc : (B, 3, H, W)  cycle-reconstructed I1

    Per-stage debug dicts (when return_debug=True):
        stage_01, stage_12, stage_cyc -- each contains:
            "pred"      : (B, 3, H, W)
            "motion"    : {flow_t0, flow_t1, mask0, mask1, conf0, conf1}
            "decoder"   : {warped0, warped1, blend, residual, pred,
                           weights0, weights1, valid0, valid1}
            "fused"     : {s1, s2, s4, s8}   cross-modal feature pyramid
            "feat_left" : {s1, s2, s4, s8}   RGB encoder output for I_left
            "feat_right": {s1, s2, s4, s8}   RGB encoder output for I_right
            "evt_feat"  : {s1, s2, s4, s8}   event encoder output
    """

    def __init__(self, cfg: Optional[EventVFIConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or EventVFIConfig()

        self._wire_shared_blocks()

        self.rgb_encoder = RGBEncoder(
            in_ch=self.cfg.rgb_in_ch,
            base_ch=self.cfg.rgb_base_ch,
            num_res_blocks_per_stage=self.cfg.encoder_res_blocks,
        )
        self.event_encoder = EventEncoder(
            num_bins=self.cfg.num_event_bins,
            base_ch=self.cfg.event_base_ch,
            num_res_blocks_per_stage=self.cfg.encoder_res_blocks,
        )

        rgb_ch = {
            "s1": self.cfg.rgb_base_ch,
            "s2": self.cfg.rgb_base_ch * 2,
            "s4": self.cfg.rgb_base_ch * 4,
            "s8": self.cfg.rgb_base_ch * 8,
        }
        evt_ch = {
            "s1": self.cfg.event_base_ch,
            "s2": self.cfg.event_base_ch * 2,
            "s4": self.cfg.event_base_ch * 4,
            "s8": self.cfg.event_base_ch * 8,
        }

        # Use a common fusion width at each scale, tied to RGB channels.
        self.fused_ch = dict(rgb_ch)

        # Keep hidden channels constant across scales to satisfy coarse-to-fine concatenation.
        hidden_const = self.cfg.rgb_base_ch * 2
        self.hidden_ch = {
            "s1": hidden_const,
            "s2": hidden_const,
            "s4": hidden_const,
            "s8": hidden_const,
        }

        self.fusion = PyramidCrossModalFusion(
            rgb_channels=cast(Any, rgb_ch),
            evt_channels=cast(Any, evt_ch),
            out_channels=cast(Any, self.fused_ch),
            use_tau=self.cfg.fusion_use_tau,
            num_res_blocks=self.cfg.fusion_res_blocks,
        )

        self.motion_head = MotionHead(
            fused_channels=cast(Any, self.fused_ch),
            hidden_channels=cast(Any, self.hidden_ch),
            num_res_blocks_per_level=self.cfg.motion_res_blocks_per_level,
        )

        self.decoder = WarpingSynthesisDecoder(
            fused_ch=self.fused_ch["s1"],
            rgb_feat_ch=rgb_ch["s1"],
            hidden_ch=self.cfg.decoder_hidden_ch,
            num_res_blocks=self.cfg.decoder_res_blocks,
            use_warped_rgb_features=self.cfg.use_warped_rgb_features,
            clamp_output=self.cfg.clamp_output,
        )

        self.cycle_loss = SelfSupervisedVFILoss()

    @classmethod
    def from_config(cls, section: str = "event_vfi_model") -> "EventVFIModel":
        """Construct the model directly from a config.json section.

        Example:
            model = EventVFIModel.from_config()
            model = EventVFIModel.from_config("event_vfi_model_smoke")
        """
        return cls(EventVFIConfig.from_config(section))

    @staticmethod
    def _wire_shared_blocks() -> None:
        """Patch modules that reference ConvAct/ResidualBlock without explicit import."""
        from models import decoder as dec_mod
        from models import encoders as enc_mod
        from models import modal_fusion as fusion_mod

        for mod in (enc_mod, fusion_mod, dec_mod):
            setattr(mod, "ConvAct", ConvAct)
            setattr(mod, "ResidualBlock", ResidualBlock)

    @staticmethod
    def _default_tau(batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.full((batch_size, 1), 0.5, device=device, dtype=dtype)

    def _prepare_tau(self, ref: torch.Tensor, tau: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.cfg.fusion_use_tau:
            return None
        if tau is None:
            return self._default_tau(ref.shape[0], ref.device, ref.dtype)
        return tau

    def interpolate_once(
        self,
        I_left: torch.Tensor,
        I_right: torch.Tensor,
        E_window: torch.Tensor,
        tau: Optional[torch.Tensor] = None,
        return_debug: bool = True,
    ) -> Dict[str, Any]:
        """Single interpolation call (one stage of the cycle).

        Inputs:
          I_left, I_right : (B, 3,        H, W)
          E_window        : (B, num_bins, H, W)
          tau             : optional (B, 1), used when fusion_use_tau=True

        Returns dict -- see class docstring for key descriptions.
        """
        tau = self._prepare_tau(I_left, tau)

        # ── 1. Encode ──────────────────────────────────────────────────────
        feat_left  = self.rgb_encoder(I_left)
        feat_right = self.rgb_encoder(I_right)
        evt_feat   = self.event_encoder(E_window)

        # ── 2. Cross-modal fusion (pyramid) ───────────────────────────────
        fused = self.fusion(feat_left, feat_right, evt_feat, tau=tau)

        # ── 3. Coarse-to-fine motion estimation ───────────────────────────
        motion = self.motion_head(fused)

        # ── 4. Warp + blend + residual refinement ─────────────────────────
        dec = self.decoder(
            I0=I_left,
            I1=I_right,
            motion=motion,
            fused_s1=fused["s1"],
            rgb_feat0_s1=feat_left["s1"],
            rgb_feat1_s1=feat_right["s1"],
        )

        out: Dict[str, Any] = {
            "pred":    dec["pred"],
            "motion":  motion,
            "decoder": dec,
        }

        if return_debug:
            out.update({
                "fused":      fused,
                "feat_left":  feat_left,
                "feat_right": feat_right,
                "evt_feat":   evt_feat,
            })
        return out

    def forward(
        self,
        I0: torch.Tensor,
        I1: torch.Tensor,
        I2: torch.Tensor,
        E01: torch.Tensor,
        E12: torch.Tensor,
        E0515: torch.Tensor,
        tau01: Optional[torch.Tensor] = None,
        tau12: Optional[torch.Tensor] = None,
        tau0515: Optional[torch.Tensor] = None,
        return_debug: bool = True,
    ) -> Dict[str, Any]:
        """Three-stage cycle-consistent forward pass.

        Stage A : [I0,      I1,      E01]   -> pred_05
        Stage B : [I1,      I2,      E12]   -> pred_15
        Stage C : [pred_05, pred_15, E0515] -> pred_1_cyc  (cycle reconstruction of I1)
        """
        # Stage A
        stage_01 = self.interpolate_once(I0, I1, E01, tau=tau01, return_debug=return_debug)
        pred_05  = stage_01["pred"]

        # Stage B
        stage_12 = self.interpolate_once(I1, I2, E12, tau=tau12, return_debug=return_debug)
        pred_15  = stage_12["pred"]

        # Stage C  (cycle)
        stage_cyc  = self.interpolate_once(pred_05, pred_15, E0515, tau=tau0515, return_debug=return_debug)
        pred_1_cyc = stage_cyc["pred"]

        return {
            "pred_05":    pred_05,
            "pred_15":    pred_15,
            "pred_1_cyc": pred_1_cyc,
            "stage_01":   stage_01,
            "stage_12":   stage_12,
            "stage_cyc":  stage_cyc,
        }

    def compute_cycle_loss(
        self,
        outputs: Dict[str, Any],
        I0: torch.Tensor,
        I1: torch.Tensor,
        I2: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute self-supervised cycle loss: pred_1_cyc vs real I1.

        Returns dict containing "loss_total" and per-component keys.
        """
        return self.cycle_loss(
            pred_center_cyc=outputs["pred_1_cyc"],
            gt_center=I1,
            motion_01=outputs["stage_01"]["motion"],
            motion_12=outputs["stage_12"]["motion"],
            motion_cyc=outputs["stage_cyc"]["motion"],
            img0=I0,
            img1=I1,
            img2=I2,
            pred05=outputs["pred_05"],
            pred15=outputs["pred_15"],
        )

    @torch.no_grad()
    def cascade_predict(
        self,
        frames: Sequence[torch.Tensor],
        events_between: Sequence[torch.Tensor],
        return_all: bool = False,
    ) -> Dict[str, Any]:
        """Cascading inference through an N-frame sequence.

        Args:
            frames         : list of N tensors (B, 3, H, W)
            events_between : list of N-1 event voxel grids (B, num_bins, H, W)
            return_all     : if True, also include per-step debug dicts

        Returns:
            {"predictions": [pred_01, pred_12, ...]}
            + {"debug": [...]} when return_all=True
        """
        if len(frames) < 2 or len(events_between) != len(frames) - 1:
            raise ValueError("Need N frames and N-1 event windows")

        preds: list[torch.Tensor] = []
        debug: list[Dict[str, Any]] = []
        for i in range(len(events_between)):
            step = self.interpolate_once(
                frames[i], frames[i + 1], events_between[i], return_debug=return_all
            )
            preds.append(step["pred"])
            if return_all:
                debug.append(step)

        out: Dict[str, Any] = {"predictions": preds}
        if return_all:
            out["debug"] = debug
        return out
