import importlib

import pytest
import torch

from models import basic_block, decoder, encoders, modal_fusion


def _inject_basic_blocks() -> None:
    """Attach shared building blocks into modules that reference them."""
    for mod in (encoders, modal_fusion, decoder):
        mod.ConvAct = basic_block.ConvAct
        mod.ResidualBlock = basic_block.ResidualBlock


@pytest.fixture(autouse=True)
def _patch_model_modules():
    _inject_basic_blocks()


# ------------------------------
# basic_block.py
# ------------------------------

def test_convact_output_shape_and_activation():
    m = basic_block.ConvAct(3, 8, kernel_size=3, stride=1, padding=1, activation="relu")
    x = torch.randn(2, 3, 16, 16)
    y = m(x)
    assert y.shape == (2, 8, 16, 16)


def test_convact_invalid_activation_raises():
    with pytest.raises(ValueError, match="Unsupported activation"):
        basic_block.ConvAct(3, 8, activation="gelu")


def test_residual_block_preserves_shape():
    m = basic_block.ResidualBlock(8)
    x = torch.randn(2, 8, 12, 10)
    y = m(x)
    assert y.shape == x.shape


# ------------------------------
# encoders.py
# ------------------------------

def test_encoder_stage_downsample_and_no_downsample():
    x = torch.randn(2, 8, 32, 24)

    ds = encoders.EncoderStage(in_ch=8, out_ch=16, downsample=True)
    y_ds = ds(x)
    assert y_ds.shape == (2, 16, 16, 12)

    no_ds = encoders.EncoderStage(in_ch=8, out_ch=16, downsample=False)
    y_no_ds = no_ds(x)
    assert y_no_ds.shape == (2, 16, 32, 24)


def test_rgb_encoder_output_pyramid_shapes():
    m = encoders.RGBEncoder(in_ch=3, base_ch=8, num_res_blocks_per_stage=1)
    x = torch.randn(2, 3, 32, 24)
    out = m(x)

    assert set(out.keys()) == {"s1", "s2", "s4", "s8"}
    assert out["s1"].shape == (2, 8, 32, 24)
    assert out["s2"].shape == (2, 16, 16, 12)
    assert out["s4"].shape == (2, 32, 8, 6)
    assert out["s8"].shape == (2, 64, 4, 3)


def test_rgb_encoder_bad_inputs_raise():
    m = encoders.RGBEncoder(base_ch=8, num_res_blocks_per_stage=1)

    with pytest.raises(ValueError, match="Expected 4D tensor"):
        m(torch.randn(3, 16, 16))

    with pytest.raises(ValueError, match="Expected 3 RGB channels"):
        m(torch.randn(2, 1, 16, 16))


def test_event_encoder_output_and_bad_channels_raise():
    m = encoders.EventEncoder(num_bins=5, base_ch=8, num_res_blocks_per_stage=1)

    x = torch.randn(2, 5, 32, 24)
    out = m(x)
    assert out["s1"].shape == (2, 8, 32, 24)
    assert out["s8"].shape == (2, 64, 4, 3)

    with pytest.raises(ValueError, match="Expected 5 event bins"):
        m(torch.randn(2, 6, 32, 24))


def test_encode_frame_pair_returns_two_feature_dicts():
    m = encoders.RGBEncoder(base_ch=8, num_res_blocks_per_stage=1)
    i0 = torch.randn(2, 3, 32, 24)
    i1 = torch.randn(2, 3, 32, 24)

    f0, f1 = encoders.encode_frame_pair(m, i0, i1)
    assert set(f0.keys()) == {"s1", "s2", "s4", "s8"}
    assert f0["s4"].shape == f1["s4"].shape == (2, 32, 8, 6)


# ------------------------------
# modal_fusion.py
# ------------------------------

def test_upsample_like_and_upsample_flow_like_scale_components():
    x = torch.randn(1, 4, 4, 4)
    ref = torch.randn(1, 4, 8, 12)
    up = modal_fusion.upsample_like(x, ref)
    assert up.shape[-2:] == (8, 12)

    flow = torch.zeros(1, 2, 4, 4)
    flow[:, 0] = 1.0
    flow[:, 1] = 2.0
    upf = modal_fusion.upsample_flow_like(flow, ref)

    assert upf.shape == (1, 2, 8, 12)
    assert torch.allclose(upf[:, 0].mean(), torch.tensor(3.0), atol=1e-5)
    assert torch.allclose(upf[:, 1].mean(), torch.tensor(4.0), atol=1e-5)


def test_cross_modal_fusion_block_forward_and_tau_validation():
    m = modal_fusion.CrossModalFusionBlock(rgb_ch=8, evt_ch=6, out_ch=10, use_tau=True, num_res_blocks=1)

    f0 = torch.randn(2, 8, 16, 12)
    f1 = torch.randn(2, 8, 16, 12)
    fe = torch.randn(2, 6, 16, 12)
    tau = torch.tensor([0.25, 0.75])

    y = m(f0, f1, fe, tau=tau)
    assert y.shape == (2, 10, 16, 12)

    with pytest.raises(ValueError, match="tau must be provided"):
        m(f0, f1, fe, tau=None)

    with pytest.raises(ValueError, match="tau must have shape"):
        m(f0, f1, fe, tau=torch.randn(2, 2, 1))


def test_cross_modal_fusion_block_mismatch_raises():
    m = modal_fusion.CrossModalFusionBlock(rgb_ch=8, evt_ch=6, out_ch=10, use_tau=False, num_res_blocks=1)
    f0 = torch.randn(2, 8, 16, 12)
    f1 = torch.randn(3, 8, 16, 12)
    fe = torch.randn(2, 6, 16, 12)

    with pytest.raises(ValueError, match="Batch sizes"):
        m(f0, f1, fe)


def test_pyramid_cross_modal_fusion_forward_and_missing_config():
    with pytest.raises(ValueError, match="Missing channel config"):
        modal_fusion.PyramidCrossModalFusion(
            rgb_channels={"s1": 8},
            evt_channels={"s1": 8},
            out_channels={"s1": 8},
        )

    rgb_ch = {"s1": 8, "s2": 16, "s4": 32, "s8": 64}
    evt_ch = {"s1": 6, "s2": 12, "s4": 24, "s8": 48}
    out_ch = {"s1": 10, "s2": 20, "s4": 40, "s8": 80}

    m = modal_fusion.PyramidCrossModalFusion(rgb_ch, evt_ch, out_ch, use_tau=True, num_res_blocks=1)

    feat0 = {
        "s1": torch.randn(2, 8, 32, 24),
        "s2": torch.randn(2, 16, 16, 12),
        "s4": torch.randn(2, 32, 8, 6),
        "s8": torch.randn(2, 64, 4, 3),
    }
    feat1 = {
        "s1": torch.randn(2, 8, 32, 24),
        "s2": torch.randn(2, 16, 16, 12),
        "s4": torch.randn(2, 32, 8, 6),
        "s8": torch.randn(2, 64, 4, 3),
    }
    evt_feat = {
        "s1": torch.randn(2, 6, 32, 24),
        "s2": torch.randn(2, 12, 16, 12),
        "s4": torch.randn(2, 24, 8, 6),
        "s8": torch.randn(2, 48, 4, 3),
    }

    out = m(feat0, feat1, evt_feat, tau=torch.tensor([0.2, 0.8]))
    assert out["s1"].shape == (2, 10, 32, 24)
    assert out["s8"].shape == (2, 80, 4, 3)


def test_motion_head_level_use_prev_false_and_true():
    lvl_coarse = modal_fusion.MotionHeadLevel(fused_ch=16, hidden_ch=10, num_res_blocks=1, use_prev=False)
    fused_s8 = torch.randn(2, 16, 4, 3)
    hidden8, pred8 = lvl_coarse(fused_s8)

    assert hidden8.shape == (2, 10, 4, 3)
    assert pred8["flow_t0"].shape == (2, 2, 4, 3)

    lvl_fine = modal_fusion.MotionHeadLevel(fused_ch=8, hidden_ch=10, num_res_blocks=1, use_prev=True)
    fused_s4 = torch.randn(2, 8, 8, 6)

    hidden4, pred4 = lvl_fine(fused_s4, prev_hidden=hidden8, prev_pred=pred8)
    assert hidden4.shape == (2, 10, 8, 6)
    assert pred4["mask_logits"].shape == (2, 2, 8, 6)

    with pytest.raises(ValueError, match="prev_hidden and prev_pred"):
        lvl_fine(fused_s4)


def test_motion_head_forward_outputs_and_mask_sum():
    m = modal_fusion.MotionHead(
        fused_channels={"s1": 8, "s2": 12, "s4": 16, "s8": 20},
        hidden_channels={"s1": 10, "s2": 10, "s4": 10, "s8": 10},
        num_res_blocks_per_level=1,
    )

    fused = {
        "s1": torch.randn(2, 8, 32, 24),
        "s2": torch.randn(2, 12, 16, 12),
        "s4": torch.randn(2, 16, 8, 6),
        "s8": torch.randn(2, 20, 4, 3),
    }
    out = m(fused)

    assert out["flow_t0"].shape == (2, 2, 32, 24)
    assert out["mask0"].shape == (2, 1, 32, 24)
    assert out["conf1"].shape == (2, 1, 32, 24)

    mask_sum = out["mask0"] + out["mask1"]
    assert torch.allclose(mask_sum.mean(), torch.tensor(1.0), atol=1e-5)


# ------------------------------
# decoder.py
# ------------------------------

def test_make_base_grid_and_pixel_grid_to_normalized():
    grid = decoder.make_base_grid(2, 3, 4, device=torch.device("cpu"), dtype=torch.float32)
    assert grid.shape == (2, 3, 4, 2)
    assert torch.equal(grid[0, 0, 0], torch.tensor([0.0, 0.0]))
    assert torch.equal(grid[0, 2, 3], torch.tensor([3.0, 2.0]))

    norm = decoder.pixel_grid_to_normalized(grid, 3, 4)
    assert norm.shape == (2, 3, 4, 2)
    assert torch.allclose(norm[0, 0, 0], torch.tensor([-1.0, -1.0]))
    assert torch.allclose(norm[0, 2, 3], torch.tensor([1.0, 1.0]))


def test_backward_warp_identity_and_errors():
    src = torch.randn(2, 3, 8, 6)
    flow = torch.zeros(2, 2, 8, 6)
    warped = decoder.backward_warp(src, flow)
    assert warped.shape == src.shape
    assert torch.allclose(warped, src, atol=1e-5)

    with pytest.raises(ValueError, match="both be 4D"):
        decoder.backward_warp(torch.randn(3, 8, 6), flow)

    with pytest.raises(ValueError, match="must have 2 channels"):
        decoder.backward_warp(src, torch.randn(2, 3, 8, 6))


def test_warp_valid_mask_bounds():
    flow = torch.zeros(1, 2, 4, 4)
    valid = decoder.warp_valid_mask(flow)
    assert valid.shape == (1, 1, 4, 4)
    assert torch.all(valid == 1)

    flow[:, 0] = 100.0
    valid_out = decoder.warp_valid_mask(flow)
    assert torch.all(valid_out == 0)


def test_refinement_decoder_output_shape():
    m = decoder.RefinementDecoder(in_ch=9, hidden_ch=16, num_res_blocks=2)
    x = torch.randn(2, 9, 16, 12)
    y = m(x)
    assert y.shape == (2, 3, 16, 12)


def test_warping_synthesis_decoder_forward_and_optional_features():
    b, h, w = 2, 16, 12
    i0 = torch.rand(b, 3, h, w)
    i1 = torch.rand(b, 3, h, w)

    motion = {
        "flow_t0": torch.zeros(b, 2, h, w),
        "flow_t1": torch.zeros(b, 2, h, w),
        "mask0": torch.full((b, 1, h, w), 0.5),
        "mask1": torch.full((b, 1, h, w), 0.5),
        "conf0": torch.ones(b, 1, h, w),
        "conf1": torch.ones(b, 1, h, w),
    }

    m = decoder.WarpingSynthesisDecoder(
        fused_ch=4,
        rgb_feat_ch=6,
        hidden_ch=16,
        num_res_blocks=1,
        use_warped_rgb_features=True,
        clamp_output=True,
    )

    out = m(
        i0,
        i1,
        motion,
        fused_s1=torch.randn(b, 4, h, w),
        rgb_feat0_s1=torch.randn(b, 6, h, w),
        rgb_feat1_s1=torch.randn(b, 6, h, w),
    )

    assert out["pred"].shape == (b, 3, h, w)
    assert torch.all(out["pred"] >= 0.0)
    assert torch.all(out["pred"] <= 1.0)


def test_warping_synthesis_decoder_requires_rgb_feat_ch_and_inputs():
    with pytest.raises(ValueError, match="rgb_feat_ch must be > 0"):
        decoder.WarpingSynthesisDecoder(use_warped_rgb_features=True)

    b, h, w = 1, 8, 8
    i0 = torch.rand(b, 3, h, w)
    i1 = torch.rand(b, 3, h, w)
    motion = {
        "flow_t0": torch.zeros(b, 2, h, w),
        "flow_t1": torch.zeros(b, 2, h, w),
        "mask0": torch.full((b, 1, h, w), 0.5),
        "mask1": torch.full((b, 1, h, w), 0.5),
        "conf0": torch.ones(b, 1, h, w),
        "conf1": torch.ones(b, 1, h, w),
    }

    m = decoder.WarpingSynthesisDecoder(rgb_feat_ch=6, use_warped_rgb_features=True, hidden_ch=8, num_res_blocks=1)
    with pytest.raises(ValueError, match="Need rgb_feat0_s1 and rgb_feat1_s1"):
        m(i0, i1, motion)
