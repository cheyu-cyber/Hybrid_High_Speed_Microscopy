"""
Standalone end-to-end smoke test for EventVFIModel.

Run from the project root:
    E:/Users/cheyu/.venv/Scripts/python.exe tests/smoke_test_end_to_end.py

Checks (on a tiny synthetic batch, CPU only):
  1. All output shapes are correct
  2. All outputs are finite
  3. Warped frames have correct shape and are finite
  4. Backward pass produces finite gradients on every parameter
  5. Loss decreases across repeated Adam steps on the same batch
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Allow running from project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.model import EventVFIConfig, EventVFIModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "[PASS]"
FAIL = "[FAIL]"


def check(name: str, cond: bool, detail: str = "") -> bool:
    status = PASS if cond else FAIL
    suffix = f" -- {detail}" if detail else ""
    print(f"  {status} {name}{suffix}")
    return cond


def assert_finite(name: str, t: torch.Tensor) -> bool:
    ok = bool(torch.isfinite(t).all())
    return check(f"{name} is finite", ok, f"shape={tuple(t.shape)}")


# ---------------------------------------------------------------------------
# Config: small enough to be fast on CPU
# ---------------------------------------------------------------------------

def make_smoke_cfg() -> EventVFIConfig:
    """Load from config.json section 'event_vfi_model_smoke' if available,
    otherwise fall back to hardcoded small defaults."""
    try:
        from utils.config import load_config
        raw = load_config("event_vfi_model_smoke")
        cfg = EventVFIConfig(
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
        print("  Config loaded from config.json [event_vfi_model_smoke]")
        return cfg
    except Exception as exc:
        print(f"  config.json not available ({exc}); using hardcoded smoke defaults")
        return EventVFIConfig(
            num_event_bins=5,
            rgb_in_ch=3,
            rgb_base_ch=8,
            event_base_ch=8,
            encoder_res_blocks=1,
            fusion_use_tau=False,
            fusion_res_blocks=1,
            motion_res_blocks_per_level=1,
            decoder_hidden_ch=24,
            decoder_res_blocks=2,
            use_warped_rgb_features=True,
            clamp_output=True,
        )


# ---------------------------------------------------------------------------
# Test sections
# ---------------------------------------------------------------------------

def test_shapes_and_finite(
    model: EventVFIModel,
    inputs: dict,
    B: int,
    C: int,
    H: int,
    W: int,
) -> bool:
    print("\n[1] Shape + finite checks")
    out = model(**inputs, return_debug=True)

    ok = True
    for key in ("pred_05", "pred_15", "pred_1_cyc"):
        t = out[key]
        ok &= check(f"{key} shape", t.shape == (B, C, H, W), f"got {tuple(t.shape)}")
        ok &= assert_finite(key, t)

    return ok


def test_warping(model: EventVFIModel, inputs: dict, B: int, C: int, H: int, W: int) -> bool:
    print("\n[2] Warping checks (stage_01.decoder)")
    out = model(**inputs, return_debug=True)

    dec = out["stage_01"]["decoder"]
    ok = True
    for key in ("warped0", "warped1"):
        t = dec[key]
        ok &= check(f"{key} shape", t.shape == (B, C, H, W), f"got {tuple(t.shape)}")
        ok &= assert_finite(key, t)

    # Blend should also be valid
    ok &= assert_finite("blend", dec["blend"])
    return ok


def test_backward(model: EventVFIModel, inputs: dict) -> bool:
    print("\n[3] Backward pass + gradient finite check")
    out = model(**inputs, return_debug=False)
    I0, I1, I2 = inputs["I0"], inputs["I1"], inputs["I2"]
    loss_dict = model.compute_cycle_loss(out, I0=I0, I1=I1, I2=I2)
    loss = loss_dict["loss_total"]

    ok = assert_finite("loss_total", loss.unsqueeze(0))
    loss.backward()

    bad_grads = []
    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            bad_grads.append(name)

    ok &= check("all gradients finite", len(bad_grads) == 0,
                f"bad params: {bad_grads[:3]}" if bad_grads else "")
    return ok


def test_loss_decreases(model: EventVFIModel, inputs: dict, steps: int = 8) -> bool:
    print(f"\n[4] Loss-decrease check ({steps} Adam steps on same batch)")
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []

    for step in range(steps):
        optim.zero_grad(set_to_none=True)
        out = model(**inputs, return_debug=False)
        I0, I1, I2 = inputs["I0"], inputs["I1"], inputs["I2"]
        loss = model.compute_cycle_loss(out, I0=I0, I1=I1, I2=I2)["loss_total"]
        loss.backward()
        optim.step()
        val = float(loss.detach())
        losses.append(val)
        print(f"    step {step+1:2d}  loss={val:.6f}")

    decreased = min(losses) < losses[0]
    check("loss decreased over steps", decreased,
          f"first={losses[0]:.6f}  best={min(losses):.6f}")
    return decreased


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cpu")

    # ── Spatial dims must be divisible by 8 (4 pyramid levels) ──
    B, C, H, W = 2, 3, 32, 32

    print("=" * 60)
    print("EventVFI end-to-end smoke test")
    print("=" * 60)

    # Build config & model
    print("\nBuilding model...")
    cfg = make_smoke_cfg()
    num_bins = cfg.num_event_bins
    model = EventVFIModel(cfg).to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Synthetic batch
    I0     = torch.rand(B, C, H, W, device=device)
    I1     = torch.rand(B, C, H, W, device=device)
    I2     = torch.rand(B, C, H, W, device=device)
    E01    = torch.randn(B, num_bins, H, W, device=device) * 0.05
    E12    = torch.randn(B, num_bins, H, W, device=device) * 0.05
    E0515  = torch.randn(B, num_bins, H, W, device=device) * 0.05

    inputs = dict(I0=I0, I1=I1, I2=I2, E01=E01, E12=E12, E0515=E0515)

    results = []
    try:
        results.append(test_shapes_and_finite(model, inputs, B, C, H, W))
    except Exception:
        print(f"  {FAIL} Exception in shape test:")
        traceback.print_exc()
        results.append(False)

    try:
        results.append(test_warping(model, inputs, B, C, H, W))
    except Exception:
        print(f"  {FAIL} Exception in warping test:")
        traceback.print_exc()
        results.append(False)

    try:
        results.append(test_backward(model, inputs))
    except Exception:
        print(f"  {FAIL} Exception in backward test:")
        traceback.print_exc()
        results.append(False)

    try:
        results.append(test_loss_decreases(model, inputs))
    except Exception:
        print(f"  {FAIL} Exception in loss-decrease test:")
        traceback.print_exc()
        results.append(False)

    # ── Summary ──
    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"Result: {passed}/{total} test sections passed")
    print("=" * 60)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
