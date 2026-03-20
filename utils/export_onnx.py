"""Export EventVFIModel to ONNX for Netron visualization.

Usage:
    python utils/export_onnx.py
    python utils/export_onnx.py --section event_vfi_model_smoke --out logs/model.onnx
    python utils/export_onnx.py --h 128 --w 128

Then open the .onnx file in:
  - https://netron.app  (browser, no install)
  - Netron desktop app  (pip install netron; python -m netron logs/model.onnx)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.model import EventVFIModel


class _ExportWrapper(torch.nn.Module):
    """Thin wrapper: returns tuple instead of dict (ONNX requirement)."""
    def __init__(self, model: EventVFIModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        I0: torch.Tensor,
        I1: torch.Tensor,
        I2: torch.Tensor,
        E01: torch.Tensor,
        E12: torch.Tensor,
        E0515: torch.Tensor,
    ):
        out = self.model(I0, I1, I2, E01, E12, E0515, return_debug=False)
        return out["pred_05"], out["pred_15"], out["pred_1_cyc"]


def export(
    section: str = "event_vfi_model_smoke",
    out_path: str = "logs/model.onnx",
    B: int = 1,
    H: int = 64,
    W: int = 64,
    opset: int = 17,
) -> None:
    device = torch.device("cpu")  # CPU for portable ONNX

    print(f"Loading model: [{section}]")
    model = EventVFIModel.from_config(section).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    bins = model.cfg.num_event_bins
    dummy = (
        torch.zeros(B, 3,    H, W, device=device),  # I0
        torch.zeros(B, 3,    H, W, device=device),  # I1
        torch.zeros(B, 3,    H, W, device=device),  # I2
        torch.zeros(B, bins, H, W, device=device),  # E01
        torch.zeros(B, bins, H, W, device=device),  # E12
        torch.zeros(B, bins, H, W, device=device),  # E0515
    )

    wrapper = _ExportWrapper(model)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to: {out}  (opset {opset})")
    torch.onnx.export(
        wrapper,
        dummy,
        str(out),
        input_names=["I0", "I1", "I2", "E01", "E12", "E0515"],
        output_names=["pred_05", "pred_15", "pred_1_cyc"],
        opset_version=opset,
        do_constant_folding=False,  # keep graph readable, don't fold constants
    )
    size_mb = out.stat().st_size / 1e6
    print(f"Done. File size: {size_mb:.1f} MB")
    print()
    print("View in browser : https://netron.app  (drag and drop the .onnx file)")
    print("View locally    : pip install netron")
    print(f"                  python -m netron {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export EventVFIModel to ONNX")
    p.add_argument("--section", default="event_vfi_model_smoke", help="config.json section")
    p.add_argument("--out",     default="logs/model.onnx",        help="output .onnx path")
    p.add_argument("--h",       type=int, default=64,              help="dummy input height")
    p.add_argument("--w",       type=int, default=64,              help="dummy input width")
    p.add_argument("--opset",   type=int, default=17,              help="ONNX opset version")
    args = p.parse_args()

    export(
        section=args.section,
        out_path=args.out,
        H=args.h,
        W=args.w,
        opset=args.opset,
    )