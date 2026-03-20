from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch


def _to_numpy_chw(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float().clamp(0.0, 1.0)
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise ValueError("Expected CHW tensor")
    return x.numpy()


def _normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max - x_min < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def _save_image_tensor(path: Path, x: torch.Tensor) -> None:
    chw = _to_numpy_chw(x)
    c, h, w = chw.shape

    if c == 1:
        img = np.repeat((chw[0] * 255.0).astype(np.uint8)[..., None], 3, axis=2)
    elif c >= 3:
        rgb = np.transpose(chw[:3], (1, 2, 0))
        img = (rgb * 255.0).astype(np.uint8)
    else:
        vis = _normalize_map(np.abs(chw).mean(axis=0))
        img = np.repeat((vis * 255.0).astype(np.uint8)[..., None], 3, axis=2)

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def _save_flow(path: Path, flow: torch.Tensor) -> None:
    if flow.ndim == 4:
        flow = flow[0]
    if flow.shape[0] != 2:
        raise ValueError("Flow must have shape (2,H,W)")

    fx = flow[0].detach().cpu().numpy()
    fy = flow[1].detach().cpu().numpy()

    mag = np.sqrt(fx * fx + fy * fy)
    ang = np.arctan2(fy, fx)
    ang = (ang + np.pi) / (2.0 * np.pi)

    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 179.0).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (_normalize_map(mag) * 255.0).astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(str(path), bgr)


def _save_event_voxel(path: Path, voxel: torch.Tensor) -> None:
    if voxel.ndim == 4:
        voxel = voxel[0]
    if voxel.ndim != 3:
        raise ValueError("Event voxel must have shape (B,C,H,W) or (C,H,W)")

    pos = voxel.clamp(min=0.0).sum(dim=0).detach().cpu().numpy()
    neg = (-voxel.clamp(max=0.0)).sum(dim=0).detach().cpu().numpy()

    pos = _normalize_map(pos)
    neg = _normalize_map(neg)

    rgb = np.stack([pos, neg, np.zeros_like(pos)], axis=2)
    img = (rgb * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def _save_if_tensor(path: Path, value: Any) -> None:
    if not isinstance(value, torch.Tensor):
        return
    if value.ndim == 4 and value.shape[1] == 2:
        _save_flow(path, value)
    elif value.ndim in (3, 4):
        _save_image_tensor(path, value)


def _slice_sample(obj: Any, b: int) -> Any:
    """Recursively slice a single batch item (index b) from dicts/tensors."""
    if isinstance(obj, torch.Tensor):
        return obj[b : b + 1]
    if isinstance(obj, dict):
        return {k: _slice_sample(v, b) for k, v in obj.items()}
    return obj


def _save_decoder_outputs(stage_dir: Path, decoder_dict: Dict[str, torch.Tensor]) -> None:
    for key, value in decoder_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        file_path = stage_dir / f"decoder_{key}.png"
        _save_if_tensor(file_path, value)


def _save_motion_outputs(stage_dir: Path, motion_dict: Dict[str, torch.Tensor]) -> None:
    for key, value in motion_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        file_path = stage_dir / f"motion_{key}.png"
        _save_if_tensor(file_path, value)


def _save_one_sample(root: Path, sample: Dict[str, torch.Tensor], outputs: Dict[str, Any]) -> None:
    """Save all visualizations for a single batch item (already sliced to B=1)."""
    root.mkdir(parents=True, exist_ok=True)

    for key in ("I0", "I1", "I2"):
        if key in sample:
            _save_image_tensor(root / f"input_{key}.png", sample[key])
    for key in ("E01", "E12", "E0515"):
        if key in sample:
            _save_event_voxel(root / f"input_{key}.png", sample[key])

    for key in ("pred_05", "pred_15", "pred_1_cyc"):
        if key in outputs and isinstance(outputs[key], torch.Tensor):
            _save_image_tensor(root / f"{key}.png", outputs[key])

    for stage_key in ("stage_01", "stage_12", "stage_cyc"):
        if stage_key not in outputs:
            continue
        stage = outputs[stage_key]
        stage_dir = root / stage_key
        stage_dir.mkdir(parents=True, exist_ok=True)

        pred = stage.get("pred")
        if isinstance(pred, torch.Tensor):
            _save_image_tensor(stage_dir / "pred.png", pred)
        decoder = stage.get("decoder")
        if isinstance(decoder, dict):
            _save_decoder_outputs(stage_dir, decoder)
        motion = stage.get("motion")
        if isinstance(motion, dict):
            _save_motion_outputs(stage_dir, motion)


def save_cycle_debug_outputs(
    out_dir: Path,
    step: int,
    sample: Dict[str, torch.Tensor],
    outputs: Dict[str, Any],
    max_samples: int = 4,
) -> None:
    """Dump major intermediates for one optimization step.

    Saved content per sample:
    - Inputs: I0, I1, I2 (RGB), E01, E12, E0515 (event voxels)
    - Stage predictions: pred_05, pred_15, pred_1_cyc
    - Per-stage decoder outputs and motion outputs

    When max_samples > 1, saves each batch item to sample_00/, sample_01/, ...
    """
    root = out_dir / f"step_{step:07d}"
    B = next(iter(sample.values())).shape[0]
    n = min(B, max_samples)

    for b in range(n):
        sample_dir = root / f"sample_{b:02d}" if n > 1 else root
        _save_one_sample(sample_dir, _slice_sample(sample, b), _slice_sample(outputs, b))
