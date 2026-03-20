"""
Dataset for cycle-consistent event-guided VFI using left-edged event windows.

Event windows are left-aligned, width = event_window_us.

For each frame triplet at times (t0, t1, t2):

    t0.5  = (t0 + t1) / 2       <- midpoint between I0 and I1
    t1.5  = (t1 + t2) / 2       <- midpoint between I1 and I2

Prediction stages use midpoint events; reconstruction stage uses integer-frame events:

    E01   = events in [t0.5, t0.5 + W]  Stage A: predict I0.5 from (I0,  I1)
    E12   = events in [t1.5, t1.5 + W]  Stage B: predict I1.5 from (I1,  I2)
    E0515 = events in [t1,   t1   + W]  Stage C: reconstruct I1 from (I0.5, I1.5)

Output keys match EventVFIModel.forward() exactly:
    I0, I1, I2, E01, E12, E0515, meta
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.data_preparation import (
    AugmentConfig,
    EventStream,
    FrameRecord,
    SampleIndices,
    SpatialConfig,
    _crop_tensor,
    _events_window_to_voxel,
    _read_frame,
    _slice_event_indices,
    load_event_stream,
    load_frame_records,
)


# ---------------------------------------------------------------------------
# Left-edged voxel helper
# ---------------------------------------------------------------------------

def _build_left_voxel(
    events: EventStream,
    start_us: float,
    window_us: float,
    num_bins: int,
    sensor_h: int,
    sensor_w: int,
) -> torch.Tensor:
    """Build a voxel grid from events in [start_us, start_us + window_us]."""
    return _events_window_to_voxel(events, start_us, start_us + window_us, num_bins, sensor_h, sensor_w)


def _count_in_window(events: EventStream, start_us: float, window_us: float) -> int:
    lo, hi = _slice_event_indices(events.t_us, start_us, start_us + window_us)
    return max(0, hi - lo)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CenteredWindowVFIDataset(Dataset):
    """Frame triplets with tight event windows centered at each target time.

    Compared to SelfSupervisedEventVFIDataset (which uses all events in the
    full inter-frame interval), this dataset uses only events within ±half_window
    of the interpolation midpoint.  This isolates the motion signal at the
    exact time of interest rather than integrating over the whole interval.

    Parameters
    ----------
    sequence_name        : identifier stored in meta
    frame_dir            : directory containing RGB frames
    frame_timestamps_csv : CSV with frame indices, timestamps, filenames
    events_path          : .raw / .csv / .npy / .npz event file
    num_event_bins       : temporal bins B for voxel representation
    event_window_us      : total event window width in µs (default 1000)
                           half_window = event_window_us / 2
    frame_time_unit      : unit of timestamps in the frame CSV ("s", "ms", "us")
    event_time_unit      : unit of timestamps in the event file ("us", "ms", "s")
    sample_step          : stride when building (i0, i1, i2) triplets
    spatial              : optional resize / crop config
    augment              : optional horizontal-flip augmentation
    near_empty_threshold : minimum events per window to flag as sparse
    strict_validation    : require event stream to fully cover frame timeline
    rng_seed             : random seed for augmentation
    """

    def __init__(
        self,
        sequence_name: str,
        frame_dir: Path | str,
        frame_timestamps_csv: Path | str,
        events_path: Path | str,
        num_event_bins: int = 9,
        event_window_us: float = 1000.0,
        frame_time_unit: str = "s",
        event_time_unit: str = "us",
        sample_step: int = 1,
        spatial: Optional[SpatialConfig] = None,
        augment: Optional[AugmentConfig] = None,
        near_empty_threshold: int = 1,
        strict_validation: bool = True,
        rng_seed: int = 0,
    ) -> None:
        super().__init__()
        self.sequence_name     = sequence_name
        self.frame_dir         = Path(frame_dir)
        self.events_path       = Path(events_path)
        self.num_event_bins    = int(num_event_bins)
        self.event_window_us   = float(event_window_us)
        self.sample_step       = int(sample_step)
        self.spatial           = spatial or SpatialConfig()
        self.augment           = augment or AugmentConfig()
        self.near_empty_threshold = int(near_empty_threshold)
        self.rng               = np.random.default_rng(rng_seed)

        if self.num_event_bins <= 0:
            raise ValueError("num_event_bins must be > 0")
        if self.event_window_us <= 0:
            raise ValueError("event_window_us must be > 0")
        if self.sample_step <= 0:
            raise ValueError("sample_step must be > 0")

        self.frames = load_frame_records(
            frame_dir=self.frame_dir,
            timestamps_csv=Path(frame_timestamps_csv),
            frame_time_unit=frame_time_unit,
        )
        self.events = load_event_stream(self.events_path, event_time_unit=event_time_unit)

        if self.events.t_us.size > 0 and strict_validation:
            f_start = self.frames[0].timestamp_us
            f_end   = self.frames[-1].timestamp_us + self.event_window_us
            e_start = float(self.events.t_us[0])
            e_end   = float(self.events.t_us[-1])
            if not (e_start <= f_start and e_end >= f_end):
                raise ValueError(
                    "Event stream does not fully cover frame timeline. "
                    "Use synced data or set strict_validation=False."
                )

        self.samples: List[SampleIndices] = [
            SampleIndices(i0=i, i1=i + 1, i2=i + 2)
            for i in range(0, len(self.frames) - 2, self.sample_step)
        ]
        if not self.samples:
            raise ValueError("No valid frame triplets found")

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _choose_crop(self, h: int, w: int) -> Tuple[int, int, int, int]:
        if self.spatial.crop_hw is None:
            return 0, 0, h, w
        crop_h, crop_w = self.spatial.crop_hw
        if crop_h > h or crop_w > w:
            raise ValueError(f"Crop {(crop_h, crop_w)} exceeds input {(h, w)}")
        if self.spatial.random_crop:
            top  = int(self.rng.integers(0, h - crop_h + 1))
            left = int(self.rng.integers(0, w - crop_w + 1))
        else:
            top  = (h - crop_h) // 2
            left = (w - crop_w) // 2
        return top, left, crop_h, crop_w

    def _apply_spatial(
        self,
        frames: Dict[str, torch.Tensor],
        voxels: Dict[str, torch.Tensor],
        do_hflip: bool,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # Resize
        if self.spatial.resize_hw is not None:
            rh, rw = self.spatial.resize_hw
            for k in list(frames.keys()):
                frames[k] = F.interpolate(
                    frames[k].unsqueeze(0), size=(rh, rw),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)
            for k in list(voxels.keys()):
                voxels[k] = F.interpolate(
                    voxels[k].unsqueeze(0), size=(rh, rw),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)

        # Crop
        ref = next(iter(frames.values())) if frames else next(iter(voxels.values()))
        _, h, w = ref.shape
        top, left, ch, cw = self._choose_crop(h, w)
        for k in list(frames.keys()):
            frames[k] = _crop_tensor(frames[k], top, left, ch, cw)
        for k in list(voxels.keys()):
            voxels[k] = _crop_tensor(voxels[k], top, left, ch, cw)

        # Horizontal flip
        if do_hflip:
            for k in list(frames.keys()):
                frames[k] = torch.flip(frames[k], dims=[2])
            for k in list(voxels.keys()):
                voxels[k] = torch.flip(voxels[k], dims=[2])

        return frames, voxels

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s  = self.samples[idx]
        t0 = float(self.frames[s.i0].timestamp_us)
        t1 = float(self.frames[s.i1].timestamp_us)
        t2 = float(self.frames[s.i2].timestamp_us)

        if not (t0 < t1 < t2):
            raise RuntimeError("Triplet timestamps not strictly monotonic")

        # Interpolation target times
        t05 = 0.5 * (t0 + t1)   # midpoint between I0 and I1  (event anchor for prediction stage A)
        t15 = 0.5 * (t1 + t2)   # midpoint between I1 and I2  (event anchor for prediction stage B)
        # t1 is the integer-frame anchor for the reconstruction stage C

        # ── Load frames ───────────────────────────────────────────────
        I0 = _read_frame(self.frames[s.i0].path)
        I1 = _read_frame(self.frames[s.i1].path)
        I2 = _read_frame(self.frames[s.i2].path)
        _, fh, fw = I0.shape
        sensor_h, sensor_w = fh, fw

        # ── Event voxels (left-edged windows, width = event_window_us) ─
        # E01   : events at midpoint t05  -- guide prediction of I0.5 from (I0, I1)
        # E12   : events at midpoint t15  -- guide prediction of I1.5 from (I1, I2)
        # E0515 : events at integer t1    -- guide reconstruction of I1 from (I0.5, I1.5)
        w = self.event_window_us
        E01   = _build_left_voxel(self.events, t05, w, self.num_event_bins, sensor_h, sensor_w)
        E12   = _build_left_voxel(self.events, t15, w, self.num_event_bins, sensor_h, sensor_w)
        E0515 = _build_left_voxel(self.events, t1,  w, self.num_event_bins, sensor_h, sensor_w)

        # ── Augmentation ──────────────────────────────────────────────
        do_hflip = (
            self.augment.horizontal_flip_prob > 0
            and float(self.rng.random()) < self.augment.horizontal_flip_prob
        )

        frames = {"I0": I0, "I1": I1, "I2": I2}
        voxels = {"E01": E01, "E12": E12, "E0515": E0515}
        frames, voxels = self._apply_spatial(frames, voxels, do_hflip)

        # ── Event density metadata ────────────────────────────────────
        count_E01   = _count_in_window(self.events, t05, w)
        count_E12   = _count_in_window(self.events, t15, w)
        count_E0515 = _count_in_window(self.events, t1,  w)
        thr = self.near_empty_threshold

        return {
            **frames,
            **voxels,
            "meta": {
                "sequence_name":   self.sequence_name,
                "sample_index":    idx,
                "frame_indices":   (self.frames[s.i0].index, self.frames[s.i1].index, self.frames[s.i2].index),
                "timestamps":      {"t0": t0, "t05": t05, "t1": t1, "t2": t2},
                "event_window_us": w,
                "hflip":           do_hflip,
                "event_counts":    {"E01": count_E01, "E12": count_E12, "E0515": count_E0515},
                "near_empty":      {
                    "E01":   count_E01   < thr,
                    "E12":   count_E12   < thr,
                    "E0515": count_E0515 < thr,
                },
            },
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_centered_window_dataset(
    sequence_name: str,
    frame_dir: Path | str,
    frame_timestamps_csv: Path | str,
    events_path: Path | str,
    num_event_bins: int = 9,
    event_window_us: float = 1000.0,
    frame_time_unit: str = "s",
    event_time_unit: str = "us",
    sample_step: int = 1,
    spatial: Optional[SpatialConfig] = None,
    augment: Optional[AugmentConfig] = None,
    near_empty_threshold: int = 1,
    strict_validation: bool = True,
    rng_seed: int = 0,
) -> CenteredWindowVFIDataset:
    return CenteredWindowVFIDataset(
        sequence_name=sequence_name,
        frame_dir=frame_dir,
        frame_timestamps_csv=frame_timestamps_csv,
        events_path=events_path,
        num_event_bins=num_event_bins,
        event_window_us=event_window_us,
        frame_time_unit=frame_time_unit,
        event_time_unit=event_time_unit,
        sample_step=sample_step,
        spatial=spatial,
        augment=augment,
        near_empty_threshold=near_empty_threshold,
        strict_validation=strict_validation,
        rng_seed=rng_seed,
    )