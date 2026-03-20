from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


CV2: Any = cv2


@dataclass(frozen=True)
class FrameRecord:
	"""One RGB frame and its timestamp (in microseconds)."""

	index: int
	timestamp_us: float
	path: Path


@dataclass(frozen=True)
class EventStream:
	"""In-memory event stream using x, y, t, p arrays."""

	x: np.ndarray
	y: np.ndarray
	t_us: np.ndarray
	p: np.ndarray
	sensor_size: Tuple[int, int]


@dataclass(frozen=True)
class SampleIndices:
	"""Indices of one triplet in the frame table."""

	i0: int
	i1: int
	i2: int


@dataclass
class SpatialConfig:
	"""Image and event tensor preprocessing options."""

	resize_hw: Optional[Tuple[int, int]] = None
	crop_hw: Optional[Tuple[int, int]] = None
	random_crop: bool = False


@dataclass
class AugmentConfig:
	"""Consistency-preserving augmentation options."""

	horizontal_flip_prob: float = 0.0
	temporal_reverse_prob: float = 0.0

# ---------------------------------------------------------------------------
# 1. Event representation — voxel grid  (torch)
# ---------------------------------------------------------------------------

def events_to_voxel_grid(events_t, events_x, events_y, events_p,
                         num_bins: int, height: int, width: int):
    """Convert a batch of events into a voxel grid tensor.

    Parameters
    ----------
    events_t : (N,) tensor of normalised timestamps in [0, 1]
    events_x : (N,) int tensor of x coordinates
    events_y : (N,) int tensor of y coordinates
    events_p : (N,) tensor of polarities (+1 / -1)
    num_bins : number of temporal bins B
    height, width : sensor resolution

    Returns
    -------
    voxel : (B, H, W) float tensor
    """

    voxel = torch.zeros(num_bins, height, width, dtype=torch.float32,
                        device=events_t.device)
    if events_t.numel() == 0:
        return voxel

    # Distribute each event across two adjacent bins via linear interpolation
    t_scaled = events_t * (num_bins - 1)
    t_low = t_scaled.floor().long().clamp(0, num_bins - 1)
    t_high = (t_low + 1).clamp(0, num_bins - 1)
    w_high = t_scaled - t_low.float()
    w_low = 1.0 - w_high

    idx_y = events_y.long()
    idx_x = events_x.long()

    voxel.index_put_((t_low, idx_y, idx_x), events_p * w_low, accumulate=True)
    voxel.index_put_((t_high, idx_y, idx_x), events_p * w_high, accumulate=True)
    return voxel


def _to_microseconds(value: float, unit: str) -> float:
	if unit == "us":
		return float(value)
	if unit == "ms":
		return float(value) * 1_000.0
	if unit == "s":
		return float(value) * 1_000_000.0
	raise ValueError(f"Unsupported time unit: {unit}")


def _find_first_existing_key(row: Dict[str, str], keys: Sequence[str]) -> Optional[str]:
	for key in keys:
		if key in row and row[key] != "":
			return key
	return None


def load_frame_records(
	frame_dir: Path,
	timestamps_csv: Path,
	frame_time_unit: str = "s",
) -> List[FrameRecord]:
	"""Load frame paths + timestamps from CSV.

	Supported CSV layouts include either:
	- frame_idx, timestamp_s, filename
	- frame_idx, t_start_us, t_end_us, filename
	- frame_idx, timestamp_us, filename
	"""
	records: List[FrameRecord] = []

	with timestamps_csv.open("r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row_idx, row in enumerate(reader):
			idx_key = _find_first_existing_key(row, ["frame_idx", "idx", "index"]) or "frame_idx"
			fname_key = _find_first_existing_key(row, ["filename", "file", "path"]) or "filename"

			t_key = _find_first_existing_key(row, ["timestamp_us", "timestamp_s", "timestamp_ms"])
			ts_us: Optional[float] = None

			if t_key is not None:
				raw_t = float(row[t_key])
				if t_key.endswith("_us"):
					ts_us = _to_microseconds(raw_t, "us")
				elif t_key.endswith("_ms"):
					ts_us = _to_microseconds(raw_t, "ms")
				else:
					ts_us = _to_microseconds(raw_t, "s")
			else:
				start_key = _find_first_existing_key(row, ["t_start_us", "start_us"])
				end_key = _find_first_existing_key(row, ["t_end_us", "end_us"])
				if start_key is None or end_key is None:
					raise ValueError(
						"Frame timestamp CSV must contain timestamp_* or (t_start_us, t_end_us)."
					)
				t0 = float(row[start_key])
				t1 = float(row[end_key])
				ts_us = 0.5 * (t0 + t1)

			frame_path = frame_dir / row[fname_key]
			if not frame_path.exists():
				raise FileNotFoundError(f"Frame not found: {frame_path}")

			try:
				frame_idx = int(row[idx_key])
			except Exception:
				frame_idx = row_idx

			records.append(FrameRecord(index=frame_idx, timestamp_us=ts_us, path=frame_path))

	if len(records) < 3:
		raise ValueError("Need at least 3 frames to build cycle-consistent triplets")

	ts = np.array([r.timestamp_us for r in records], dtype=np.float64)
	if not np.all(np.diff(ts) > 0):
		raise ValueError("Frame timestamps must be strictly monotonic increasing")

	return records


def _load_events_from_csv(path: Path, event_time_unit: str) -> EventStream:
	with path.open("r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		rows = list(reader)

	if not rows:
		raise ValueError(f"No events found in CSV: {path}")

	x_key = "x"
	y_key = "y"
	p_key = "p"
	t_key = _find_first_existing_key(rows[0], ["t", "t_us", "timestamp", "timestamp_us"])
	if t_key is None:
		raise ValueError("Event CSV must contain one of: t, t_us, timestamp, timestamp_us")

	x = np.asarray([int(r[x_key]) for r in rows], dtype=np.int64)
	y = np.asarray([int(r[y_key]) for r in rows], dtype=np.int64)
	p = np.asarray([float(r[p_key]) for r in rows], dtype=np.float32)
	t_raw = np.asarray([float(r[t_key]) for r in rows], dtype=np.float64)

	if t_key.endswith("_us"):
		t_us = t_raw
	else:
		t_us = np.asarray([_to_microseconds(v, event_time_unit) for v in t_raw], dtype=np.float64)

	p = np.where(p > 0, 1.0, -1.0).astype(np.float32)

	h = int(y.max()) + 1
	w = int(x.max()) + 1
	return EventStream(x=x, y=y, t_us=t_us, p=p, sensor_size=(h, w))


def _load_events_from_np(path: Path, event_time_unit: str) -> EventStream:
	data = np.load(path, allow_pickle=False)

	if isinstance(data, np.ndarray) and data.dtype.names is not None:
		# Structured array.
		x = data["x"].astype(np.int64)
		y = data["y"].astype(np.int64)
		p = data["p"].astype(np.float32)
		t_raw = data["t"].astype(np.float64)
	elif isinstance(data, np.ndarray):
		if data.ndim != 2 or data.shape[1] < 4:
			raise ValueError("Expected N x 4 ndarray for events: [x, y, t, p]")
		x = data[:, 0].astype(np.int64)
		y = data[:, 1].astype(np.int64)
		t_raw = data[:, 2].astype(np.float64)
		p = data[:, 3].astype(np.float32)
	else:
		# NPZ file.
		keys = set(data.keys())
		required = {"x", "y", "t", "p"}
		if not required.issubset(keys):
			raise ValueError(f"NPZ events must include keys {required}, got {keys}")
		x = np.asarray(data["x"], dtype=np.int64)
		y = np.asarray(data["y"], dtype=np.int64)
		t_raw = np.asarray(data["t"], dtype=np.float64)
		p = np.asarray(data["p"], dtype=np.float32)

	t_us = np.asarray([_to_microseconds(v, event_time_unit) for v in t_raw], dtype=np.float64)
	p = np.where(p > 0, 1.0, -1.0).astype(np.float32)

	h = int(y.max()) + 1 if y.size > 0 else 1
	w = int(x.max()) + 1 if x.size > 0 else 1
	return EventStream(x=x, y=y, t_us=t_us, p=p, sensor_size=(h, w))


def _load_events_from_raw(path: Path) -> EventStream:
	"""Load OpenEB .raw events into memory.

	This path is intended for smaller clips or pre-cut files used for training.
	"""
	try:
		from metavision_core.event_io import EventsIterator
	except Exception as exc:
		raise RuntimeError(
			"Reading .raw requires metavision_core.event_io.EventsIterator"
		) from exc

	iterator = EventsIterator(input_path=str(path), start_ts=0, delta_t=1_000)
	size = iterator.get_size()
	if size is None:
		raise RuntimeError("Could not infer sensor size from raw event file")
	h, w = size

	xs: List[np.ndarray] = []
	ys: List[np.ndarray] = []
	ts: List[np.ndarray] = []
	ps: List[np.ndarray] = []
	for ev in iterator:
		ev_arr = np.asarray(ev)
		if ev_arr.size == 0:
			continue
		xs.append(ev_arr["x"].astype(np.int64))
		ys.append(ev_arr["y"].astype(np.int64))
		ts.append(ev_arr["t"].astype(np.float64))
		p = np.where(ev_arr["p"].astype(np.int8) > 0, 1.0, -1.0).astype(np.float32)
		ps.append(p)

	if not xs:
		return EventStream(
			x=np.empty(0, dtype=np.int64),
			y=np.empty(0, dtype=np.int64),
			t_us=np.empty(0, dtype=np.float64),
			p=np.empty(0, dtype=np.float32),
			sensor_size=(h, w),
		)

	return EventStream(
		x=np.concatenate(xs),
		y=np.concatenate(ys),
		t_us=np.concatenate(ts),
		p=np.concatenate(ps),
		sensor_size=(h, w),
	)


def load_event_stream(events_path: Path, event_time_unit: str = "us") -> EventStream:
	"""Load events from CSV/NPY/NPZ/RAW into EventStream."""
	suffix = events_path.suffix.lower()
	if suffix == ".csv":
		stream = _load_events_from_csv(events_path, event_time_unit=event_time_unit)
	elif suffix in {".npy", ".npz"}:
		stream = _load_events_from_np(events_path, event_time_unit=event_time_unit)
	elif suffix == ".raw":
		stream = _load_events_from_raw(events_path)
	else:
		raise ValueError(f"Unsupported event file extension: {suffix}")

	if stream.t_us.size > 1 and not np.all(np.diff(stream.t_us) >= 0):
		order = np.argsort(stream.t_us, kind="mergesort")
		stream = EventStream(
			x=stream.x[order],
			y=stream.y[order],
			t_us=stream.t_us[order],
			p=stream.p[order],
			sensor_size=stream.sensor_size,
		)

	return stream


def _slice_event_indices(t_us: np.ndarray, start_us: float, end_us: float) -> Tuple[int, int]:
	"""Return [lo, hi) indices for no-leakage temporal slicing."""
	lo = int(np.searchsorted(t_us, start_us, side="left"))
	hi = int(np.searchsorted(t_us, end_us, side="left"))
	return lo, hi


def _normalize_times_for_window(event_t_us: np.ndarray, start_us: float, end_us: float) -> np.ndarray:
	if event_t_us.size == 0:
		return np.empty(0, dtype=np.float32)
	denom = max(end_us - start_us, 1e-6)
	t_norm = (event_t_us - start_us) / denom
	return np.clip(t_norm, 0.0, 1.0).astype(np.float32)


def _events_window_to_voxel(
	events: EventStream,
	start_us: float,
	end_us: float,
	num_bins: int,
	sensor_h: int,
	sensor_w: int,
) -> torch.Tensor:
	lo, hi = _slice_event_indices(events.t_us, start_us, end_us)
	x = events.x[lo:hi]
	y = events.y[lo:hi]
	t = events.t_us[lo:hi]
	p = events.p[lo:hi]

	if t.size > 0:
		if t.min() < start_us or t.max() >= end_us:
			raise RuntimeError("Temporal leakage detected while slicing event window")

	t_norm = _normalize_times_for_window(t, start_us, end_us)
	t_t = torch.from_numpy(t_norm)
	x_t = torch.from_numpy(x.astype(np.int64, copy=False))
	y_t = torch.from_numpy(y.astype(np.int64, copy=False))
	p_t = torch.from_numpy(p.astype(np.float32, copy=False))

	voxel = events_to_voxel_grid(
		events_t=t_t,
		events_x=x_t,
		events_y=y_t,
		events_p=p_t,
		num_bins=num_bins,
		height=sensor_h,
		width=sensor_w,
	)

	if voxel.shape != (num_bins, sensor_h, sensor_w):
		raise RuntimeError(
			f"Voxel shape mismatch: got {tuple(voxel.shape)} expected {(num_bins, sensor_h, sensor_w)}"
		)
	return voxel.float()


def _read_frame(path: Path, force_grayscale: bool = False) -> torch.Tensor:
	flag = CV2.IMREAD_GRAYSCALE if force_grayscale else CV2.IMREAD_COLOR
	img = CV2.imread(str(path), flag)
	if img is None:
		raise FileNotFoundError(f"Failed to read image: {path}")

	if force_grayscale:
		x = torch.from_numpy(img).unsqueeze(0).float() / 255.0
	else:
		rgb = CV2.cvtColor(img, CV2.COLOR_BGR2RGB)
		x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
	return x


def _crop_tensor(x: torch.Tensor, top: int, left: int, crop_h: int, crop_w: int) -> torch.Tensor:
	return x[:, top : top + crop_h, left : left + crop_w]


def _temporal_reverse_events(events: EventStream, start_us: float, end_us: float) -> EventStream:
	"""Reverse event time within one interval and invert polarity."""
	lo, hi = _slice_event_indices(events.t_us, start_us, end_us)
	x = events.x[lo:hi].copy()
	y = events.y[lo:hi].copy()
	t = (start_us + end_us) - events.t_us[lo:hi]
	p = -events.p[lo:hi]

	if t.size > 1:
		order = np.argsort(t, kind="mergesort")
		x = x[order]
		y = y[order]
		t = t[order]
		p = p[order]

	return EventStream(x=x, y=y, t_us=t, p=p, sensor_size=events.sensor_size)


class SelfSupervisedEventVFIDataset(Dataset):
	"""Cycle-consistent sample builder for event-guided VFI.

	The returned sample supports:
	  [I0, I1, E01]   -> I0.5
	  [I1, I2, E12]   -> I1.5
	  [I0.5, I1.5, E0515] -> reconstruct I1
	"""

	def __init__(
		self,
		sequence_name: str,
		frame_dir: Path | str,
		frame_timestamps_csv: Path | str,
		events_path: Path | str,
		num_event_bins: int = 9,
		frame_time_unit: str = "s",
		event_time_unit: str = "us",
		sample_step: int = 1,
		spatial: Optional[SpatialConfig] = None,
		augment: Optional[AugmentConfig] = None,
		normalize_mean_std: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
		force_grayscale: bool = False,
		use_frames: bool = True,
		use_events: bool = True,
		task_mode: str = "self_supervised",
		near_empty_event_threshold: int = 1,
		strict_validation: bool = True,
		rng_seed: int = 0,
	) -> None:
		super().__init__()
		self.sequence_name = sequence_name
		self.frame_dir = Path(frame_dir)
		self.frame_timestamps_csv = Path(frame_timestamps_csv)
		self.events_path = Path(events_path)

		self.num_event_bins = int(num_event_bins)
		self.sample_step = int(sample_step)
		self.spatial = spatial or SpatialConfig()
		self.augment = augment or AugmentConfig()
		self.normalize_mean_std = normalize_mean_std
		self.force_grayscale = bool(force_grayscale)
		self.use_frames = bool(use_frames)
		self.use_events = bool(use_events)
		self.task_mode = task_mode
		self.near_empty_event_threshold = int(near_empty_event_threshold)
		self.strict_validation = bool(strict_validation)
		self.rng = np.random.default_rng(rng_seed)

		if not self.use_frames and not self.use_events:
			raise ValueError("At least one modality must be enabled: use_frames/use_events")
		if self.task_mode not in {"self_supervised", "supervised"}:
			raise ValueError("task_mode must be 'self_supervised' or 'supervised'")
		if self.num_event_bins <= 0:
			raise ValueError("num_event_bins must be > 0")
		if self.sample_step <= 0:
			raise ValueError("sample_step must be > 0")

		self.frames = load_frame_records(
			frame_dir=self.frame_dir,
			timestamps_csv=self.frame_timestamps_csv,
			frame_time_unit=frame_time_unit,
		)
		self.events = load_event_stream(self.events_path, event_time_unit=event_time_unit)

		# Check global overlap between frame timeline and event timeline.
		if self.events.t_us.size > 0:
			f_start = self.frames[0].timestamp_us
			f_end = self.frames[-1].timestamp_us
			e_start = float(self.events.t_us[0])
			e_end = float(self.events.t_us[-1])
			if strict_validation and not (e_start <= f_start and e_end >= f_end):
				raise ValueError(
					"Event stream does not fully cover frame timeline. "
					"Use synced data or disable strict_validation."
				)

		self.samples: List[SampleIndices] = []
		for i in range(0, len(self.frames) - 2, self.sample_step):
			self.samples.append(SampleIndices(i0=i, i1=i + 1, i2=i + 2))

		if not self.samples:
			raise ValueError("No valid frame triplets available")

	def __len__(self) -> int:
		return len(self.samples)

	def _choose_crop(self, h: int, w: int) -> Tuple[int, int, int, int]:
		if self.spatial.crop_hw is None:
			return 0, 0, h, w

		crop_h, crop_w = self.spatial.crop_hw
		if crop_h > h or crop_w > w:
			raise ValueError(f"Crop size {(crop_h, crop_w)} exceeds input size {(h, w)}")

		if self.spatial.random_crop:
			top = int(self.rng.integers(0, h - crop_h + 1))
			left = int(self.rng.integers(0, w - crop_w + 1))
		else:
			top = (h - crop_h) // 2
			left = (w - crop_w) // 2
		return top, left, crop_h, crop_w

	def _normalize_frame(self, x: torch.Tensor) -> torch.Tensor:
		if self.normalize_mean_std is None:
			return x
		mean, std = self.normalize_mean_std
		mean_t = torch.tensor(mean, dtype=x.dtype).view(-1, 1, 1)
		std_t = torch.tensor(std, dtype=x.dtype).view(-1, 1, 1)
		return (x - mean_t) / (std_t + 1e-8)

	def _apply_joint_spatial_ops(
		self,
		frames: Dict[str, torch.Tensor],
		voxels: Dict[str, torch.Tensor],
	) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
		# Resize first.
		if self.spatial.resize_hw is not None:
			resize_h, resize_w = self.spatial.resize_hw
			for k in list(frames.keys()):
				frames[k] = F.interpolate(
					frames[k].unsqueeze(0),
					size=(resize_h, resize_w),
					mode="bilinear",
					align_corners=False,
				).squeeze(0)
			for k in list(voxels.keys()):
				voxels[k] = F.interpolate(
					voxels[k].unsqueeze(0),
					size=(resize_h, resize_w),
					mode="bilinear",
					align_corners=False,
				).squeeze(0)

		# Shared crop.
		if frames:
			_, h, w = next(iter(frames.values())).shape
		else:
			_, h, w = next(iter(voxels.values())).shape

		top, left, crop_h, crop_w = self._choose_crop(h, w)
		for k in list(frames.keys()):
			frames[k] = _crop_tensor(frames[k], top, left, crop_h, crop_w)
		for k in list(voxels.keys()):
			voxels[k] = _crop_tensor(voxels[k], top, left, crop_h, crop_w)

		# Shared horizontal flip.
		if self.augment.horizontal_flip_prob > 0 and float(self.rng.random()) < self.augment.horizontal_flip_prob:
			for k in list(frames.keys()):
				frames[k] = torch.flip(frames[k], dims=[2])
			for k in list(voxels.keys()):
				voxels[k] = torch.flip(voxels[k], dims=[2])

		return frames, voxels

	def _build_timestamps(self, s: SampleIndices) -> Dict[str, float]:
		t0 = float(self.frames[s.i0].timestamp_us)
		t1 = float(self.frames[s.i1].timestamp_us)
		t2 = float(self.frames[s.i2].timestamp_us)

		if not (t0 < t1 < t2):
			raise RuntimeError("Triplet timestamps are not strictly monotonic")

		t05 = 0.5 * (t0 + t1)
		t15 = 0.5 * (t1 + t2)
		return {
			"t0": t0,
			"t1": t1,
			"t2": t2,
			"t0.5": t05,
			"t1.5": t15,
		}

	def _window_event_counts(self, start_us: float, end_us: float) -> int:
		lo, hi = _slice_event_indices(self.events.t_us, start_us, end_us)
		return max(0, hi - lo)

	def __getitem__(self, idx: int) -> Dict[str, Any]:
		s = self.samples[idx]
		ts = self._build_timestamps(s)

		# Load frames.
		frames: Dict[str, torch.Tensor] = {}
		if self.use_frames:
			frames["I0"] = _read_frame(self.frames[s.i0].path, force_grayscale=self.force_grayscale)
			frames["I1"] = _read_frame(self.frames[s.i1].path, force_grayscale=self.force_grayscale)
			frames["I2"] = _read_frame(self.frames[s.i2].path, force_grayscale=self.force_grayscale)

		# Temporal reversal (careful with event polarity and interval mapping).
		apply_time_reverse = (
			self.augment.temporal_reverse_prob > 0
			and float(self.rng.random()) < self.augment.temporal_reverse_prob
		)

		if apply_time_reverse:
			if self.use_frames:
				frames = {"I0": frames["I2"], "I1": frames["I1"], "I2": frames["I0"]}

			# In reversed timeline: E01 <- reverse(E12), E12 <- reverse(E01).
			ev_01 = _temporal_reverse_events(self.events, ts["t1"], ts["t2"])
			ev_12 = _temporal_reverse_events(self.events, ts["t0"], ts["t1"])
			ev_0515 = _temporal_reverse_events(self.events, ts["t0.5"], ts["t1.5"])
		else:
			ev_01 = self.events
			ev_12 = self.events
			ev_0515 = self.events

		# Build event voxel tensors.
		voxels: Dict[str, torch.Tensor] = {}
		if self.use_events:
			if self.use_frames:
				_, fh, fw = frames["I0"].shape
				sensor_h, sensor_w = fh, fw
			else:
				sensor_h, sensor_w = self.events.sensor_size

			if apply_time_reverse:
				# ev_01 was built by reversing events from [t1,t2]; timestamps live in [t1,t2].
				# ev_12 was built by reversing events from [t0,t1]; timestamps live in [t0,t1].
				# Pass the matching window so _slice_event_indices actually finds the events.
				voxels["E01"] = _events_window_to_voxel(
					ev_01, ts["t1"], ts["t2"], self.num_event_bins, sensor_h, sensor_w
				)
				voxels["E12"] = _events_window_to_voxel(
					ev_12, ts["t0"], ts["t1"], self.num_event_bins, sensor_h, sensor_w
				)
				voxels["E0515"] = _events_window_to_voxel(
					ev_0515, ts["t0.5"], ts["t1.5"], self.num_event_bins, sensor_h, sensor_w
				)
			else:
				voxels["E01"] = _events_window_to_voxel(
					self.events, ts["t0"], ts["t1"], self.num_event_bins, sensor_h, sensor_w
				)
				voxels["E12"] = _events_window_to_voxel(
					self.events, ts["t1"], ts["t2"], self.num_event_bins, sensor_h, sensor_w
				)
				voxels["E0515"] = _events_window_to_voxel(
					self.events, ts["t0.5"], ts["t1.5"], self.num_event_bins, sensor_h, sensor_w
				)

		frames, voxels = self._apply_joint_spatial_ops(frames, voxels)

		# Per-frame normalization after geometric transforms.
		for k in list(frames.keys()):
			frames[k] = self._normalize_frame(frames[k])

		# Sanity flags for sparse event windows.
		# After temporal reversal E01 draws from [t1,t2] and E12 from [t0,t1].
		if apply_time_reverse:
			count_e01 = self._window_event_counts(ts["t1"], ts["t2"])
			count_e12 = self._window_event_counts(ts["t0"], ts["t1"])
		else:
			count_e01 = self._window_event_counts(ts["t0"], ts["t1"])
			count_e12 = self._window_event_counts(ts["t1"], ts["t2"])
		count_e0515 = self._window_event_counts(ts["t0.5"], ts["t1.5"])

		sample: Dict[str, Any] = {
			**frames,
			**voxels,
			"timestamps": ts,
			"meta": {
				"sequence_name": self.sequence_name,
				"sample_index": idx,
				"frame_indices": (self.frames[s.i0].index, self.frames[s.i1].index, self.frames[s.i2].index),
				"temporal_reversed": apply_time_reverse,
				"event_counts": {
					"E01": count_e01,
					"E12": count_e12,
					"E0515": count_e0515,
				},
				"near_empty_events": {
					"E01": count_e01 < self.near_empty_event_threshold,
					"E12": count_e12 < self.near_empty_event_threshold,
					"E0515": count_e0515 < self.near_empty_event_threshold,
				},
			},
		}

		# Optional modality ablations.
		if not self.use_frames:
			sample.pop("I0", None)
			sample.pop("I1", None)
			sample.pop("I2", None)
		if not self.use_events:
			sample.pop("E01", None)
			sample.pop("E12", None)
			sample.pop("E0515", None)

		return sample


def build_self_supervised_vfi_dataset(
	sequence_name: str,
	frame_dir: Path | str,
	frame_timestamps_csv: Path | str,
	events_path: Path | str,
	num_event_bins: int = 9,
	frame_time_unit: str = "s",
	event_time_unit: str = "us",
	sample_step: int = 1,
	spatial: Optional[SpatialConfig] = None,
	augment: Optional[AugmentConfig] = None,
	normalize_mean_std: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
	force_grayscale: bool = False,
	use_frames: bool = True,
	use_events: bool = True,
	task_mode: str = "self_supervised",
	near_empty_event_threshold: int = 1,
	strict_validation: bool = True,
	rng_seed: int = 0,
) -> SelfSupervisedEventVFIDataset:
	"""Factory function for cleaner call sites."""
	return SelfSupervisedEventVFIDataset(
		sequence_name=sequence_name,
		frame_dir=frame_dir,
		frame_timestamps_csv=frame_timestamps_csv,
		events_path=events_path,
		num_event_bins=num_event_bins,
		frame_time_unit=frame_time_unit,
		event_time_unit=event_time_unit,
		sample_step=sample_step,
		spatial=spatial,
		augment=augment,
		normalize_mean_std=normalize_mean_std,
		force_grayscale=force_grayscale,
		use_frames=use_frames,
		use_events=use_events,
		task_mode=task_mode,
		near_empty_event_threshold=near_empty_event_threshold,
		strict_validation=strict_validation,
		rng_seed=rng_seed,
	)

