"""Video processing utilities for hybrid high-speed microscopy.

Provides functions to:
- Read AVI frame-by-frame
- Extract video metadata (fps, resolution, frame count, duration)
- Read event .raw files as event arrays
- Merge/stack event frames with conventional video frames
  to synthesize high-speed output
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from utils.config import load_config


# ---------------------------------------------------------------------------
# Video (AVI) helpers
# ---------------------------------------------------------------------------

def get_video_metadata(video_path: str | Path) -> dict:
    """Return metadata dict for a video file.

    Keys: fps, frame_count, width, height, duration_s, codec.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }
    meta["duration_s"] = meta["frame_count"] / meta["fps"] if meta["fps"] > 0 else 0.0
    cap.release()
    return meta


def iter_video_frames(video_path: str | Path, grayscale: bool = False):
    """Yield (frame_index, frame_array) for every frame in the video.

    Parameters
    ----------
    video_path : path to the video file
    grayscale  : if True, convert each frame to single-channel uint8
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if grayscale and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        yield idx, frame
        idx += 1
    cap.release()


def get_video_frame(video_path: str | Path, frame_idx: int, grayscale: bool = False) -> np.ndarray:
    """Read a single frame by index.

    Returns the frame as a numpy array (BGR or grayscale).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IndexError(f"Cannot read frame {frame_idx} from {video_path}")
    if grayscale and frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


# ---------------------------------------------------------------------------
# Event (.raw) helpers
# ---------------------------------------------------------------------------

def iter_event_frames(raw_path: str | Path, delta_t_us: int = 1000):
    """Yield event arrays from an OpenEB .raw file in slices of *delta_t_us*.

    Each yielded array has dtype with fields (x, y, p, t).
    Requires metavision_core.
    """
    from metavision_core.event_io import EventsIterator

    it = EventsIterator(input_path=str(raw_path), delta_t=delta_t_us)
    for evs in it:
        if evs.size:
            yield evs


def accumulate_events(events: np.ndarray, height: int, width: int,
                      mode: str = "count") -> np.ndarray:
    """Accumulate an event array into a 2-D image.

    Parameters
    ----------
    events : structured array with (x, y, p, t)
    height, width : sensor dimensions
    mode : "count" -> signed count (ON - OFF) mapped to uint8
           "binary" -> 255 where any event occurred, else 0

    Returns
    -------
    uint8 image (H, W)
    """
    if mode == "binary":
        img = np.zeros((height, width), dtype=np.uint8)
        if events.size:
            ys = events["y"].astype(np.intp)
            xs = events["x"].astype(np.intp)
            img[ys, xs] = 255
        return img

    # signed count
    acc = np.zeros((height, width), dtype=np.int32)
    if events.size:
        on = events["p"] > 0
        off = ~on
        ys = events["y"].astype(np.intp)
        xs = events["x"].astype(np.intp)
        if on.any():
            np.add.at(acc, (ys[on], xs[on]), 1)
        if off.any():
            np.add.at(acc, (ys[off], xs[off]), -1)
    img = np.clip(128 + 4.0 * acc, 0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Merge / stack:  conventional frames  +  event data  →  high-speed output
# ---------------------------------------------------------------------------

def merge_frame_and_events(base_frame: np.ndarray,
                           event_img: np.ndarray,
                           alpha: float = 0.5) -> np.ndarray:
    """Alpha-blend a base video frame with an event accumulation image.

    Both inputs are converted to float, blended, and returned as uint8.
    """
    if base_frame.ndim == 3 and event_img.ndim == 2:
        event_img = cv2.cvtColor(event_img, cv2.COLOR_GRAY2BGR)
    if base_frame.ndim == 2 and event_img.ndim == 3:
        base_frame = cv2.cvtColor(base_frame, cv2.COLOR_GRAY2BGR)

    blended = cv2.addWeighted(base_frame.astype(np.float32), 1.0 - alpha,
                              event_img.astype(np.float32), alpha, 0)
    return np.clip(blended, 0, 255).astype(np.uint8)


def generate_high_speed_frames(cfg=None):
    """Main pipeline: merge conventional video frames with event data.

    Reads settings from config.json section "video_processing" (or accepts
    a pre-loaded config namespace).

    For each conventional frame period, multiple event sub-frames are
    generated and blended with the base frame to produce a higher
    effective frame rate.

    Yields (output_index, merged_frame) numpy arrays.
    """
    if cfg is None:
        cfg = load_config("video_processing")

    video_path = Path(cfg.video_path)
    raw_path = Path(cfg.raw_path)
    output_fps = int(cfg.output_fps)
    alpha = float(cfg.blend_alpha)
    grayscale = bool(cfg.grayscale)

    vmeta = get_video_metadata(video_path)
    video_fps = vmeta["fps"]
    height, width = vmeta["height"], vmeta["width"]

    # How many event sub-frames per conventional frame
    upsample = max(1, int(round(output_fps / video_fps)))
    sub_period_us = int(round(1e6 / output_fps))

    from metavision_core.event_io import EventsIterator
    ev_iter = EventsIterator(input_path=str(raw_path), delta_t=sub_period_us)

    out_idx = 0
    for frame_idx, base_frame in iter_video_frames(video_path, grayscale=grayscale):
        if grayscale and base_frame.ndim == 2:
            base_bgr = cv2.cvtColor(base_frame, cv2.COLOR_GRAY2BGR)
        else:
            base_bgr = base_frame

        for sub in range(upsample):
            try:
                evs = next(ev_iter)
            except StopIteration:
                return
            ev_img = accumulate_events(evs, height, width)
            merged = merge_frame_and_events(base_bgr, ev_img, alpha=alpha)
            yield out_idx, merged
            out_idx += 1


def save_high_speed_video(cfg=None):
    """Run the merge pipeline and write output to an AVI file.

    Reads settings from config.json "video_processing" section.
    """
    if cfg is None:
        cfg = load_config("video_processing")

    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_fps = int(cfg.output_fps)

    writer = None
    count = 0
    for idx, frame in generate_high_speed_frames(cfg):
        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(str(out_path), fourcc, output_fps, (w, h))
        writer.write(frame)
        count += 1

    if writer is not None:
        writer.release()
    print(f"Wrote {count} frames to {out_path} at {output_fps} fps")
