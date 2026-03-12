#!/usr/bin/env python3
"""Convert a .raw event recording into edge-velocity visualisation frames.

Pipeline per time window:
  raw events → denoise → edge detection → velocity estimation → 3-panel image

Output panels (left → right):
  A. Denoised event frame (grayscale)
  B. Edge speed heat-map (JET colourmap)
  C. Edge velocity quiver overlay (HSV direction-coded arrows)

Configuration is read from config.json section "raw_to_edge_frames".
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from utils.config import load_config
from datasets.event_processing import (
    filter_events,
    compute_event_density,
    detect_edges,
    estimate_edge_velocity,
    window_image,
    speed_heatmap,
    angle_to_bgr,
    quiver_overlay,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Config ────────────────────────────────────────────────────────────────

def load_args():
    """Load and validate config.json section "raw_to_edge_frames"."""
    args = load_config("raw_to_edge_frames")

    if not args.input_raw:
        print("'input_raw' must be set in config.json", file=sys.stderr)
        sys.exit(2)
    if not args.out_dir:
        print("'out_dir' must be set in config.json", file=sys.stderr)
        sys.exit(2)

    # Resolve relative paths against project root
    args.input_raw = str((PROJECT_ROOT / args.input_raw).resolve())
    args.out_dir = str((PROJECT_ROOT / args.out_dir).resolve())
    return args


def resolve_timing(args):
    """Derive period_us, accum_us, delta_t_us, start_us, proc_end_us."""
    if args.period_us > 0:
        period_us = int(args.period_us)
    elif args.fps > 0:
        period_us = int(round(1e6 / float(args.fps)))
    elif args.accum_us > 0:
        period_us = int(args.accum_us)
    else:
        period_us = 33_333  # ~30 fps

    accum_us = int(args.accum_us) if getattr(args, "accum_us", 0) > 0 else period_us
    delta_t_us = int(args.delta_t_us)
    start_us = int(args.start_us)
    proc_end_us = (start_us + int(args.duration_us)) if (args.duration_us and args.duration_us > 0) else None

    return period_us, accum_us, delta_t_us, start_us, proc_end_us



def to_bgr(gray_u8):
    """Grayscale → 3-channel BGR."""
    return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)


# ── Windowed event iteration ─────────────────────────────────────────────

def iter_event_windows(iterator, period_us, accum_us, delta_t_us,
                       start_us, proc_end_us, max_frames):
    """Yield (t_start, t_end, events_t, events_x, events_y, events_p) per window.

    Handles the sliding-window accumulation over EventsIterator chunks so that
    callers only see complete, concatenated windows.
    """
    cursor_us = (start_us // delta_t_us) * delta_t_us
    next_win_start = start_us
    frame_count = 0

    # Current accumulation window
    win_start = win_end = 0
    buf_t, buf_x, buf_y, buf_p = [], [], [], []
    active = False

    def _open_window():
        nonlocal active, win_start, win_end, buf_t, buf_x, buf_y, buf_p
        t_end = next_win_start + accum_us
        if proc_end_us is not None:
            if next_win_start >= proc_end_us:
                return False
            t_end = min(t_end, proc_end_us)
        if t_end <= next_win_start:
            return False
        win_start, win_end = int(next_win_start), int(t_end)
        buf_t, buf_x, buf_y, buf_p = [], [], [], []
        active = True
        return True

    for evs in iterator:
        slice_end_us = cursor_us + delta_t_us
        cursor_us = slice_end_us

        # Open a window if we don't have one
        while not active and next_win_start < slice_end_us:
            if max_frames and frame_count >= max_frames:
                return
            if not _open_window():
                return
            next_win_start += period_us

        # Collect events into the active window
        if evs.size and active:
            t = evs["t"].astype(np.float64, copy=False)
            x = evs["x"].astype(np.int64, copy=False)
            y = evs["y"].astype(np.int64, copy=False)
            p_raw = evs["p"].astype(np.float32, copy=False)
            p = np.where(p_raw > 0, 1.0, -1.0).astype(np.float32)

            valid = (t >= win_start) & (t < win_end)
            if proc_end_us is not None:
                valid &= t < proc_end_us
            if valid.any():
                buf_t.append(t[valid])
                buf_x.append(x[valid])
                buf_y.append(y[valid])
                buf_p.append(p[valid])

        # Flush when window is complete
        if active and slice_end_us >= win_end:
            if buf_t:
                yield (win_start, win_end,
                       np.concatenate(buf_t), np.concatenate(buf_x),
                       np.concatenate(buf_y), np.concatenate(buf_p))
            else:
                empty = np.empty(0)
                yield (win_start, win_end,
                       empty, empty.astype(np.int64),
                       empty.astype(np.int64), empty.astype(np.float32))
            active = False
            frame_count += 1

        if max_frames and frame_count >= max_frames:
            return
        if proc_end_us is not None and slice_end_us >= proc_end_us:
            return

    # Flush a trailing partial window
    if active:
        if buf_t:
            yield (win_start, win_end,
                   np.concatenate(buf_t), np.concatenate(buf_x),
                   np.concatenate(buf_y), np.concatenate(buf_p))
        else:
            empty = np.empty(0)
            yield (win_start, win_end,
                   empty, empty.astype(np.int64),
                   empty.astype(np.int64), empty.astype(np.float32))


# ── Per-window processing ─────────────────────────────────────────────────

def process_window(ev_t, ev_x, ev_y, ev_p, h, w, args):
    """Denoise → edge detect → velocity estimate.  Returns (ev_frame, vel, n_raw, n_filt)."""
    n_raw = ev_t.size

    # 1. Denoise
    if n_raw > 0:
        filt = filter_events(
            ev_t, ev_x, ev_y, ev_p, h, w,
            hot_pixel_freq=float(args.hot_pixel_freq),
            refractory_us=float(args.refractory_us),
            nn_delta_t_us=float(args.nn_delta_t_us),
            polarity_consistency=bool(args.polarity_consistency),
            polarity_delta_t_us=float(args.polarity_delta_t_us),
            polarity_min_agreement=float(args.polarity_min_agreement),
        )
        fx, fy, fp = filt["events_x"], filt["events_y"], filt["events_p"]
    else:
        fx, fy, fp = ev_x, ev_y, ev_p
    n_filt = fx.size

    # 2. Accumulate signed event frame
    on_acc = np.zeros((h, w), dtype=np.int32)
    off_acc = np.zeros((h, w), dtype=np.int32)
    if n_filt > 0:
        on = fp > 0
        if on.any():
            np.add.at(on_acc, (fy[on].astype(np.intp), fx[on].astype(np.intp)), 1)
        off = fp <= 0
        if off.any():
            np.add.at(off_acc, (fy[off].astype(np.intp), fx[off].astype(np.intp)), 1)
    ev_frame = window_image(np, on_acc - off_acc,
                             contrast=float(args.contrast),
                             bit_depth=int(args.bit_depth))

    # 3. Edge detection
    edges = detect_edges(ev_frame,
                         kernel_size=int(args.edge_kernel_size),
                         grad_threshold=float(args.grad_threshold))

    # 4. Event density + polarity map
    if n_filt > 0:
        density = compute_event_density(fx, fy, fp, h, w)
        count_map, pol_map = density["count_map"], density["polarity_map"]
    else:
        count_map = np.zeros((h, w), dtype=np.int32)
        pol_map = np.zeros((h, w), dtype=np.float32)

    # 5. Edge velocity
    vel = estimate_edge_velocity(
        edges["magnitude"], edges["direction"],
        count_map, pol_map,
        contrast_threshold=float(args.contrast_threshold),
        grad_min=float(args.grad_min),
    )

    return ev_frame, vel, n_raw, n_filt


def render_panels(ev_frame, vel, args):
    """Build and return the 3 visualisation panels separately."""
    panel_a = to_bgr(ev_frame)
    panel_b = speed_heatmap(vel["speed"], float(args.speed_max_clip))
    panel_c = quiver_overlay(
        panel_a, vel["velocity_x"], vel["velocity_y"], vel["mask"],
        int(args.arrow_step), float(args.arrow_scale),
    )
    return panel_a, panel_b, panel_c


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> int:
    args = load_args()

    input_path = Path(args.input_raw)
    if not input_path.is_file():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from metavision_core.event_io import EventsIterator
    except Exception as e:
        print(f"Missing: metavision_core  ({e})", file=sys.stderr)
        return 4

    period_us, accum_us, delta_t_us, start_us, proc_end_us = resolve_timing(args)

    iterator_start_us = (start_us // delta_t_us) * delta_t_us
    iterator_max_duration = int(proc_end_us - iterator_start_us) if proc_end_us else None

    it = EventsIterator(
        input_path=str(input_path),
        start_ts=int(iterator_start_us),
        delta_t=int(delta_t_us),
        max_duration=iterator_max_duration,
    )

    h, w = it.get_size()
    if h is None or w is None:
        print("Could not read sensor size from file.", file=sys.stderr)
        return 6

    max_frames = int(args.max_frames) if args.max_frames else 0
    fmt = getattr(args, "to", "png")

    # Output folders
    event_dir = out_dir / "event_frame"
    speed_dir = out_dir / "speed_heatmap"
    quiver_dir = out_dir / "quiver_overlay"
    event_dir.mkdir(parents=True, exist_ok=True)
    speed_dir.mkdir(parents=True, exist_ok=True)
    quiver_dir.mkdir(parents=True, exist_ok=True)

    # CSV log
    csv_path = out_dir / "edge_frames.csv"
    csv_f = csv_path.open("w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["frame_idx", "t_start_us", "t_end_us",
                     "events_raw", "events_filtered",
                     "event_file", "speed_file", "quiver_file"])

    print(f"Edge-frame pipeline: period={period_us} µs, accum={accum_us} µs, "
          f"denoise(hot={args.hot_pixel_freq}, refrac={args.refractory_us}, "
          f"nn={args.nn_delta_t_us})")

    frame_idx = 0
    try:
        windows = iter_event_windows(
            it, period_us, accum_us, delta_t_us,
            start_us, proc_end_us, max_frames,
        )
        for t_start, t_end, ev_t, ev_x, ev_y, ev_p in windows:
            ev_frame, vel, n_raw, n_filt = process_window(
                ev_t, ev_x, ev_y, ev_p, h, w, args,
            )
            panel_a, panel_b, panel_c = render_panels(ev_frame, vel, args)

            event_name = f"event_{frame_idx:06d}.{fmt}"
            speed_name = f"speed_{frame_idx:06d}.{fmt}"
            quiver_name = f"quiver_{frame_idx:06d}.{fmt}"

            event_rel = f"event_frame/{event_name}"
            speed_rel = f"speed_heatmap/{speed_name}"
            quiver_rel = f"quiver_overlay/{quiver_name}"

            cv2.imwrite(str(event_dir / event_name), panel_a)
            cv2.imwrite(str(speed_dir / speed_name), panel_b)
            cv2.imwrite(str(quiver_dir / quiver_name), panel_c)

            csv_w.writerow([
                frame_idx, t_start, t_end, n_raw, n_filt,
                event_rel, speed_rel, quiver_rel,
            ])
            frame_idx += 1
    finally:
        csv_f.close()

    print(f"Wrote {frame_idx} edge frames to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
