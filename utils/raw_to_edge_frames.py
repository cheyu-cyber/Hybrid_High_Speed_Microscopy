#!/usr/bin/env python3
"""Convert a .raw event recording into edge-velocity visualisation frames.

Pipeline per time window:
  1. Read events from .raw via EventsIterator
  2. Denoise:  hot pixel → refractory → NN activity (→ optional polarity)
  3. Build event density / polarity maps from filtered events
  4. Accumulate a conventional-style grayscale event frame
  5. Compute spatial gradient (Sobel) on that frame → edge magnitude & direction
  6. Estimate per-pixel edge-normal velocity from gradient + event polarity
  7. Render three-panel visualisation per window and save to disk

Output panels (left → right):
  A. Denoised event frame (grayscale)
  B. Edge speed heat-map (jet colourmap)
  C. Edge velocity quiver overlay on event frame

Configuration is read from config.json section "raw_to_edge_frames".
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from config import load_config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.event_processing import (
    filter_events,
    compute_event_density,
    detect_edges,
    estimate_edge_velocity,
    window_image,
    save_event_image,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_args():
    """Load configuration from config.json (section: raw_to_edge_frames)."""
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


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _speed_heatmap(speed, speed_max_clip):
    """Render speed as a JET-coloured uint8 image via matplotlib's colourmap."""
    import cv2

    if speed_max_clip > 0:
        vmax = speed_max_clip
    else:
        vmax = speed.max() if speed.max() > 0 else 1.0

    norm = np.clip(speed / vmax, 0, 1)
    gray_u8 = (norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(gray_u8, cv2.COLORMAP_JET)
    # Mask regions with zero speed → black
    heatmap[speed == 0] = 0
    return heatmap


def _angle_to_bgr(angle_rad):
    """Map an angle (radians) to a BGR colour via HSV hue wheel.

    0 → red (right), π/2 → green (down), π → cyan (left), 3π/2 → magenta (up).
    """
    import cv2
    hue = ((np.degrees(angle_rad) % 360) / 2).astype(np.uint8)  # 0-179
    hsv = np.stack([hue,
                    np.full_like(hue, 255),
                    np.full_like(hue, 255)], axis=-1).reshape(1, -1, 3)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)
    return bgr


def _quiver_overlay(base_bgr, vx, vy, mask, arrow_step, arrow_scale):
    """Draw velocity arrows on top of a BGR image.

    Arrow colour encodes movement direction via the HSV hue wheel:
      red → right, green → down, cyan → left, magenta → up.
    """
    import cv2

    canvas = base_bgr.copy()
    h, w = vx.shape

    # Collect all arrows first so we can batch-convert colours
    arrows = []  # (x0, y0, dx, dy)
    for y0 in range(arrow_step // 2, h, arrow_step):
        for x0 in range(arrow_step // 2, w, arrow_step):
            y_lo = max(y0 - arrow_step // 2, 0)
            y_hi = min(y0 + arrow_step // 2, h)
            x_lo = max(x0 - arrow_step // 2, 0)
            x_hi = min(x0 + arrow_step // 2, w)

            patch_mask = mask[y_lo:y_hi, x_lo:x_hi]
            if not patch_mask.any():
                continue

            dx = vx[y_lo:y_hi, x_lo:x_hi][patch_mask].mean()
            dy = vy[y_lo:y_hi, x_lo:x_hi][patch_mask].mean()

            mag = np.sqrt(dx * dx + dy * dy)
            if mag < 1e-6:
                continue

            arrows.append((x0, y0, dx, dy))

    if not arrows:
        return canvas

    arr = np.array(arrows, dtype=np.float64)
    angles = np.arctan2(arr[:, 3], arr[:, 2])  # direction angle
    bgr_colours = _angle_to_bgr(angles)

    for i, (x0, y0, dx, dy) in enumerate(arrows):
        x1 = int(x0 + dx * arrow_scale)
        y1 = int(y0 + dy * arrow_scale)
        c = tuple(int(v) for v in bgr_colours[i])
        cv2.arrowedLine(canvas, (int(x0), int(y0)), (x1, y1), c, 2,
                        tipLength=0.25)

    return canvas


def _to_bgr(gray_u8):
    """Stack a grayscale image to 3-channel BGR."""
    return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    import cv2

    # Timing
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

    iterator_start_us = (start_us // delta_t_us) * delta_t_us
    iterator_max_duration = None
    if proc_end_us is not None:
        iterator_max_duration = int(proc_end_us - iterator_start_us)

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

    # CSV log
    csv_path = out_dir / "edge_frames.csv"
    csv_f = csv_path.open("w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["frame_idx", "t_start_us", "t_end_us",
                     "events_raw", "events_filtered", "filename"])

    # Config shortcuts
    arrow_step = int(getattr(args, "arrow_step", 16))
    arrow_scale = float(getattr(args, "arrow_scale", 2.0))
    speed_max_clip = float(getattr(args, "speed_max_clip", 0))
    edge_ksize = int(getattr(args, "edge_kernel_size", 5))
    grad_thresh = float(getattr(args, "grad_threshold", 10.0))
    c_thresh = float(getattr(args, "contrast_threshold", 0.15))
    grad_min = float(getattr(args, "grad_min", 5.0))
    fmt = getattr(args, "to", "png")

    # Accumulators for the current window
    active_win = None
    frame_idx = 0
    next_win_start = start_us
    cursor_us = iterator_start_us

    def start_window(t_start):
        nonlocal active_win
        t_end = t_start + accum_us
        if proc_end_us is not None:
            if t_start >= proc_end_us:
                return False
            t_end = min(t_end, proc_end_us)
        if t_end <= t_start:
            return False
        if args.max_frames and frame_idx >= int(args.max_frames):
            return False
        active_win = {
            "start_us": int(t_start),
            "end_us": int(t_end),
            "t": [], "x": [], "y": [], "p": [],
        }
        return True

    def flush_window():
        nonlocal active_win, frame_idx
        if active_win is None:
            return
        win = active_win
        active_win = None

        # Concatenate all event chunks
        if win["t"]:
            ev_t = np.concatenate(win["t"])
            ev_x = np.concatenate(win["x"])
            ev_y = np.concatenate(win["y"])
            ev_p = np.concatenate(win["p"])
        else:
            ev_t = np.empty(0, dtype=np.float64)
            ev_x = np.empty(0, dtype=np.int64)
            ev_y = np.empty(0, dtype=np.int64)
            ev_p = np.empty(0, dtype=np.float32)

        n_raw = ev_t.size

        # --- 1. Denoise ---
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
            ft, fx, fy, fp = filt["events_t"], filt["events_x"], filt["events_y"], filt["events_p"]
        else:
            ft, fx, fy, fp = ev_t, ev_x, ev_y, ev_p

        n_filt = ft.size

        # --- 2. Build event frame (count_gray, contrast=8) for edge detection ---
        on_acc = np.zeros((h, w), dtype=np.int32)
        off_acc = np.zeros((h, w), dtype=np.int32)
        if n_filt > 0:
            on_mask = fp > 0
            if on_mask.any():
                np.add.at(on_acc, (fy[on_mask].astype(np.intp), fx[on_mask].astype(np.intp)), 1)
            off_mask = fp <= 0
            if off_mask.any():
                np.add.at(off_acc, (fy[off_mask].astype(np.intp), fx[off_mask].astype(np.intp)), 1)

        signed = on_acc - off_acc
        ev_frame = window_image(np, signed, contrast=8.0, bit_depth=8)

        # --- 3. Edge detection on event frame ---
        edges = detect_edges(ev_frame, kernel_size=edge_ksize,
                             grad_threshold=grad_thresh)

        # --- 4. Event density + polarity ---
        if n_filt > 0:
            density = compute_event_density(fx, fy, fp, h, w)
            count_map = density["count_map"]
            pol_map = density["polarity_map"]
        else:
            count_map = np.zeros((h, w), dtype=np.int32)
            pol_map = np.zeros((h, w), dtype=np.float32)

        # --- 5. Edge velocity ---
        vel = estimate_edge_velocity(
            edges["magnitude"], edges["direction"],
            count_map, pol_map,
            contrast_threshold=c_thresh,
            grad_min=grad_min,
        )

        # --- 6. Render 3-panel visualisation ---
        # Panel A: denoised event frame (grayscale → BGR)
        panel_a = _to_bgr(ev_frame)

        # Panel B: speed heatmap
        panel_b = _speed_heatmap(vel["speed"], speed_max_clip)

        # Panel C: quiver arrows on event frame
        panel_c = _quiver_overlay(
            panel_a, vel["velocity_x"], vel["velocity_y"], vel["mask"],
            arrow_step, arrow_scale,
        )

        composite = np.concatenate([panel_a, panel_b, panel_c], axis=1)

        out_name = f"edge_{frame_idx:06d}.{fmt}"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), composite)

        csv_w.writerow([frame_idx, win["start_us"], win["end_us"],
                         n_raw, n_filt, out_name])
        frame_idx += 1

    # --- Main loop ---
    print(f"Edge-frame pipeline: period={period_us} µs, accum={accum_us} µs, "
          f"denoise(hot={args.hot_pixel_freq}, refrac={args.refractory_us}, "
          f"nn={args.nn_delta_t_us})")
    try:
        for evs in it:
            slice_end_us = cursor_us + delta_t_us
            cursor_us = slice_end_us

            # Start new windows as needed
            while active_win is None and next_win_start < slice_end_us:
                if not start_window(next_win_start):
                    break
                next_win_start += period_us

            # Collect events into current window
            if evs.size and active_win is not None:
                t_all = evs["t"].astype(np.float64, copy=False)
                x_all = evs["x"].astype(np.int64, copy=False)
                y_all = evs["y"].astype(np.int64, copy=False)
                p_raw = evs["p"].astype(np.float32, copy=False)
                # Convert Metavision polarity (0=OFF) to signed (−1=OFF)
                p_all = np.where(p_raw > 0, 1.0, -1.0).astype(np.float32)

                valid = (t_all >= active_win["start_us"]) & (t_all < active_win["end_us"])
                if proc_end_us is not None:
                    valid &= t_all < proc_end_us
                if valid.any():
                    active_win["t"].append(t_all[valid])
                    active_win["x"].append(x_all[valid])
                    active_win["y"].append(y_all[valid])
                    active_win["p"].append(p_all[valid])

            # Flush if window ended
            if active_win is not None and slice_end_us >= active_win["end_us"]:
                flush_window()

            if args.max_frames and frame_idx >= int(args.max_frames):
                break
            if proc_end_us is not None and slice_end_us >= proc_end_us:
                break

        # Flush trailing window
        if active_win is not None:
            flush_window()

    finally:
        csv_f.close()

    print(f"Wrote {frame_idx} edge frames to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
