#!/usr/bin/env python3
"""Convert an OpenEB/Metavision event .raw recording into accumulated image frames.

This creates an intensity-like visualization by accumulating events into explicit
time windows.
You control:
- frame period (window start spacing), and
- accumulation duration (window length).

When `accum_us == period_us` windows are non-overlapping.
When `accum_us < period_us` there are gaps.
When `accum_us > period_us` windows overlap.

Output:
- 8-bit PGM frames (fast, viewable, no extra deps)
- PNG or TIFF if Pillow is installed (optional 16-bit output)
- CSV timestamps for each frame (start/end ts)

Notes:
- Requires Metavision/OpenEB Python modules (metavision_core, metavision_sdk_core) and numpy.
- If you built OpenEB from source, make sure you sourced the environment script so Python can import them.

Example:
  python3 tools/raw_to_frames.py --in runs/session_*/events.raw --out runs/session_*/ev_frames --fps 100
  python3 tools/raw_to_frames.py --in events.raw --out ev_frames --render-mode metavision_dark --to png
  python3 tools/raw_to_frames.py --in events.raw --start-us 17477823 --period-us 33365 --duration-us 33365 --max-frames 1
  
  python3 tools/raw_to_frames.py \
  --in runs/session_20260224_155826/events_cut.raw \
  --out runs/session_20260224_155826/ev_frames_1ms \
  --start-us 0 \
  --period-us 33367 \
  --accum-us 5000 \
  --to pgm --render-mode metavision_dark
  """

from __future__ import annotations

import csv
import sys
from pathlib import Path

from utils.config import load_config
from datasets.event_processing import window_image, bgr_to_gray_u8, save_event_image


def load_args():
    """Load configuration from config.json (section: raw_to_frames)."""
    args = load_config("raw_to_frames")

    # Validate required fields
    if not args.input_raw:
        print("'input_raw' must be set in config.json", file=sys.stderr)
        sys.exit(2)
    if not args.out_dir:
        print("'out_dir' must be set in config.json", file=sys.stderr)
        sys.exit(2)

    # Enforce mutual exclusivity: fps and accum_us
    if args.fps and args.accum_us:
        print("'fps' and 'accum_us' are mutually exclusive in config.json", file=sys.stderr)
        sys.exit(2)

    return args


# _window_image, _bgr_to_gray_u8, _save_image moved to datasets/event_processing.py
# Imported above as: window_image, bgr_to_gray_u8, save_event_image


def main() -> int:
    args = load_args()

    input_path = Path(args.input_raw)
    if not input_path.is_file():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Imports (numpy + OpenEB python bindings)
    try:
        import numpy as np
    except Exception as e:
        print("Missing dependency: numpy", file=sys.stderr)
        print("Install with:", file=sys.stderr)
        print("  python3 -m pip install --user numpy", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        return 3

    try:
        from metavision_core.event_io import EventsIterator
    except Exception as e:
        print("Missing dependency: metavision_core.event_io (OpenEB Python bindings)", file=sys.stderr)
        print("Make sure you sourced OpenEB's environment (e.g. setup_env.sh) or installed the Python wheel.", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        return 4

    save_pil = args.to in ("png", "tiff")
    pil_Image = None
    if save_pil:
        try:
            from PIL import Image
            pil_Image = Image
        except Exception as e:
            print("PNG/TIFF output requested but Pillow is missing.", file=sys.stderr)
            print("Install with:", file=sys.stderr)
            print("  python3 -m pip install --user pillow", file=sys.stderr)
            print(f"Import error: {e}", file=sys.stderr)
            return 5

    if args.bit_depth == 16 and not save_pil:
        print("--bit-depth 16 is only supported with --to png or --to tiff", file=sys.stderr)
        return 2

    if args.render_mode == "metavision_dark" and args.bit_depth == 16:
        print("--bit-depth 16 is not supported with --render-mode metavision_dark (OpenEB renderer is 8-bit).",
              file=sys.stderr)
        return 2

    if args.render_mode == "metavision_dark" and args.contrast != 4.0:
        print("Warning: --contrast is ignored in --render-mode metavision_dark.", file=sys.stderr)

    if args.render_mode != "metavision_dark" and args.metavision_gray:
        print("Warning: --metavision-gray has effect only with --render-mode metavision_dark.", file=sys.stderr)

    base_frame_algo = None
    mv_palette = None
    if args.render_mode == "metavision_dark":
        try:
            from metavision_sdk_core import BaseFrameGenerationAlgorithm, ColorPalette
            base_frame_algo = BaseFrameGenerationAlgorithm
            mv_palette = ColorPalette.Gray if args.metavision_gray else ColorPalette.Dark
        except Exception as e:
            print("Missing dependency: metavision_sdk_core (required for --render-mode metavision_dark).",
                  file=sys.stderr)
            print("Make sure OpenEB/Metavision Python environment is sourced.", file=sys.stderr)
            print(f"Import error: {e}", file=sys.stderr)
            return 7

    if args.period_us > 0:
        period_us = int(args.period_us)
    elif args.fps > 0:
        period_us = int(round(1e6 / float(args.fps)))
    elif args.accum_us > 0:
        period_us = int(args.accum_us)
    else:
        # Reasonable default: 100 fps
        period_us = 10_000

    accum_us = int(args.accum_us) if args.accum_us > 0 else period_us

    if period_us <= 0 or accum_us <= 0:
        print("Invalid timing: period_us and accum_us must be > 0", file=sys.stderr)
        return 2

    if int(args.delta_t_us) <= 0:
        print("Invalid timing: delta_t_us must be > 0", file=sys.stderr)
        return 2

    start_us = int(args.start_us)
    delta_t_us = int(args.delta_t_us)
    proc_end_us = (start_us + int(args.duration_us)) if (args.duration_us and args.duration_us > 0) else None

    # OpenEB requires start_ts to be a multiple of delta_t.
    # Align down and filter by timestamp in software to preserve exact user windows.
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

    timestamps_csv = out_dir / "frames.csv"
    csv_f = timestamps_csv.open("w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["frame_idx", "t_start_us", "t_end_us", "filename"])

    # Active frame windows for overlap support.
    active = []
    created_frames = 0
    next_frame_start_us = start_us
    iterator_cursor_us = iterator_start_us

    print(f"Render mode: {args.render_mode}")
    if args.render_mode == "metavision_dark":
        palette_name = "Gray" if args.metavision_gray else "Dark"
        print(f"OpenEB palette: {palette_name}")

    def create_window(t_start_us: int):
        nonlocal created_frames
        t_end_us = t_start_us + accum_us
        if proc_end_us is not None and t_start_us >= proc_end_us:
            return False
        if proc_end_us is not None and t_end_us > proc_end_us:
            t_end_us = proc_end_us
        if t_end_us <= t_start_us:
            return False
        if args.max_frames and created_frames >= int(args.max_frames):
            return False
        active.append({
            "idx": created_frames,
            "start_us": int(t_start_us),
            "end_us": int(t_end_us),
            "acc_on": np.zeros((h, w), dtype=np.int32) if args.render_mode == "count_gray" else None,
            "acc_off": np.zeros((h, w), dtype=np.int32) if args.render_mode == "count_gray" else None,
            "events": [] if args.render_mode == "metavision_dark" else None,
        })
        created_frames += 1
        return True

    def flush_finished_windows(boundary_us: int):
        nonlocal active
        while active and active[0]["end_us"] <= boundary_us:
            win = active.pop(0)

            if args.render_mode == "count_gray":
                if args.polarity == "on":
                    signed = win["acc_on"]
                elif args.polarity == "off":
                    signed = -win["acc_off"]
                else:
                    signed = win["acc_on"] - win["acc_off"]
                img = window_image(np, signed, float(args.contrast), int(args.bit_depth))
            else:
                if win["events"]:
                    evs_window = np.concatenate(win["events"])
                else:
                    evs_window = np.empty((0,), dtype=[
                        ("x", np.uint16), ("y", np.uint16), ("p", np.uint8), ("t", np.int64)
                    ])
                img = np.zeros((h, w, 3), dtype=np.uint8)
                base_frame_algo.generate_frame(evs_window, img, accumulation_time_us=0, palette=mv_palette)

            if save_pil:
                ext = ".png" if args.to == "png" else ".tiff"
                out_name = f"evframe_{win['idx']:06d}{ext}"
            else:
                out_name = f"evframe_{win['idx']:06d}.pgm"

            out_path = out_dir / out_name
            if not save_pil and img.dtype != np.uint8:
                img = (img >> 8).astype(np.uint8)
            save_event_image(np, out_path, save_pil, pil_Image, img,
                            bit_depth=int(args.bit_depth), fmt=args.to)
            csv_w.writerow([win["idx"], win["start_us"], win["end_us"], out_name])

    try:
        for evs in it:
            slice_end_us = iterator_cursor_us + delta_t_us
            iterator_cursor_us = slice_end_us

            # Create all windows that start before this slice end.
            while next_frame_start_us < slice_end_us:
                if not create_window(next_frame_start_us):
                    break
                next_frame_start_us += period_us

            if evs.size and active:
                t_all = evs["t"].astype(np.int64, copy=False)
                x_all = evs["x"].astype(np.int64, copy=False)
                y_all = evs["y"].astype(np.int64, copy=False)
                p_all = evs["p"].astype(np.int8, copy=False)

                # Clip to requested global processing range.
                valid = t_all >= start_us
                if proc_end_us is not None:
                    valid &= (t_all < proc_end_us)

                if valid.any():
                    t_use = t_all[valid]
                    x_use = x_all[valid]
                    y_use = y_all[valid]
                    p_use = p_all[valid]

                    for win in active:
                        m = (t_use >= win["start_us"]) & (t_use < win["end_us"])
                        if not m.any():
                            continue
                        xs = x_use[m]
                        ys = y_use[m]
                        ps = p_use[m]
                        ts = t_use[m]

                        if args.render_mode == "count_gray":
                            if args.polarity in ("both", "on"):
                                mon = ps > 0
                                if mon.any():
                                    np.add.at(win["acc_on"], (ys[mon], xs[mon]), 1)
                            if args.polarity in ("both", "off"):
                                mof = ps == 0
                                if mof.any():
                                    np.add.at(win["acc_off"], (ys[mof], xs[mof]), 1)
                        else:
                            if args.polarity == "on":
                                keep = ps > 0
                            elif args.polarity == "off":
                                keep = ps == 0
                            else:
                                keep = np.ones(ps.shape, dtype=bool)
                            if keep.any():
                                sub = np.empty(keep.sum(), dtype=evs.dtype)
                                sub["x"] = xs[keep]
                                sub["y"] = ys[keep]
                                sub["p"] = ps[keep]
                                sub["t"] = ts[keep]
                                win["events"].append(sub)

            flush_finished_windows(slice_end_us if proc_end_us is None else min(slice_end_us, proc_end_us))

            if args.max_frames and created_frames >= int(args.max_frames) and not active:
                break

            if proc_end_us is not None and slice_end_us >= proc_end_us and not active:
                break

        # Final flush for windows that fully ended by the last processed timestamp.
        final_boundary = iterator_cursor_us if proc_end_us is None else min(iterator_cursor_us, proc_end_us)
        flush_finished_windows(final_boundary)

    finally:
        csv_f.close()

    print(f"Wrote {created_frames - len(active)} frames to {out_dir} (period_us={period_us}, accum_us={accum_us}, delta_t_us={delta_t_us})")
    if active:
        print(f"Note: dropped {len(active)} incomplete trailing window(s) at end-of-stream.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
