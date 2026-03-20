"""Extract individual frames from a video file (AVI, MP4, etc.) and save as images.

Reads settings from config.json section "video_to_frames".
Outputs numbered image files (PNG or PGM) and a frames.csv with timestamps.

Example:
    python utils/video_to_frames.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2

from utils.config import load_config


def main() -> int:
    cfg = load_config("video_processing")

    video_path = Path(cfg.video_path)
    if not video_path.is_file():
        print(f"Input not found: {video_path}", file=sys.stderr)
        return 2

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grayscale = bool(cfg.grayscale)
    fmt = cfg.format  # "png" or "pgm"
    max_frames = int(cfg.max_frames)
    start_frame = int(cfg.start_frame)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}", file=sys.stderr)
        return 3

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    timestamp_offset_us = float(getattr(cfg, "timestamp_offset_us", 0.0))

    print(f"Video: {video_path}")
    print(f"  Resolution: {width}x{height}, FPS: {fps:.2f}, Total frames: {total}")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    timestamps_csv = out_dir / "frames.csv"
    csv_f = timestamps_csv.open("w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["frame_idx", "timestamp_s", "filename"])

    saved = 0
    frame_idx = start_frame

    try:
        while True:
            if max_frames > 0 and saved >= max_frames:
                break

            ok, frame = cap.read()
            if not ok:
                break

            if grayscale and frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if fmt == "pgm":
                if frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                out_name = f"frame_{saved:06d}.pgm"
                out_path = out_dir / out_name
                h, w = frame.shape[:2]
                with out_path.open("wb") as f:
                    f.write(b"P5\n")
                    f.write(f"{w} {h}\n255\n".encode("ascii"))
                    f.write(frame.tobytes(order="C"))
            else:
                out_name = f"frame_{saved:06d}.png"
                out_path = out_dir / out_name
                cv2.imwrite(str(out_path), frame)

            timestamp_s = (frame_idx / fps * 1e6 + timestamp_offset_us) / 1e6 if fps > 0 else 0.0
            csv_w.writerow([frame_idx, f"{timestamp_s:.9f}", out_name])
            saved += 1
            frame_idx += 1
    finally:
        csv_f.close()
        cap.release()

    print(f"Saved {saved} frames to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
