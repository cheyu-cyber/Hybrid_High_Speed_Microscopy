"""Tests for srcs/video_to_frames.py."""

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestVideoToFrames(unittest.TestCase):
    """End-to-end tests using a small synthetic AVI."""

    _tmpdir = None
    _video_path = None
    _num_frames = 5
    _width = 16
    _height = 12
    _fps = 25.0

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._video_path = Path(cls._tmpdir.name) / "input.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(cls._video_path), fourcc,
                                 cls._fps, (cls._width, cls._height))
        for i in range(cls._num_frames):
            val = int(40 + i * 40) % 256
            frame = np.full((cls._height, cls._width, 3), val, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def _run_main(self, extra_cfg=None):
        """Run video_to_frames.main() with a patched config."""
        import json
        import utils.config as cfg_mod

        out_dir = Path(tempfile.mkdtemp(dir=self._tmpdir.name))
        cfg_data = {
            "video_processing": {
                "video_path": str(self._video_path),
                "out_dir": str(out_dir),
                "format": "png",
                "grayscale": False,
                "max_frames": 0,
                "start_frame": 0,
            }
        }
        if extra_cfg:
            cfg_data["video_processing"].update(extra_cfg)

        tmp_cfg = Path(self._tmpdir.name) / "config.json"
        tmp_cfg.write_text(json.dumps(cfg_data), encoding="utf-8")

        orig = cfg_mod.CONFIG_PATH
        cfg_mod.CONFIG_PATH = tmp_cfg
        try:
            from srcs.video_to_frames import main
            ret = main()
        finally:
            cfg_mod.CONFIG_PATH = orig
        return ret, out_dir

    # -- Tests --

    def test_extracts_all_frames_png(self):
        ret, out_dir = self._run_main()
        self.assertEqual(ret, 0)
        pngs = sorted(out_dir.glob("frame_*.png"))
        self.assertEqual(len(pngs), self._num_frames)

    def test_csv_created(self):
        ret, out_dir = self._run_main()
        csv_path = out_dir / "frames.csv"
        self.assertTrue(csv_path.exists())
        with csv_path.open() as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, ["frame_idx", "timestamp_s", "filename"])
            rows = list(reader)
        self.assertEqual(len(rows), self._num_frames)

    def test_max_frames_limits_output(self):
        ret, out_dir = self._run_main({"max_frames": 2})
        self.assertEqual(ret, 0)
        pngs = sorted(out_dir.glob("frame_*.png"))
        self.assertEqual(len(pngs), 2)

    def test_grayscale_output(self):
        ret, out_dir = self._run_main({"grayscale": True})
        self.assertEqual(ret, 0)
        img = cv2.imread(str(next(out_dir.glob("frame_*.png"))), cv2.IMREAD_UNCHANGED)
        self.assertEqual(img.ndim, 2)

    def test_pgm_format(self):
        ret, out_dir = self._run_main({"format": "pgm"})
        self.assertEqual(ret, 0)
        pgms = sorted(out_dir.glob("frame_*.pgm"))
        self.assertEqual(len(pgms), self._num_frames)
        # Verify PGM header
        with pgms[0].open("rb") as f:
            magic = f.readline()
            self.assertEqual(magic.strip(), b"P5")

    def test_start_frame_offset(self):
        ret, out_dir = self._run_main({"start_frame": 2})
        self.assertEqual(ret, 0)
        pngs = sorted(out_dir.glob("frame_*.png"))
        self.assertEqual(len(pngs), self._num_frames - 2)

    def test_bad_input_returns_error(self):
        import json
        import utils.config as cfg_mod

        cfg_data = {
            "video_processing": {
                "video_path": "nonexistent.avi",
                "out_dir": str(Path(self._tmpdir.name) / "bad"),
                "format": "png",
                "grayscale": False,
                "max_frames": 0,
                "start_frame": 0,
            }
        }
        tmp_cfg = Path(self._tmpdir.name) / "config_bad.json"
        tmp_cfg.write_text(json.dumps(cfg_data), encoding="utf-8")

        orig = cfg_mod.CONFIG_PATH
        cfg_mod.CONFIG_PATH = tmp_cfg
        try:
            from srcs.video_to_frames import main
            ret = main()
        finally:
            cfg_mod.CONFIG_PATH = orig
        self.assertNotEqual(ret, 0)


if __name__ == "__main__":
    unittest.main()
