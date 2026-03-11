"""Tests for datasets/video_processing.py.

Tests are split into:
- Pure-numpy functions (accumulate_events, merge_frame_and_events) — always run
- Video I/O functions (get_video_metadata, iter_video_frames, get_video_frame) — need a small
  synthetic AVI written via cv2.VideoWriter, created in setUpClass
- Event / Metavision functions — skipped if metavision_core is not installed
"""

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

# Ensure project packages are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "datasets"))
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

from video_processing import (
    accumulate_events,
    get_video_frame,
    get_video_metadata,
    iter_video_frames,
    merge_frame_and_events,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_events(xs, ys, ps, ts):
    """Build a structured numpy array matching the Metavision event dtype."""
    dt = np.dtype([("x", np.uint16), ("y", np.uint16),
                   ("p", np.uint8), ("t", np.int64)])
    n = len(xs)
    evs = np.empty(n, dtype=dt)
    evs["x"] = np.asarray(xs, dtype=np.uint16)
    evs["y"] = np.asarray(ys, dtype=np.uint16)
    evs["p"] = np.asarray(ps, dtype=np.uint8)
    evs["t"] = np.asarray(ts, dtype=np.int64)
    return evs


# ---------------------------------------------------------------------------
# accumulate_events tests
# ---------------------------------------------------------------------------

class TestAccumulateEvents(unittest.TestCase):

    def test_binary_mode_marks_event_pixels(self):
        evs = _make_events([2, 5], [1, 3], [1, 0], [100, 200])
        img = accumulate_events(evs, height=10, width=10, mode="binary")
        self.assertEqual(img.shape, (10, 10))
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(img[1, 2], 255)
        self.assertEqual(img[3, 5], 255)
        self.assertEqual(img[0, 0], 0)

    def test_binary_mode_empty(self):
        evs = _make_events([], [], [], [])
        img = accumulate_events(evs, height=4, width=4, mode="binary")
        self.assertTrue(np.all(img == 0))

    def test_count_mode_baseline_128(self):
        """With no events the count image should be uniform 128."""
        evs = _make_events([], [], [], [])
        img = accumulate_events(evs, height=4, width=4, mode="count")
        self.assertTrue(np.all(img == 128))

    def test_count_mode_on_events_increase(self):
        evs = _make_events([0], [0], [1], [100])
        img = accumulate_events(evs, height=2, width=2, mode="count")
        # pixel (0,0) should be > 128
        self.assertGreater(int(img[0, 0]), 128)

    def test_count_mode_off_events_decrease(self):
        evs = _make_events([0], [0], [0], [100])
        img = accumulate_events(evs, height=2, width=2, mode="count")
        # pixel (0,0) should be < 128
        self.assertLess(int(img[0, 0]), 128)

    def test_count_mode_on_off_cancel(self):
        evs = _make_events([0, 0], [0, 0], [1, 0], [100, 200])
        img = accumulate_events(evs, height=2, width=2, mode="count")
        self.assertEqual(int(img[0, 0]), 128)

    def test_output_is_uint8(self):
        evs = _make_events([0] * 100, [0] * 100, [1] * 100, list(range(100)))
        img = accumulate_events(evs, height=2, width=2, mode="count")
        self.assertEqual(img.dtype, np.uint8)
        self.assertTrue(np.all(img <= 255))


# ---------------------------------------------------------------------------
# merge_frame_and_events tests
# ---------------------------------------------------------------------------

class TestMergeFrameAndEvents(unittest.TestCase):

    def test_same_shape_blend(self):
        base = np.full((4, 4, 3), 200, dtype=np.uint8)
        ev = np.full((4, 4, 3), 100, dtype=np.uint8)
        merged = merge_frame_and_events(base, ev, alpha=0.5)
        self.assertEqual(merged.shape, (4, 4, 3))
        self.assertEqual(merged.dtype, np.uint8)
        # Expected: 0.5 * 200 + 0.5 * 100 = 150
        np.testing.assert_array_almost_equal(merged, 150, decimal=0)

    def test_alpha_zero_returns_base(self):
        base = np.full((4, 4, 3), 50, dtype=np.uint8)
        ev = np.full((4, 4, 3), 250, dtype=np.uint8)
        merged = merge_frame_and_events(base, ev, alpha=0.0)
        np.testing.assert_array_equal(merged, base)

    def test_alpha_one_returns_event(self):
        base = np.full((4, 4, 3), 50, dtype=np.uint8)
        ev = np.full((4, 4, 3), 250, dtype=np.uint8)
        merged = merge_frame_and_events(base, ev, alpha=1.0)
        np.testing.assert_array_equal(merged, ev)

    def test_gray_event_upconverted_to_bgr(self):
        base = np.full((4, 4, 3), 100, dtype=np.uint8)
        ev_gray = np.full((4, 4), 200, dtype=np.uint8)
        merged = merge_frame_and_events(base, ev_gray, alpha=0.5)
        self.assertEqual(merged.ndim, 3)
        self.assertEqual(merged.shape[2], 3)

    def test_gray_base_upconverted_to_bgr(self):
        base_gray = np.full((4, 4), 100, dtype=np.uint8)
        ev = np.full((4, 4, 3), 200, dtype=np.uint8)
        merged = merge_frame_and_events(base_gray, ev, alpha=0.5)
        self.assertEqual(merged.ndim, 3)

    def test_output_clipped_to_uint8(self):
        base = np.full((2, 2, 3), 255, dtype=np.uint8)
        ev = np.full((2, 2, 3), 255, dtype=np.uint8)
        merged = merge_frame_and_events(base, ev, alpha=0.5)
        self.assertTrue(np.all(merged <= 255))


# ---------------------------------------------------------------------------
# Video I/O tests (use a small synthetic AVI)
# ---------------------------------------------------------------------------

class TestVideoIO(unittest.TestCase):

    _tmpdir = None
    _video_path = None
    _num_frames = 10
    _width = 16
    _height = 12
    _fps = 25.0

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._video_path = Path(cls._tmpdir.name) / "test.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(cls._video_path), fourcc,
                                 cls._fps, (cls._width, cls._height))
        for i in range(cls._num_frames):
            # Each frame has a unique grey value so we can identify it
            val = int(20 + i * 20) % 256
            frame = np.full((cls._height, cls._width, 3), val, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    # -- get_video_metadata --

    def test_metadata_keys(self):
        meta = get_video_metadata(self._video_path)
        for k in ("fps", "frame_count", "width", "height", "duration_s", "codec"):
            self.assertIn(k, meta)

    def test_metadata_values(self):
        meta = get_video_metadata(self._video_path)
        self.assertEqual(meta["width"], self._width)
        self.assertEqual(meta["height"], self._height)
        self.assertEqual(meta["frame_count"], self._num_frames)
        self.assertAlmostEqual(meta["fps"], self._fps, places=1)
        self.assertGreater(meta["duration_s"], 0)

    def test_metadata_bad_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            get_video_metadata("nonexistent_video.avi")

    # -- iter_video_frames --

    def test_iter_yields_all_frames(self):
        frames = list(iter_video_frames(self._video_path))
        self.assertEqual(len(frames), self._num_frames)
        for idx, frame in frames:
            self.assertEqual(frame.shape[:2], (self._height, self._width))

    def test_iter_indices_sequential(self):
        indices = [idx for idx, _ in iter_video_frames(self._video_path)]
        self.assertEqual(indices, list(range(self._num_frames)))

    def test_iter_grayscale(self):
        for _, frame in iter_video_frames(self._video_path, grayscale=True):
            self.assertEqual(frame.ndim, 2)

    def test_iter_bad_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            list(iter_video_frames("nonexistent_video.avi"))

    # -- get_video_frame --

    def test_get_frame_returns_correct_shape(self):
        frame = get_video_frame(self._video_path, 0)
        self.assertEqual(frame.shape, (self._height, self._width, 3))

    def test_get_frame_grayscale(self):
        frame = get_video_frame(self._video_path, 0, grayscale=True)
        self.assertEqual(frame.ndim, 2)

    def test_get_frame_bad_index_raises(self):
        with self.assertRaises(IndexError):
            get_video_frame(self._video_path, 99999)

    def test_get_frame_bad_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            get_video_frame("nonexistent_video.avi", 0)


# ---------------------------------------------------------------------------
# Event iterator — skip if metavision_core unavailable
# ---------------------------------------------------------------------------

try:
    from metavision_core.event_io import EventsIterator as _EI  # noqa: F401
    _HAS_METAVISION = True
except ImportError:
    _HAS_METAVISION = False


@unittest.skipUnless(_HAS_METAVISION, "metavision_core not available")
class TestIterEventFrames(unittest.TestCase):

    def test_import_works(self):
        from video_processing import iter_event_frames  # noqa: F401
        self.assertTrue(callable(iter_event_frames))


if __name__ == "__main__":
    unittest.main()
