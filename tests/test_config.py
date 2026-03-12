"""Tests for utils/config.py — load_config()."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from utils.config import load_config, CONFIG_PATH


class TestLoadConfigReal(unittest.TestCase):
    """Tests that use the real config.json shipped with the project."""

    def test_loads_raw_to_frames_section(self):
        cfg = load_config("raw_to_frames")
        self.assertIsInstance(cfg, SimpleNamespace)
        # Check that all expected keys exist
        for key in ("input_raw", "out_dir", "fps", "accum_us", "period_us",
                     "delta_t_us", "max_frames", "start_us", "duration_us",
                     "polarity", "render_mode", "metavision_gray", "to",
                     "bit_depth", "contrast"):
            self.assertTrue(hasattr(cfg, key), f"Missing key: {key}")

    def test_loads_video_processing_section(self):
        cfg = load_config("video_processing")
        self.assertIsInstance(cfg, SimpleNamespace)
        for key in ("video_path", "raw_path", "output_path",
                     "output_fps", "blend_alpha", "grayscale"):
            self.assertTrue(hasattr(cfg, key), f"Missing key: {key}")

    def test_loads_video_to_frames_section(self):
        cfg = load_config("video_to_frames")
        self.assertIsInstance(cfg, SimpleNamespace)
        for key in ("input_video", "out_dir", "format", "grayscale",
                     "max_frames", "start_frame"):
            self.assertTrue(hasattr(cfg, key), f"Missing key: {key}")

    def test_loads_model_section(self):
        cfg = load_config("model")
        self.assertIsInstance(cfg, SimpleNamespace)
        for key in ("num_bins", "image_channels", "base_channels", "unet_depth",
                     "learning_rate", "batch_size", "num_epochs", "loss",
                     "checkpoint_dir", "resume_checkpoint"):
            self.assertTrue(hasattr(cfg, key), f"Missing key: {key}")

    def test_missing_section_raises(self):
        with self.assertRaises(KeyError):
            load_config("nonexistent_section_xyz")

    def test_attribute_access_values(self):
        cfg = load_config("raw_to_frames")
        self.assertIsInstance(cfg.fps, (int, float))
        self.assertIsInstance(cfg.bit_depth, int)
        self.assertIsInstance(cfg.polarity, str)
        self.assertIn(cfg.polarity, ("both", "on", "off"))

    def test_video_processing_defaults(self):
        cfg = load_config("video_processing")
        self.assertIsInstance(cfg.output_fps, (int, float))
        self.assertGreater(cfg.output_fps, 0)
        self.assertIsInstance(cfg.blend_alpha, (int, float))
        self.assertGreaterEqual(cfg.blend_alpha, 0.0)
        self.assertLessEqual(cfg.blend_alpha, 1.0)


class TestLoadConfigIsolated(unittest.TestCase):
    """Tests using a temporary config.json to avoid depending on real file content."""

    def setUp(self):
        self._orig_path = __import__("utils.config", fromlist=["config"]).CONFIG_PATH
        self._tmpdir = tempfile.TemporaryDirectory()
        self._tmp_config = Path(self._tmpdir.name) / "config.json"

    def tearDown(self):
        # Restore original CONFIG_PATH
        import utils.config as cfg_mod
        cfg_mod.CONFIG_PATH = self._orig_path
        self._tmpdir.cleanup()

    def _write_and_patch(self, data: dict):
        self._tmp_config.write_text(json.dumps(data), encoding="utf-8")
        import utils.config as cfg_mod
        cfg_mod.CONFIG_PATH = self._tmp_config

    def test_simple_section(self):
        self._write_and_patch({"my_section": {"key1": 42, "key2": "hello"}})
        cfg = load_config("my_section")
        self.assertEqual(cfg.key1, 42)
        self.assertEqual(cfg.key2, "hello")

    def test_bool_values(self):
        self._write_and_patch({"s": {"flag": True}})
        cfg = load_config("s")
        self.assertIs(cfg.flag, True)

    def test_empty_section(self):
        self._write_and_patch({"empty": {}})
        cfg = load_config("empty")
        self.assertIsInstance(cfg, SimpleNamespace)


if __name__ == "__main__":
    unittest.main()
