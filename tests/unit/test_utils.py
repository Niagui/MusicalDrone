import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

sys.modules.setdefault("soundfile", types.SimpleNamespace(info=lambda *_a, **_k: None))
sys.modules.setdefault("logger_config", types.SimpleNamespace(log_info=lambda *_a, **_k: None))
sys.modules.setdefault(
    "torch",
    types.SimpleNamespace(
        backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(enabled=False)),
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
            get_device_name=lambda *_a, **_k: "",
        ),
    ),
)

import utils


class UtilsPathTests(unittest.TestCase):
    def test_save_as_json_uses_cache_dir_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache-json"
            with mock.patch.dict(os.environ, {"DRONE_JSON_DIR": str(cache_dir)}):
                utils.save_as_json("demo", {"value": 1})

            output_path = cache_dir / "demo.json"
            self.assertTrue(output_path.exists())
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                {"value": 1},
            )

    def test_get_pipeline_json_path_prefers_cached_generated_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache-json"
            cache_dir.mkdir()
            cached_path = cache_dir / "k_beat_segments.json"
            cached_path.write_text("[]", encoding="utf-8")

            with mock.patch.dict(os.environ, {"DRONE_JSON_DIR": str(cache_dir)}):
                resolved = utils.get_pipeline_json_path("k_beat_segments")

            self.assertEqual(resolved, cached_path)

    def test_get_pipeline_json_path_falls_back_to_shared_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_cache_dir = Path(temp_dir) / "missing-cache-json"
            with mock.patch.dict(
                os.environ, {"DRONE_JSON_DIR": str(missing_cache_dir)}
            ):
                resolved = utils.get_pipeline_json_path("anchor_labels")

            self.assertEqual(resolved, utils.get_shared_json_path("anchor_labels"))


if __name__ == "__main__":
    unittest.main()
