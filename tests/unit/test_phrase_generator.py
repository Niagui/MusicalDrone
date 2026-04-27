import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import phrase_generator as pg


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "clap_results_sample.json"


def load_fixture():
    with open(FIXTURE_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def make_valid_model_record(phrase_index: int):
    return {
        "phrase_index": phrase_index,
        "section_role": "buildup",
        "motion_mode": "advance",
        "height_level": 0.7,
        "depth_level": 0.55,
        "speed_level": 0.65,
        "vertical_trend": "rise",
        "transition_style": "surge",
    }


class PhraseGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_phrases = load_fixture()
        cls.sample_beats = [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            5.5,
            6.0,
            6.5,
            7.0,
            7.5,
            8.0,
            8.5,
            9.0,
            9.5,
            10.0,
            10.5,
            11.0,
            11.5,
            12.0,
        ]

    def test_phrase_block_contains_compact_summary_and_derived_beat_count(self):
        sections = [pg.SectionSpan(index=0, start=0.0, end=10.0)]
        block = pg.phrase_to_block(
            self.sample_phrases[0],
            0,
            beat_times=self.sample_beats,
            sections=sections,
        )

        self.assertEqual(block.section_index, 0)
        self.assertEqual(block.beat_count, 8)
        self.assertIn("phrase_index: 0", block.prompt_text)
        self.assertIn("moods: sad, atmospheric", block.prompt_text)
        self.assertIn("arousal: low arousal", block.prompt_text)

    def test_strip_code_fences_handles_json_fences(self):
        payload = {"phrases": [make_valid_model_record(0)]}
        raw = json.dumps(payload)
        fenced = f"```json\n{raw}\n```"

        self.assertEqual(pg.strip_code_fences(fenced), raw)
        self.assertEqual(json.loads(pg.strip_code_fences(fenced)), payload)

    def test_normalize_plan_records_fills_defaults_for_invalid_values(self):
        blocks = pg.build_phrase_blocks(
            self.sample_phrases[:2], beat_times=self.sample_beats, sections=[]
        )
        obj = {
            "phrases": [
                make_valid_model_record(0),
                {
                    "phrase_index": 1,
                    "section_role": "not-real",
                    "motion_mode": "bad",
                    "height_level": 9,
                    "depth_level": "oops",
                    "speed_level": -1,
                    "vertical_trend": "bad",
                    "transition_style": "bad",
                },
            ]
        }

        normalized = pg.normalize_plan_records(obj, blocks)

        self.assertEqual(normalized[0]["motion_mode"], "advance")
        self.assertEqual(normalized[1]["section_role"], "stable")
        self.assertEqual(normalized[1]["height_level"], 1.0)
        self.assertEqual(normalized[1]["depth_level"], 0.5)
        self.assertEqual(normalized[1]["speed_level"], 0.0)

    def test_build_beat_plan_is_heuristic_and_not_model_supplied(self):
        block = pg.PhraseBlock(
            phrase_index=0,
            start=0.0,
            end=4.0,
            beat_count=8,
            prompt_text="demo",
        )
        plan = make_valid_model_record(0)

        beat_plan = pg.build_beat_plan(block, plan)

        self.assertEqual(beat_plan[0], {"beat": 1, "action": "hold"})
        self.assertIn({"beat": 6, "action": "advance"}, beat_plan)
        self.assertIn({"beat": 8, "action": "settle"}, beat_plan)

    def test_pack_batches_preserves_phrase_order(self):
        blocks = pg.build_phrase_blocks(
            self.sample_phrases, beat_times=self.sample_beats, sections=[]
        )
        batches = pg.pack_batches(blocks, max_chars=120)
        flattened = [block.phrase_index for batch in batches for block in batch]

        self.assertEqual(flattened, [0, 1, 2])
        self.assertGreaterEqual(len(batches), 2)

    def test_build_output_payload_keeps_one_phrase_per_input_phrase(self):
        blocks = pg.build_phrase_blocks(
            self.sample_phrases, beat_times=self.sample_beats, sections=[]
        )
        normalized = {
            block.phrase_index: pg.default_phrase_plan(block) for block in blocks
        }

        payload = pg.build_output_payload(
            blocks,
            normalized,
            source_file="json/clap_results.json",
            model="test-model",
        )

        self.assertEqual(payload["total_phrases"], len(blocks))
        self.assertEqual(payload["planner"], "single_attractor_phrase_planner")
        self.assertEqual(
            [item["phrase_index"] for item in payload["phrases"]],
            [0, 1, 2],
        )
        self.assertIn("beat_plan", payload["phrases"][0])

    def test_generated_json_path_uses_cache_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "json-cache"
            with mock.patch.dict(os.environ, {"DRONE_JSON_DIR": str(cache_dir)}):
                self.assertEqual(
                    pg.get_generated_json_path("phrase_plan"),
                    cache_dir / "phrase_plan.json",
                )

    def test_pipeline_json_path_falls_back_to_repo_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "json-cache"
            with mock.patch.dict(os.environ, {"DRONE_JSON_DIR": str(cache_dir)}):
                self.assertEqual(
                    pg.get_pipeline_json_path("sections"),
                    pg.PROJECT_ROOT / "json" / "sections.json",
                )


if __name__ == "__main__":
    unittest.main()
