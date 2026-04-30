import json
import tempfile
import unittest
from pathlib import Path

from src import descriptor_anchor_mapper as dam


class DescriptorAnchorMapperTests(unittest.TestCase):
    def test_normalize_motion_mix_validates_and_normalizes(self):
        mix = dam.normalize_motion_mix({"sad": 2, "sleepy": 1})

        self.assertAlmostEqual(mix["sad"], 2 / 3)
        self.assertAlmostEqual(mix["sleepy"], 1 / 3)
        self.assertAlmostEqual(sum(mix.values()), 1.0)

        with self.assertRaises(ValueError):
            dam.normalize_motion_mix({"angry": 1.0})

        with self.assertRaises(ValueError):
            dam.normalize_motion_mix({"sad": -1.0})

    def test_resolve_anchor_records_uses_deterministic_fallbacks(self):
        config = {
            "version": 1,
            "smoothing_alpha": 0.25,
            "anchors": [
                {
                    "name": "fragile",
                    "description": "quiet, delicate, uncertain",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "anchors.json"
            path.write_text(json.dumps(config), encoding="utf-8")

            anchors, metadata = dam.resolve_anchor_records(
                path,
                clap=None,
                use_llm=False,
            )

        self.assertEqual(metadata["smoothing_alpha"], 0.25)
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0].name, "fragile")
        self.assertTrue(anchors[0].prompts)
        self.assertGreater(anchors[0].motion_mix["shy"], 0.0)
        self.assertGreater(anchors[0].motion_mix["sleepy"], 0.0)
        self.assertAlmostEqual(sum(anchors[0].motion_mix.values()), 1.0)

    def test_smooth_weight_rows_blends_previous_segment(self):
        rows = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]

        smoothed = dam.smooth_weight_rows(rows, alpha=0.25)

        self.assertAlmostEqual(smoothed[0][0], 1.0)
        self.assertAlmostEqual(smoothed[1][0], 0.25)
        self.assertAlmostEqual(smoothed[1][1], 0.75)
        self.assertAlmostEqual(sum(smoothed[1]), 1.0)

    def test_claire_de_lune_like_scores_do_not_become_grumpy(self):
        anchors = [
            dam.ResolvedAnchor(
                name="sad",
                description="",
                prompts=["melancholic soft music"],
                motion_mix=dam.normalize_motion_mix({"sad": 1.0}),
            ),
            dam.ResolvedAnchor(
                name="sleepy",
                description="",
                prompts=["slow calm dreamy music"],
                motion_mix=dam.normalize_motion_mix({"sleepy": 1.0}),
            ),
            dam.ResolvedAnchor(
                name="shy",
                description="",
                prompts=["quiet delicate music"],
                motion_mix=dam.normalize_motion_mix({"shy": 1.0}),
            ),
            dam.ResolvedAnchor(
                name="grumpy",
                description="",
                prompts=["harsh irritated music"],
                motion_mix=dam.normalize_motion_mix({"grumpy": 1.0}),
            ),
        ]
        prompt_scores = {
            "melancholic soft music": 0.42,
            "slow calm dreamy music": 0.35,
            "quiet delicate music": 0.18,
            "harsh irritated music": 0.05,
        }

        anchor_scores = dam.aggregate_anchor_scores(prompt_scores, anchors)
        weights = dam.motion_weights_from_anchor_scores(anchor_scores, anchors)

        self.assertGreater(weights[dam.MOTION_ANCHORS.index("sad")], 0.0)
        self.assertGreater(
            weights[dam.MOTION_ANCHORS.index("sad")],
            weights[dam.MOTION_ANCHORS.index("grumpy")],
        )
        self.assertGreater(
            weights[dam.MOTION_ANCHORS.index("sleepy")],
            weights[dam.MOTION_ANCHORS.index("grumpy")],
        )
        self.assertAlmostEqual(sum(weights), 1.0)


if __name__ == "__main__":
    unittest.main()
