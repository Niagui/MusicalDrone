import unittest

from src import body_modulation as bm
from src import hierarchical_choreography as hc
from src import mind_planner as mp
from src import structure_features as sf


def repeated_phrase(start, end, mood, characteristic, valence, arousal, tension):
    return {
        "start": start,
        "end": end,
        "feature": {
            "moods": [
                {"label": mood, "score": 0.72},
                {"label": "atmospheric", "score": 0.18},
            ],
            "characteristics": [
                {"label": characteristic, "score": 0.61},
                {"label": "airy", "score": 0.19},
            ],
            "valence": [{"label": valence, "score": 0.81}],
            "arousal": [{"label": arousal, "score": 0.77}],
            "tension": [{"label": tension, "score": 0.74}],
        },
    }


class HierarchicalChoreographyTests(unittest.TestCase):
    def setUp(self):
        self.config = hc.load_hierarchical_config("config/hierarchical_config.json")
        self.repeated_segments = [
            repeated_phrase(0.0, 4.0, "sad", "sparse", "low valence", "low arousal", "moderate tension"),
            repeated_phrase(4.0, 8.0, "brave", "dense", "high valence", "high arousal", "high tension"),
            repeated_phrase(8.0, 12.0, "sad", "sparse", "low valence", "low arousal", "moderate tension"),
        ]
        self.beat_times = [
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

    def test_structure_features_track_repeat_family_and_decay(self):
        features = sf.build_structure_features(
            self.repeated_segments,
            beat_times=self.beat_times,
            sections=[],
            history_decay=0.8,
        )

        self.assertEqual(features[0].motif_family, features[2].motif_family)
        self.assertEqual(features[2].motif_repeat_index, 1)
        self.assertGreater(features[2].novelty, 0.0)
        self.assertNotEqual(
            features[2].current_emotion_vector,
            features[2].decayed_history_vector,
        )

    def test_macro_planner_changes_repeated_material_when_history_changes(self):
        features = sf.build_structure_features(
            self.repeated_segments,
            beat_times=self.beat_times,
            sections=[],
            history_decay=0.8,
        )
        macro_states = mp.plan_macro_states(features, config=self.config)

        self.assertEqual(macro_states[2].motif_mode, "vary")
        self.assertNotEqual(macro_states[0].z_base, macro_states[2].z_base)
        self.assertIn(macro_states[1].rotation_mode, {"none", "orbit", "swirl"})

    def test_body_planner_preserves_local_signature_for_repeat_family(self):
        features = sf.build_structure_features(
            self.repeated_segments,
            beat_times=self.beat_times,
            sections=[],
            history_decay=0.8,
        )
        macro_states = mp.plan_macro_states(features, config=self.config)
        body_states = bm.plan_body_states(features, macro_states, config=self.config)

        self.assertEqual(body_states[0].motif_seed, body_states[2].motif_seed)
        self.assertEqual(body_states[0].beat_pattern, body_states[2].beat_pattern)
        self.assertGreater(len(body_states[1].accent_events), 0)

    def test_generate_hierarchical_plan_payload_contains_nested_layers(self):
        payload = hc.generate_hierarchical_plan(
            clap_segments=self.repeated_segments,
            beat_times=self.beat_times,
            sections=[],
            anchor_weight_segments=[],
            output_path="/tmp/test_hierarchical_plan.json",
            config_path="config/hierarchical_config.json",
        )

        self.assertEqual(payload["planner"], "hierarchical_choreography")
        self.assertEqual(payload["total_phrases"], 3)
        phrase = payload["phrases"][0]
        self.assertIn("macro_state", phrase)
        self.assertIn("attractor_frame", phrase)
        self.assertIn("body_state", phrase)
        self.assertIn("translation_bias", phrase["macro_state"])
        self.assertIn("boid_deltas", phrase["body_state"])


if __name__ == "__main__":
    unittest.main()
