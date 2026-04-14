import copy
import json
import unittest
from unittest import mock

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
        self.neutral_config = copy.deepcopy(self.config)
        self.neutral_config["mind"]["planner_mode"] = "neutral"
        self.neutral_config["mind"]["fallback_to_neutral"] = True
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
        self.assertGreater(features[1].normalized_change, features[0].normalized_change)
        self.assertGreater(features[2].normalized_change, 0.0)
        self.assertNotEqual(
            features[2].current_emotion_vector,
            features[2].decayed_history_vector,
        )
        self.assertNotEqual(
            features[1].difference_vector,
            features[1].normalized_difference_vector,
        )

    def test_neutral_macro_planner_does_not_infer_motion_from_features(self):
        features = sf.build_structure_features(
            self.repeated_segments,
            beat_times=self.beat_times,
            sections=[],
            history_decay=0.8,
        )
        macro_plan = mp.plan_macro_states(features, config=self.neutral_config)
        macro_states = macro_plan.states

        self.assertEqual(macro_plan.planner_mode, "neutral")
        for macro in macro_states:
            self.assertEqual(macro.z_base, 0.5)
            self.assertEqual(macro.audience_bias, 0.5)
            self.assertEqual(macro.radius_base, 0.5)
            self.assertEqual(macro.rotation_mode, "none")
            self.assertEqual(macro.persistence, 0.5)

    def test_body_planner_preserves_local_signature_for_repeat_family(self):
        features = sf.build_structure_features(
            self.repeated_segments,
            beat_times=self.beat_times,
            sections=[],
            history_decay=0.8,
        )
        macro_states = mp.plan_macro_states(features, config=self.neutral_config).states
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
            macro_planner_mode="neutral",
        )

        self.assertEqual(payload["planner"], "hierarchical_choreography")
        self.assertEqual(payload["total_phrases"], 3)
        phrase = payload["phrases"][0]
        self.assertIn("macro_state", phrase)
        self.assertIn("attractor_frame", phrase)
        self.assertIn("body_state", phrase)
        self.assertIn("translation_bias", phrase["macro_state"])
        self.assertIn("boid_deltas", phrase["body_state"])
        self.assertEqual(payload["macro_planner"]["mode"], "neutral")

    def test_llm_macro_planner_is_optional_and_uses_change_features(self):
        features = sf.build_structure_features(
            self.repeated_segments,
            beat_times=self.beat_times,
            sections=[],
            history_decay=0.8,
        )
        llm_payload = {
            "phrases": [
                {
                    "phrase_index": 0,
                    "z_base": 0.4,
                    "audience_bias": 0.45,
                    "radius_base": 0.35,
                    "rotation_mode": "none",
                    "rotation_bias": 0.0,
                    "translation_bias_x": 0.0,
                    "translation_bias_y": -0.1,
                    "persistence": 0.85,
                },
                {
                    "phrase_index": 1,
                    "z_base": 0.82,
                    "audience_bias": 0.7,
                    "radius_base": 0.75,
                    "rotation_mode": "orbit",
                    "rotation_bias": 0.55,
                    "translation_bias_x": 0.2,
                    "translation_bias_y": 0.45,
                    "persistence": 0.4,
                },
                {
                    "phrase_index": 2,
                    "z_base": 0.55,
                    "audience_bias": 0.35,
                    "radius_base": 0.4,
                    "rotation_mode": "none",
                    "rotation_bias": 0.0,
                    "translation_bias_x": 0.1,
                    "translation_bias_y": -0.25,
                    "persistence": 0.75,
                },
            ]
        }

        with mock.patch.object(mp, "OPENAI_API_KEY", "test-key"), mock.patch.object(
            mp, "create_chat_completion", return_value=json.dumps(llm_payload)
        ) as mocked_completion:
            macro_plan = mp.plan_macro_states(features, config=self.config)

        self.assertEqual(macro_plan.planner_mode, "llm")
        self.assertEqual(macro_plan.states[1].rotation_mode, "orbit")
        self.assertEqual(macro_plan.states[1].z_base, 0.82)
        self.assertEqual(macro_plan.states[1].persistence, 0.4)

        request_kwargs = mocked_completion.call_args.args[0]
        prompt = request_kwargs["messages"][1]["content"]
        self.assertIn("decayed_history", prompt)
        self.assertIn("delta:", prompt)
        self.assertIn("normalized_change", prompt)
        self.assertNotIn("motif_signature", prompt)
        self.assertNotIn("section_role", prompt)
        self.assertNotIn("strength", prompt)

    def test_llm_macro_planner_splits_batches_after_parse_failure(self):
        features = sf.build_structure_features(
            self.repeated_segments[:2],
            beat_times=self.beat_times[:17],
            sections=[],
            history_decay=0.8,
        )
        batched_config = copy.deepcopy(self.config)
        batched_config["mind"]["max_batch_phrases"] = 8

        def fake_completion(request_kwargs):
            prompt = request_kwargs["messages"][1]["content"]
            phrase_count = prompt.count("phrase_index:")
            if phrase_count > 1:
                return '{"phrases": [invalid]}'

            phrase_index_line = next(
                line for line in prompt.splitlines() if line.startswith("phrase_index:")
            )
            phrase_index = int(phrase_index_line.split(":", 1)[1].strip())
            payload = {
                "phrases": [
                    {
                        "phrase_index": phrase_index,
                        "z_base": 0.4 + 0.2 * phrase_index,
                        "audience_bias": 0.5,
                        "radius_base": 0.45,
                        "rotation_mode": "none",
                        "rotation_bias": 0.0,
                        "translation_bias_x": 0.0,
                        "translation_bias_y": 0.1 * phrase_index,
                        "persistence": 0.65,
                    }
                ]
            }
            return json.dumps(payload)

        with mock.patch.object(mp, "OPENAI_API_KEY", "test-key"), mock.patch.object(
            mp, "create_chat_completion", side_effect=fake_completion
        ):
            macro_plan = mp.plan_macro_states(features, config=batched_config)

        self.assertEqual(macro_plan.planner_mode, "llm")
        self.assertAlmostEqual(macro_plan.states[0].z_base, 0.4)
        self.assertAlmostEqual(macro_plan.states[1].z_base, 0.6)
        self.assertAlmostEqual(macro_plan.states[1].persistence, 0.65)


if __name__ == "__main__":
    unittest.main()
