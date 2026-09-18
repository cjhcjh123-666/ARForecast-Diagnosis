"""Regression tests for the ICLR deadline protocol audit.

These tests encode measurement invariants.  They deliberately avoid model loading so they can be
run before any expensive experiment is restarted.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


class FactorizedQAProtocolTests(unittest.TestCase):
    def test_qwen_base_alias_rejects_qwen3ts_checkpoint(self):
        from scripts.extract_qa_reps import validate_model_identity

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps({
                "model_type": "qwen3ts", "architectures": ["Qwen3TSForCausalLM"]
            }))
            with self.assertRaises(ValueError):
                validate_model_identity("qwen3_8b_base", path)

    def test_task_filter_keeps_ids_features_and_gold_aligned(self):
        from scripts.run_factorized_qa import select_finite_items

        ids = np.array([10, 11, 12, 13])
        gold = np.array(["A", "D", "B", "C"])
        task_type = np.array(["keep", "drop", "keep", "drop"])
        items = {
            i: {"id": i, "answer_format": "multiple_choice_abcd"} for i in ids
        }
        keep, y = select_finite_items(ids, gold, task_type, items, {"keep"})
        np.testing.assert_array_equal(keep, [0, 2])
        np.testing.assert_array_equal(y, [0, 1])

    def test_question_branch_is_fixed_to_pretrained_cache(self):
        from scripts.run_factorized_qa import factorized_inputs

        pre = {"h_q": np.full((3, 2), 1.0), "h_x_full": np.full((3, 2), 2.0)}
        rnd = {"h_q": np.full((3, 2), 9.0), "h_x_full": np.full((3, 2), 3.0)}
        q_pre, x_pre = factorized_inputs(pre, pre, "full", np.arange(3))
        q_rnd, x_rnd = factorized_inputs(pre, rnd, "full", np.arange(3))
        np.testing.assert_array_equal(q_pre, q_rnd)
        np.testing.assert_array_equal(q_rnd, pre["h_q"])
        self.assertFalse(np.array_equal(x_pre, x_rnd))


class IRTSAdapterTests(unittest.TestCase):
    def test_native_decoder_uses_left_padding(self):
        from scripts.run_native_qa import prepare_decoder_tokenizer

        class Tokenizer:
            padding_side = "right"
            pad_token_id = None
            pad_token = None
            eos_token = "<eos>"

        tok = prepare_decoder_tokenizer(Tokenizer())
        self.assertEqual(tok.padding_side, "left")
        self.assertEqual(tok.pad_token, "<eos>")

    def setUp(self):
        from temporal_abstraction.qa_adapters import irts

        self.irts = irts
        self.item = {
            "id": 1050,
            "answer_format": "multiple_choice_abcd",
            "question": (
                "Meta text.\nPrefix time series: [-0.080, 9.47, null]\n"
                "Prefix timestamps: [0.0, 1.0, 2.0]\n"
                "Full timestamps: [0.0, 1.0, 2.0, 3.0]\n"
                "Which continuation?\nA: 1.2000, 2.30\nB: 4.0, 5.00"
            ),
        }

    def test_extracts_series_and_all_timestamp_arrays(self):
        blocks = self.irts.extract_temporal_blocks(self.item["question"])
        self.assertEqual([b["name"] for b in blocks], [
            "Prefix time series", "Prefix timestamps", "Full timestamps"
        ])
        self.assertEqual(len(blocks), 3)

    def test_question_and_temporal_branches_are_isolated(self):
        item = self.irts.prepare_item(dict(self.item))
        q = self.irts.render_question(item)
        x = self.irts.render_temporal(item, "full", seed=7)
        self.assertNotIn("-0.080", q)
        self.assertIn("Which continuation?", q)
        self.assertNotIn("Which continuation?", x)
        self.assertIn("Prefix timestamps", x)
        self.assertIn("Full timestamps", x)

    def test_shuffle_preserves_numeric_lexemes_and_timestamp_precision(self):
        item = self.irts.prepare_item(dict(self.item))
        full = self.irts.render_temporal(item, "full", seed=7)
        shuffled = self.irts.render_temporal(item, "shuffled_ts", seed=7)
        self.assertIn("-0.080", shuffled)
        self.assertIn("9.47", shuffled)
        self.assertIn("[0.0, 1.0, 2.0]", shuffled)
        self.assertIn("[0.0, 1.0, 2.0, 3.0]", shuffled)
        self.assertCountEqual(
            self.irts.series_lexemes(full), self.irts.series_lexemes(shuffled)
        )

    def test_parser_matches_official_final_answer_behavior(self):
        # The released evaluator currently treats the last ``Choice X:`` marker as an answer.
        # We preserve that behavior for comparability and record the ambiguity in the audit.
        self.assertEqual(self.irts.parse(
            "Choice B: values\nChoice C: values\nB and C are similar.",
            "multiple_choice_abcd",
        ), "C")
        self.assertEqual(self.irts.parse("Reasoning.\nFinal answer: B", "multiple_choice_abcd"), "B")
        self.assertEqual(self.irts.parse("Reasoning.\n\nTrue", "true_false"), "T")


class Level2ManifestTests(unittest.TestCase):
    def test_stable_seed_is_cross_process_stable(self):
        from scripts.run_level2_probes import stable_seed

        expected = stable_seed("R1", "TREND", 7)
        code = (
            "from scripts.run_level2_probes import stable_seed; "
            "print(stable_seed('R1','TREND',7))"
        )
        values = []
        for hash_seed in ("1", "987654"):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed)
            values.append(int(subprocess.check_output(
                [sys.executable, "-c", code], cwd=REPO, env=env, text=True
            ).strip()))
        self.assertEqual(values, [expected, expected])

    def test_r4_and_r7_are_not_the_same_builder(self):
        from scripts.run_level2_probes import RELATIONS

        builders = dict(RELATIONS)
        self.assertIsNot(builders["R4_same_family"], builders["R7_similarity_ordering"])

    def test_constant_unseen_labels_are_flagged(self):
        from scripts.run_level2_probes import valid_c1_axes

        y = np.asarray([[1, 0, 1], [1, 1, 1], [1, 0, 1]])
        self.assertEqual(valid_c1_axes(y), [1])


class MetricDirectionTests(unittest.TestCase):
    def test_lower_is_better_delta_is_oriented_as_improvement(self):
        from scripts.aggregate_abstraction import oriented_delta

        self.assertAlmostEqual(oriented_delta("nmae", 0.2, 0.6), 0.4)
        self.assertAlmostEqual(oriented_delta("balanced_accuracy", 0.7, 0.5), 0.2)


if __name__ == "__main__":
    unittest.main()
