from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tsqa_evidence.benchmark import (CONTROL_TASK, TASK_IDS, gold, make_base, make_irrelevant,
                                     make_relevant, parse_series, serialize_series, stable_seed)
from tsqa_evidence.candidate_scorer import QuestionEvidenceScorer
from tsqa_evidence.evaluation import improvement, paired_metrics
from tsqa_evidence.rendering import render_question_option, render_temporal


class EvidenceFamilyTests(unittest.TestCase):
    def test_stable_seed_is_cross_process_stable(self):
        expected = stable_seed("F0007", "A1")
        code = "from tsqa_evidence.benchmark import stable_seed; print(stable_seed('F0007','A1'))"
        observed = []
        for hash_seed in ("1", "991"):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed)
            observed.append(int(subprocess.check_output([sys.executable, "-c", code], cwd=REPO,
                                                        env=env, text=True).strip()))
        self.assertEqual(observed, [expected, expected])

    def test_visible_roundtrip_preserves_every_gold(self):
        values, _ = make_base("F0007", "train", 20260918)
        reconstructed = parse_series(serialize_series(values))
        np.testing.assert_array_equal(values, reconstructed)
        for task_id in TASK_IDS:
            self.assertEqual(gold(task_id, values), gold(task_id, reconstructed))

    def test_verified_interventions_have_required_relations(self):
        for i in range(24):
            values, _ = make_base(f"D{i:04d}", "train", 20260918)
            for task_id in TASK_IDS:
                relevant = make_relevant(values, task_id)
                irrelevant = make_irrelevant(values, task_id)
                self.assertNotEqual(gold(task_id, values), gold(task_id, relevant))
                self.assertEqual(gold(task_id, values), gold(task_id, irrelevant))
                control = CONTROL_TASK[task_id]
                self.assertEqual(gold(control, values), gold(control, relevant))

    def test_branch_rendering_is_separated(self):
        values, _ = make_base("D0001", "train", 20260918)
        row = {"question_text": "Which interval?", "options_json":
               '{"A":"first","B":"second","C":"third","D":"fourth"}'}
        temporal = render_temporal(values)
        question = render_question_option(row, "A")
        self.assertNotIn("Which interval?", temporal)
        self.assertNotIn("Candidate A", temporal)
        self.assertNotIn("sensor_A\n+", question)
        self.assertIn("Which interval?", question)


class CandidateScorerTests(unittest.TestCase):
    def test_candidate_score_difference_depends_on_temporal_input(self):
        scorer = QuestionEvidenceScorer(4, 5, projection_dim=8, hidden_dim=8, seed=7)
        q = torch.tensor([[[1., 0, 0, 0, 0], [0., 1, 0, 0, 0]],
                          [[1., 0, 0, 0, 0], [0., 1, 0, 0, 0]]])
        x = torch.tensor([[1., 0, 0, 0], [0., 1, 0, 0]])
        scores = scorer(x, q).detach()
        self.assertFalse(torch.isclose(scores[0, 0] - scores[0, 1],
                                       scores[1, 0] - scores[1, 1]))

    def test_paired_models_share_fixed_projection_and_initialization(self):
        p = QuestionEvidenceScorer(7, 9, projection_dim=8, hidden_dim=6, seed=17)
        r = QuestionEvidenceScorer(7, 9, projection_dim=8, hidden_dim=6, seed=17)
        self.assertEqual(p.projection_hashes(), r.projection_hashes())
        for a, b in zip(p.parameters(), r.parameters()):
            self.assertTrue(torch.equal(a, b))


class EvaluationTests(unittest.TestCase):
    def test_metric_directions(self):
        self.assertAlmostEqual(improvement("nmae", 0.2, 0.6), 0.4)
        self.assertAlmostEqual(improvement("accuracy", 0.7, 0.5), 0.2)
        self.assertAlmostEqual(improvement("r2", -0.2, -0.5), 0.3)

    def test_parse_failures_remain_in_pair_denominator(self):
        rows = [
            {"family_id": "f1", "task_id": "t", "condition": "FULL", "correct": True},
            {"family_id": "f1", "task_id": "t", "condition": "RELEVANT_EDIT", "correct": True},
            {"family_id": "f2", "task_id": "t", "condition": "FULL", "correct": True},
            {"family_id": "f2", "task_id": "t", "condition": "RELEVANT_EDIT", "correct": False,
             "parse_status": "UNPARSED"},
        ]
        metric = paired_metrics(rows)[0]
        self.assertEqual((metric["numerator"], metric["denominator"], metric["value"]), (1, 2, 0.5))


if __name__ == "__main__":
    unittest.main()
