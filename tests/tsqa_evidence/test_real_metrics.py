from __future__ import annotations

import unittest

from tsqa_evidence.real_metrics import (
    aligned_improvement,
    assert_metric_consistency,
    cluster_bootstrap,
    four_cell_metrics,
    hierarchical_macro,
    numeric_metrics,
)


class FourCellMetricTests(unittest.TestCase):
    def test_exact_metrics_and_full_denominator(self):
        rows = [
            dict(family_id="f1", t0=1, c0=1, t1=1, c1=1,
                 pred_t0="a", pred_t1="b", pred_c0="c", pred_c1="c"),
            dict(family_id="f2", t0=1, c0=1, t1=0, c1=1,
                 pred_t0="a", pred_t1=None, pred_c0="c", pred_c1="c",
                 valid_t1=False),
            dict(family_id="f3", t0=1, c0=0, t1=1, c1=0,
                 pred_t0="a", pred_t1="b", pred_c0="c", pred_c1="d"),
            dict(family_id="f4", t0=0, c0=1, t1=0, c1=1,
                 pred_t0="a", pred_t1="a", pred_c0="c", pred_c1="c"),
        ]
        metric = four_cell_metrics(rows)
        self.assertEqual(metric["fcjs"], 0.25)
        self.assertEqual(metric["tus"], 0.5)
        self.assertEqual(metric["cis"], 0.75)
        self.assertEqual(metric["qsja"], 0.375)
        self.assertEqual(metric["vor"], 0.75)
        self.assertEqual(metric["tus_given_t0_correct"], {"numerator": 2, "denominator": 3, "value": 2 / 3})
        self.assertEqual(metric["cis_given_c0_correct"], {"numerator": 3, "denominator": 3, "value": 1.0})

    def test_consistency_assertion_rejects_impossible_values(self):
        with self.assertRaises(AssertionError):
            assert_metric_consistency({"fcjs": .8, "tus": .5, "cis": .9, "qsja": .9})


class NumericMetricTests(unittest.TestCase):
    def test_parse_failure_is_wrong_but_not_imputed_into_mae(self):
        rows = [
            {"gold": 10, "prediction": 11, "valid_output": True},
            {"gold": 20, "prediction": None, "valid_output": False},
            {"gold": 30, "prediction": 28, "valid_output": True},
        ]
        metric = numeric_metrics(rows, abs_tol=1.0, rel_tol=0.0, scale_task=10.0)
        self.assertEqual(metric["valid_n"], 2)
        self.assertEqual(metric["vor"], 2 / 3)
        self.assertEqual(metric["tolerance_success_full_denominator"], 1 / 3)
        self.assertEqual(metric["mae_valid_only"], 1.5)
        self.assertEqual(metric["nmae_valid_only"], 0.15)

    def test_lower_is_better_alignment(self):
        item = aligned_improvement("NMAE", pretrained=.2, random_init=.6)
        self.assertAlmostEqual(item["raw_delta_pretrained_minus_random"], -.4)
        self.assertAlmostEqual(item["aligned_improvement_positive_is_better"], .4)
        self.assertEqual(item["direction"], "lower_is_better")


class AggregationTests(unittest.TestCase):
    def test_domain_macro_does_not_let_large_source_dominate(self):
        rows = []
        for i in range(10):
            rows.append({"domain": "large", "task_type": "t", "group_id": f"g{i}", "family_id": f"f{i}", "value": 1.0})
        rows.append({"domain": "small", "task_type": "t", "group_id": "gX", "family_id": "fX", "value": 0.0})
        result = hierarchical_macro(rows)
        self.assertEqual(result["headline_domain_macro"], 0.5)
        self.assertAlmostEqual(result["raw_pooled_descriptive"], 10 / 11)

    def test_cluster_bootstrap_resamples_whole_groups(self):
        rows = [
            {"group_id": "a", "value": 0.0}, {"group_id": "a", "value": 0.0},
            {"group_id": "b", "value": 1.0}, {"group_id": "b", "value": 1.0},
        ]
        result = cluster_bootstrap(rows, lambda sample: sum(x["value"] for x in sample) / len(sample), iterations=100, seed=7)
        self.assertEqual(result["n_groups"], 2)
        self.assertEqual(result["estimate"], 0.5)
        self.assertEqual(result["ci_stability"], "UNSTABLE_FEW_GROUPS")


if __name__ == "__main__":
    unittest.main()
