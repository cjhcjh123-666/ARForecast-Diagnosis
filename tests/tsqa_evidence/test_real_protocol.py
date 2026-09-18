from __future__ import annotations

import json
import unittest
from pathlib import Path

from tsqa_evidence.real_protocol import (
    parse_strict_choice,
    render_native,
    render_question_option,
    render_temporal,
    temporal_payload,
)


REPO = Path(__file__).resolve().parents[2]


def first_family_and_item():
    with (REPO / "results/tsqa_evidence/real_v1/manifests/families_candidate.jsonl").open() as handle:
        family = json.loads(next(handle))
    with (REPO / "results/tsqa_evidence/real_v1/manifests/item_manifest.jsonl").open() as handle:
        items = [json.loads(line) for line in handle]
    item = next(x for x in items if x["family_id"] == family["family_id"] and x["cell"] == "t0")
    return family, item


class RenderingTests(unittest.TestCase):
    def test_temporal_branch_excludes_question_options_and_gold(self):
        family, item = first_family_and_item()
        text = render_temporal(family, "x")
        self.assertNotIn(item["question"], text)
        self.assertNotIn("Candidate A", text)
        self.assertNotIn(item["gold_semantic"], text)

    def test_question_option_branch_excludes_temporal_values(self):
        family, item = first_family_and_item()
        text = render_question_option(item, "A")
        self.assertIn(item["question"], text)
        self.assertNotIn(str(family["original_evidence"][0][0]), text)
        self.assertNotIn("timestamps", text)

    def test_full_and_shuffle_use_identical_precision_and_multiset(self):
        family, _ = first_family_and_item()
        full = temporal_payload(family, "x", "full")
        shuffled = temporal_payload(family, "x", "shuffled")
        self.assertEqual(full["decimal_places"], shuffled["decimal_places"])
        self.assertEqual(sorted(map(tuple, full["values"])), sorted(map(tuple, shuffled["values"])))
        self.assertEqual(full["timestamps"], shuffled["timestamps"])

    def test_question_only_removes_all_evidence(self):
        family, item = first_family_and_item()
        text = render_native(family, item, "question_only")
        self.assertIn("[omitted]", text)
        self.assertNotIn(str(family["original_evidence"][0][0]), text)


class ParserTests(unittest.TestCase):
    def test_strict_parser_accepts_protocol_outputs(self):
        self.assertEqual(parse_strict_choice("A", ("A", "B")), "A")
        self.assertEqual(parse_strict_choice("Answer: b", ("A", "B")), "B")

    def test_strict_parser_rejects_explanations_and_out_of_range(self):
        self.assertIsNone(parse_strict_choice("I think A because..."))
        self.assertIsNone(parse_strict_choice("C", ("A", "B")))
        self.assertIsNone(parse_strict_choice("A or B", ("A", "B")))


if __name__ == "__main__":
    unittest.main()
