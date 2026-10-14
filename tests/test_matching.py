"""Unit tests for the pure OCR-matching logic (no model needed).

`ocr_service` is imported with a stubbed `paddleocr` module so the
heavy PaddlePaddle import is skipped. The tested functions
(`normalize_for_match`, `get_top_matches`, `build_ocr_response`,
`process_medicine_image` unreadable-path guard) never touch the model.

Run:  python -m unittest discover -s tests
"""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fake_paddle = types.ModuleType("paddleocr")
fake_paddle.PaddleOCR = object
sys.modules.setdefault("paddleocr", fake_paddle)

from ocr_service import (  # noqa: E402
    build_ocr_response,
    drug_database,
    get_top_matches,
    normalize_for_match,
    process_medicine_image,
)


class NormalizeTest(unittest.TestCase):

    def test_alef_variants_unified(self):
        self.assertEqual(
            normalize_for_match("أوجمنتين"),
            normalize_for_match("اوجمنتين"),
        )
        self.assertEqual(
            normalize_for_match("إيبوبروفين"),
            normalize_for_match("ايبوبروفين"),
        )

    def test_latin_lowercased(self):
        self.assertEqual(normalize_for_match("Panadol"), "panadol")


class TopMatchesTest(unittest.TestCase):

    def test_exact_brand_match_scores_one(self):
        detected = [{"text": "Panadol", "confidence": 0.99}]
        matches = get_top_matches(detected, drug_database, top_n=3)

        self.assertEqual(matches[0]["drug_name"], "Panadol")
        self.assertEqual(matches[0]["score"], 1.0)

    def test_arabic_alias_match(self):
        detected = [{"text": "بانادول", "confidence": 0.95}]
        matches = get_top_matches(detected, drug_database, top_n=3)

        self.assertEqual(matches[0]["drug_name"], "Panadol")

    def test_unique_drugs_only(self):
        detected = [
            {"text": "Panadol extra", "confidence": 0.9},
            {"text": "panadol", "confidence": 0.8},
        ]
        matches = get_top_matches(detected, drug_database, top_n=3)
        drugs = [m["drug_name"] for m in matches]

        self.assertEqual(len(drugs), len(set(drugs)))

    def test_top_n_respected(self):
        detected = [{"text": "x", "confidence": 0.5}]
        matches = get_top_matches(detected, drug_database, top_n=2)

        self.assertLessEqual(len(matches), 2)


class ResponseTest(unittest.TestCase):

    def _match(self, drug, score):
        return [{
            "drug_name": drug,
            "matched_alias": drug,
            "ocr_text": drug,
            "score": score,
        }]

    def test_confirmed_threshold(self):
        res = build_ocr_response(self._match("Panadol", 0.85))

        self.assertEqual(res["status"], "confirmed")
        self.assertEqual(res["drug_name"], "Panadol")
        self.assertTrue(res["ocr_success"])

    def test_needs_confirmation_band(self):
        res = build_ocr_response(self._match("Panadol", 0.70))

        self.assertEqual(res["status"], "needs_confirmation")
        self.assertIsNone(res["drug_name"])
        self.assertEqual(res["best_candidate"], "Panadol")

    def test_uncertain_low_score(self):
        res = build_ocr_response(self._match("Panadol", 0.40))

        self.assertEqual(res["status"], "uncertain")
        self.assertIsNone(res["drug_name"])

    def test_empty_matches_not_found(self):
        res = build_ocr_response([])

        self.assertEqual(res["status"], "not_found")
        self.assertFalse(res["ocr_success"])


class UnreadableImageTest(unittest.TestCase):

    def test_missing_file_returns_envelope(self):
        res = process_medicine_image(os.path.join("nope", "missing.jpg"))

        self.assertEqual(res["status"], "not_found")
        self.assertFalse(res["ocr_success"])
        self.assertEqual(res["source"], "ocr")
        self.assertEqual(res["intent"], "search_medicine")


if __name__ == "__main__":
    unittest.main()
