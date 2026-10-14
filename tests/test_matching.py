"""Unit tests for the pure OCR-matching logic (no model needed).

`ocr_service` is imported with a stubbed `paddleocr` module so the
heavy PaddlePaddle import is skipped. The tested functions
(`normalize_for_match`, `get_top_matches`, `build_ocr_response`,
`process_medicine_image` unreadable-path guard) never touch the model.

Run:  python -m unittest discover -s tests
"""
import os
import sys
import tempfile
import types
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fake_paddle = types.ModuleType("paddleocr")
fake_paddle.PaddleOCR = object
sys.modules.setdefault("paddleocr", fake_paddle)

from ocr_service import (  # noqa: E402
    build_ocr_response,
    downscale_if_needed,
    drug_database,
    get_top_matches,
    normalize_for_match,
    preprocess_image,
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


class DebugFieldsTest(unittest.TestCase):

    def setUp(self):
        self._old = os.environ.get("DEBUG_OCR")
        os.environ["DEBUG_OCR"] = "true"
        self.addCleanup(self._restore)

    def _restore(self):
        if self._old is None:
            os.environ.pop("DEBUG_OCR", None)
        else:
            os.environ["DEBUG_OCR"] = self._old

    def test_debug_off_hides_fields(self):
        os.environ["DEBUG_OCR"] = "false"
        res = process_medicine_image(os.path.join("nope", "missing.jpg"))

        self.assertNotIn("raw_texts", res)
        self.assertNotIn("top_matches", res)

    def test_debug_on_shows_fields(self):
        res = process_medicine_image(os.path.join("nope", "missing.jpg"))

        self.assertIn("raw_texts", res)
        self.assertIn("top_matches", res)
        self.assertEqual(res["raw_texts"], [])
        self.assertEqual(res["top_matches"], [])


class ImageSizeTest(unittest.TestCase):

    def _write_image(self, width, height):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        path = os.path.join(tempfile.gettempdir(), "ocr_size_test.jpg")
        cv2.imwrite(path, img)
        self.addCleanup(
            lambda: os.path.exists(path) and os.remove(path)
        )

        return path

    def test_small_image_kept_as_is(self):
        path = self._write_image(100, 80)
        used, temp = downscale_if_needed(path)

        self.assertEqual(used, path)
        self.assertIsNone(temp)

    def test_large_image_downscaled(self):
        path = self._write_image(2000, 1000)
        used, temp = downscale_if_needed(path)
        self.addCleanup(
            lambda: temp and os.path.exists(temp) and os.remove(temp)
        )

        img = cv2.imread(used)

        self.assertLessEqual(max(img.shape[:2]), 1600)

    def test_preprocess_respects_cap(self):
        path = self._write_image(200, 200)
        out = preprocess_image(path)
        self.addCleanup(
            lambda: out and os.path.exists(out) and os.remove(out)
        )

        self.assertIsNotNone(out)

        img = cv2.imread(out)

        self.assertLessEqual(max(img.shape[:2]), 1600)


if __name__ == "__main__":
    unittest.main()
