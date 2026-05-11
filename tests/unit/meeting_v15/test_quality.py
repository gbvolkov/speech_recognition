from __future__ import annotations

import unittest

from meeting_v15.quality import has_repetition, pick_best_candidate, rerun_reasons, temperature_ladder


class TestQuality(unittest.TestCase):
    def test_has_repetition(self) -> None:
        repetitive = "hello world hello world hello world hello world hello world"
        self.assertTrue(has_repetition(repetitive))

    def test_rerun_reasons(self) -> None:
        reasons = rerun_reasons(
            text="",
            duration_s=3.5,
            has_vad_speech=True,
            overlap_flag=True,
            overlap_second_pass=True,
        )
        self.assertIn("empty_with_speech", reasons)
        self.assertIn("overlap", reasons)

    def test_pick_best_candidate_prefers_non_repetitive(self) -> None:
        candidates = [
            {"text": "aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa", "words": []},
            {"text": "normal meeting sentence with varied terms", "words": []},
        ]
        best = pick_best_candidate(candidates)
        self.assertIn("normal meeting sentence", best["text"])

    def test_temperature_ladder(self) -> None:
        self.assertEqual(temperature_ladder(), (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))


if __name__ == "__main__":
    unittest.main()

