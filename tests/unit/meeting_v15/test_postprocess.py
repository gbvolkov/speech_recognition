from __future__ import annotations

import unittest

from meeting_v15.postprocess import build_segments, normalize_text


class TestPostprocess(unittest.TestCase):
    def test_normalize_text(self) -> None:
        self.assertEqual(normalize_text("  hello   ,   world  "), "hello, world")

    def test_build_segments_sorts_and_normalizes(self) -> None:
        records = [
            {
                "start_s": 5.0,
                "end_s": 6.0,
                "speaker_id": "OVERLAP",
                "overlap_flag": True,
                "text": "  hi   there ",
                "words": [{"start_s": 5.0, "end_s": 5.2, "word": "  hi "}],
                "source": "overlap_second_pass",
            },
            {
                "start_s": 1.0,
                "end_s": 2.0,
                "speaker_id": "SPEAKER_1",
                "overlap_flag": False,
                "text": "  first sentence ",
                "words": [],
                "source": "diarized_turn",
            },
        ]
        segments = build_segments(records)
        self.assertEqual(segments[0]["start_s"], 1.0)
        self.assertEqual(segments[1]["speaker_id"], "OVERLAP")
        self.assertEqual(segments[1]["text"], "hi there")


if __name__ == "__main__":
    unittest.main()

