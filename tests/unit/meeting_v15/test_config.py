from __future__ import annotations

import os
import unittest

from meeting_v15.config import MeetingConfig, from_env


class TestMeetingConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        config = MeetingConfig()
        self.assertEqual(config.whisper_model_id, "openai/whisper-large-v3")
        self.assertEqual(config.language, MeetingConfig.language)
        self.assertTrue(config.overlap_second_pass)

    def test_validation_min_speakers_gt_max(self) -> None:
        with self.assertRaises(ValueError):
            MeetingConfig(min_speakers=4, max_speakers=2)

    def test_validation_negative_duration(self) -> None:
        with self.assertRaises(ValueError):
            MeetingConfig(chunk_pad_s=-0.1)

    def test_from_env_overrides(self) -> None:
        os.environ["MV15_MIN_SPEAKERS"] = "3"
        os.environ["MV15_MAX_SPEAKERS"] = "7"
        os.environ["MV15_OVERLAP_SECOND_PASS"] = "false"
        try:
            config = from_env()
        finally:
            del os.environ["MV15_MIN_SPEAKERS"]
            del os.environ["MV15_MAX_SPEAKERS"]
            del os.environ["MV15_OVERLAP_SECOND_PASS"]

        self.assertEqual(config.min_speakers, 3)
        self.assertEqual(config.max_speakers, 7)
        self.assertFalse(config.overlap_second_pass)


if __name__ == "__main__":
    unittest.main()
