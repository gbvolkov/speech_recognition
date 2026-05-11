from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from meeting_v15.config import MeetingConfig
from meeting_v15.pipeline import run_meeting_transcription


@unittest.skipUnless(
    os.getenv("MV15_RUN_INTEGRATION") == "1",
    "Set MV15_RUN_INTEGRATION=1 to run heavy integration tests.",
)
class TestSegmentsSchema(unittest.TestCase):
    def test_segments_json_schema(self) -> None:
        audio_path = Path("audio/dialog_ru_timed.wav")
        if not audio_path.exists():
            self.skipTest(f"Missing audio fixture: {audio_path}")

        if not Path("hf.txt").exists() and not os.getenv("HF_TOKEN"):
            self.skipTest("HF token is required for integration run.")

        with tempfile.TemporaryDirectory(prefix="meeting_v15_schema_") as tmp_dir_name:
            config = replace(MeetingConfig(), output_dir=tmp_dir_name, emit_debug=False)
            result = run_meeting_transcription(str(audio_path), config=config)
            payload = json.loads(Path(result["outputs"]["json_path"]).read_text(encoding="utf-8"))

        self.assertIn("segments", payload)
        for segment in payload["segments"]:
            self.assertEqual(
                set(segment.keys()),
                {"start_s", "end_s", "speaker_id", "overlap_flag", "text", "source", "words"},
            )
            self.assertIsInstance(segment["start_s"], float)
            self.assertIsInstance(segment["end_s"], float)
            self.assertIsInstance(segment["speaker_id"], str)
            self.assertIsInstance(segment["overlap_flag"], bool)
            self.assertIsInstance(segment["text"], str)
            self.assertIsInstance(segment["source"], str)
            self.assertIsInstance(segment["words"], list)
            for word in segment["words"]:
                self.assertIn("start_s", word)
                self.assertIn("end_s", word)
                self.assertIn("word", word)


if __name__ == "__main__":
    unittest.main()

