from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from meeting_v15.exporters import write_segments_json, write_text_transcript


class TestExporters(unittest.TestCase):
    def test_write_text_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            output = Path(tmp_dir_name) / "out.txt"
            payload = write_text_transcript(
                [
                    {
                        "start_s": 0.0,
                        "end_s": 2.0,
                        "speaker_id": "SPEAKER_1",
                        "overlap_flag": False,
                        "text": "hello",
                        "words": [],
                        "source": "diarized_turn",
                    },
                    {
                        "start_s": 2.0,
                        "end_s": 4.0,
                        "speaker_id": "OVERLAP",
                        "overlap_flag": True,
                        "text": "cross talk",
                        "words": [],
                        "source": "overlap_second_pass",
                    },
                ],
                output,
            )
            self.assertTrue(output.exists())
            self.assertIn("**SPEAKER_1**", payload)
            self.assertIn("[OVERLAP] cross talk", payload)

    def test_write_segments_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            output = Path(tmp_dir_name) / "out.json"
            result = {
                "source_file": "audio/example.wav",
                "duration_s": 6.0,
                "segments": [
                    {
                        "start_s": 0.0,
                        "end_s": 1.0,
                        "speaker_id": "SPEAKER_1",
                        "overlap_flag": False,
                        "text": "hello",
                        "words": [{"start_s": 0.0, "end_s": 0.4, "word": "hello"}],
                        "source": "diarized_turn",
                    }
                ],
                "metadata": {},
                "outputs": {},
            }
            payload = write_segments_json(result, output)
            self.assertTrue(output.exists())
            self.assertEqual(set(payload.keys()), {"source_file", "duration_s", "segments", "metadata", "outputs"})
            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(loaded["segments"][0]["speaker_id"], "SPEAKER_1")


if __name__ == "__main__":
    unittest.main()

