from __future__ import annotations

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
class TestPipelineSmoke(unittest.TestCase):
    def test_smoke_dialog_ru_timed(self) -> None:
        audio_path = Path("audio/dialog_ru_timed.wav")
        if not audio_path.exists():
            self.skipTest(f"Missing audio fixture: {audio_path}")

        if not Path("hf.txt").exists() and not os.getenv("HF_TOKEN"):
            self.skipTest("HF token is required for integration run.")

        with tempfile.TemporaryDirectory(prefix="meeting_v15_it_") as tmp_dir_name:
            config = replace(
                MeetingConfig(),
                output_dir=tmp_dir_name,
                emit_debug=False,
                move_processed_to=None,
            )
            result = run_meeting_transcription(str(audio_path), config=config)
            outputs = result["outputs"]
            self.assertTrue(Path(outputs["text_path"]).exists())
            self.assertTrue(Path(outputs["json_path"]).exists())


if __name__ == "__main__":
    unittest.main()

