from __future__ import annotations

import unittest

from meeting_v15.chunk_builder import build_chunks, build_fallback_chunks_from_vad


class TestChunkBuilder(unittest.TestCase):
    def test_build_chunks_merge_split_overlap(self) -> None:
        turns = [
            {"speaker_id": "SPEAKER_1", "start_s": 0.0, "end_s": 10.0},
            {"speaker_id": "SPEAKER_1", "start_s": 10.3, "end_s": 22.0},
        ]
        vad_regions = [{"start_s": 0.0, "end_s": 22.0}]
        overlap_windows = [{"start_s": 5.0, "end_s": 6.0}, {"start_s": 18.0, "end_s": 19.0}]

        chunks = build_chunks(
            turns=turns,
            vad_regions=vad_regions,
            overlap_windows=overlap_windows,
            audio_duration_s=25.0,
            merge_gap_s=0.5,
            max_segment_s=10.0,
            stride_s=2.0,
            pad_s=0.0,
        )

        self.assertEqual(len(chunks), 3)
        self.assertEqual([c["speaker_id"] for c in chunks], ["SPEAKER_1", "SPEAKER_1", "SPEAKER_1"])
        self.assertEqual([c["overlap_flag"] for c in chunks], [True, False, True])
        self.assertEqual([c["source"] for c in chunks], ["diarized_turn"] * 3)

    def test_fallback_chunks_from_vad(self) -> None:
        chunks = build_fallback_chunks_from_vad(
            vad_regions=[{"start_s": 1.0, "end_s": 4.0}],
            overlap_windows=[],
            audio_duration_s=10.0,
            max_segment_s=30.0,
            stride_s=0.75,
            pad_s=0.0,
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["speaker_id"], "UNKNOWN")
        self.assertEqual(chunks[0]["source"], "vad_fallback")


if __name__ == "__main__":
    unittest.main()

