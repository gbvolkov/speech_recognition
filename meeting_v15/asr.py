from __future__ import annotations

import copy
import uuid
from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from .audio_io import extract_audio_window
from .config import MeetingConfig
from .types import WordItem


class WhisperAsr:
    def __init__(self, config: MeetingConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            config.whisper_model_id,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            torch_dtype=self.torch_dtype,
        )
        self.processor = AutoProcessor.from_pretrained(config.whisper_model_id)
        model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=config.language, task=config.task
        )
        model.to(self.device)

        pipeline_device = 0 if self.device == "cuda" else -1
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=pipeline_device,
            torch_dtype=self.torch_dtype,
        )
        self.base_generation_config = copy.deepcopy(self.pipe.model.generation_config)
        self._apply_generation_defaults(self.base_generation_config)

    def _apply_generation_defaults(self, generation_config) -> None:
        generation_config.num_beams = int(self.config.decode_num_beams)
        generation_config.max_new_tokens = int(self.config.decode_max_new_tokens)
        generation_config.condition_on_prev_tokens = bool(
            self.config.decode_condition_on_prev_tokens
        )
        generation_config.no_speech_threshold = float(self.config.decode_no_speech_threshold)
        generation_config.compression_ratio_threshold = float(
            self.config.decode_compression_ratio_threshold
        )
        generation_config.logprob_threshold = float(self.config.decode_logprob_threshold)

        # Keep these explicit for Whisper generation behavior.
        if hasattr(generation_config, "language"):
            generation_config.language = self.config.language
        if hasattr(generation_config, "task"):
            generation_config.task = self.config.task

    def build_generation_config(
        self,
        num_beams: int | None = None,
        disable_whisper_internal_fallback: bool = False,
    ):
        generation_config = copy.deepcopy(self.base_generation_config)
        if num_beams is not None:
            generation_config.num_beams = int(num_beams)
        if disable_whisper_internal_fallback:
            generation_config.compression_ratio_threshold = None
            generation_config.logprob_threshold = None
            generation_config.no_speech_threshold = None
        return generation_config

    def decode_window(
        self,
        wav_path: str,
        start_s: float,
        end_s: float,
        work_dir: str | Path,
        generation_config=None,
        temperature: float | None = None,
        word_timestamps: bool = True,
    ) -> dict:
        work_path = Path(work_dir)
        work_path.mkdir(parents=True, exist_ok=True)
        chunk_wav = work_path / f"asr_chunk_{uuid.uuid4().hex}.wav"

        extract_audio_window(
            input_wav_path=wav_path,
            output_wav_path=chunk_wav,
            start_s=start_s,
            end_s=end_s,
            sample_rate=self.config.sample_rate,
        )

        try:
            config_for_call = generation_config or self.build_generation_config()
            self.pipe.generation_config = config_for_call
            decode_temperature = (
                float(temperature)
                if temperature is not None
                else float(self.config.decode_temperature)
            )
            script = self.pipe(
                str(chunk_wav),
                return_timestamps="word" if word_timestamps else False,
                temperature=decode_temperature,
            )
        finally:
            chunk_wav.unlink(missing_ok=True)
        text = (script.get("text") or "").strip()
        words = self._extract_words(script, base_start_s=start_s, enabled=word_timestamps)

        return {
            "text": text,
            "words": words,
            "raw": script,
            "generation_config": config_for_call.to_dict()
            if hasattr(config_for_call, "to_dict")
            else {},
        }

    def decode_full_audio(
        self,
        wav_path: str,
        generation_config=None,
        temperature: float | None = None,
        word_timestamps: bool = True,
    ) -> dict:
        config_for_call = generation_config or self.build_generation_config()
        self.pipe.generation_config = config_for_call
        decode_temperature = (
            float(temperature)
            if temperature is not None
            else float(self.config.decode_temperature)
        )
        script = self.pipe(
            str(wav_path),
            return_timestamps="word" if word_timestamps else False,
            temperature=decode_temperature,
            chunk_length_s=float(self.config.whisper_chunk_length_s),
            stride_length_s=float(self.config.whisper_stride_length_s),
        )
        text = (script.get("text") or "").strip()
        words = self._extract_words(script, base_start_s=0.0, enabled=word_timestamps)
        return {
            "text": text,
            "words": words,
            "raw": script,
            "generation_config": config_for_call.to_dict()
            if hasattr(config_for_call, "to_dict")
            else {},
        }

    def _extract_words(self, script: dict, base_start_s: float, enabled: bool) -> list[WordItem]:
        words: list[WordItem] = []
        if not enabled:
            return words

        for raw in script.get("chunks", []):
            timestamp = raw.get("timestamp")
            if not isinstance(timestamp, (tuple, list)) or len(timestamp) != 2:
                continue
            local_start, local_end = timestamp
            if local_start is None or local_end is None:
                continue
            local_start = float(local_start)
            local_end = float(local_end)
            if local_end <= local_start:
                local_end = local_start + 0.01
            token = (raw.get("text") or "").strip()
            if not token:
                continue

            word: WordItem = {
                "start_s": base_start_s + local_start,
                "end_s": base_start_s + local_end,
                "word": token,
            }
            if "confidence" in raw and raw["confidence"] is not None:
                try:
                    word["conf"] = float(raw["confidence"])
                except (TypeError, ValueError):
                    pass
            words.append(word)
        return words
