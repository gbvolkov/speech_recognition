from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path


def ensure_ffmpeg_available() -> str:
    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        raise RuntimeError(
            "ffmpeg is not available in PATH. Install ffmpeg and retry."
        )
    return ffmpeg_exe


def _run(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(stderr or "Subprocess command failed.") from exc


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        sample_rate = wav.getframerate()
        if sample_rate <= 0:
            return 0.0
        return frames / float(sample_rate)


def normalize_audio(input_path: str | Path, work_dir: str | Path, sample_rate: int = 16000) -> dict:
    ffmpeg_exe = ensure_ffmpeg_available()
    source = Path(input_path)
    if not source.exists():
        raise RuntimeError(f"Input file does not exist: {source}")

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    output_wav = work / f"{source.stem}.normalized.wav"

    filter_chain = "loudnorm=I=-16:TP=-1.5:LRA=11,alimiter=limit=0.95"
    command = [
        ffmpeg_exe,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-af",
        filter_chain,
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_wav),
    ]
    _run(command)

    duration_s = _wav_duration_seconds(output_wav)
    return {
        "wav_path": str(output_wav),
        "duration_s": duration_s,
        "sample_rate": sample_rate,
        "channels": 1,
    }


def extract_audio_window(
    input_wav_path: str | Path,
    output_wav_path: str | Path,
    start_s: float,
    end_s: float,
    sample_rate: int = 16000,
) -> None:
    ffmpeg_exe = ensure_ffmpeg_available()
    start_s = max(0.0, float(start_s))
    end_s = max(start_s + 0.01, float(end_s))
    duration_s = end_s - start_s

    command = [
        ffmpeg_exe,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{duration_s:.3f}",
        "-i",
        str(input_wav_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_wav_path),
    ]
    _run(command)

