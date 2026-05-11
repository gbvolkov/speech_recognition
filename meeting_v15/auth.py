from __future__ import annotations

from pathlib import Path


def load_hf_token(explicit: str | None, token_file: str = "hf.txt") -> str:
    if explicit and explicit.strip():
        return explicit.strip()

    import os

    env_token = os.getenv("HF_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()

    candidate = Path(token_file)
    if candidate.exists():
        token = candidate.read_text(encoding="utf-8").strip()
        if token:
            return token

    raise RuntimeError(
        "Hugging Face token is required but not found. "
        "Provide --hf-token, set HF_TOKEN, or create hf.txt."
    )
