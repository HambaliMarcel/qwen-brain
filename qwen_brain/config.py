"""Paths and ports. ASR stays on 9999; this brain uses a second llama-server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LLAMA = Path(r"C:\AI\llama.cpp\llama-server.exe")
DEFAULT_MODEL = Path(r"C:\AI\models\Qwen3.5-4B-Q8_0.gguf")
DEFAULT_BRAIN_PORT = 8080
DEFAULT_STT_PORT = 18765
DEFAULT_HERMES_API = "http://127.0.0.1:8642"

VOICE_SYSTEM_PROMPT = (
    "You are the local voice brain on this PC. You hear live speech-to-text, "
    "which can contain small ASR mistakes — answer the intended meaning. "
    "Bracketed tags like [finger snapping] or [music] are environmental sounds "
    "from PANN, not words; use them as scene context and do not answer the tag "
    "as if it were a question. "
    "Reply in the user's language (Indonesian stays Indonesian, English stays "
    "English, mixed stays mixed). Keep answers short enough to speak: 1-3 "
    "sentences, no emoji, no markdown, no lists unless asked. Sound like a "
    "helpful friend. Do not mention tools, models, or that you are an AI "
    "unless asked."
)


def _path(name: str, fallback: Path) -> Path:
    v = os.environ.get(name)
    return Path(v) if v else fallback


@dataclass
class BrainConfig:
    llama_server: Path = DEFAULT_LLAMA
    model: Path = DEFAULT_MODEL
    host: str = "127.0.0.1"
    port: int = DEFAULT_BRAIN_PORT
    ctx: int = 8192
    ngl: int = 99
    stt_host: str = "127.0.0.1"
    stt_port: int = DEFAULT_STT_PORT
    max_tokens: int = 120
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    history_turns: int = 6
    eager_silence_sec: float = 0.12
    eager_revise_chars: int = 8
    eager: bool = True
    backend: str = "llm"  # llm | hermes
    hermes_api: str = DEFAULT_HERMES_API
    hermes_api_key: str = ""
    system_prompt: str = VOICE_SYSTEM_PROMPT

    @property
    def url(self) -> str:
        h = self.host
        if h.startswith("http://") or h.startswith("https://"):
            return h
        return f"http://{h}:{self.port}"

    @classmethod
    def from_env(cls) -> "BrainConfig":
        return cls(
            llama_server=_path("LLAMA_SERVER", DEFAULT_LLAMA),
            model=_path("QWEN_BRAIN_MODEL", DEFAULT_MODEL),
            host=os.environ.get("QWEN_BRAIN_HOST", "127.0.0.1"),
            port=int(os.environ.get("QWEN_BRAIN_PORT", DEFAULT_BRAIN_PORT)),
            ctx=int(os.environ.get("QWEN_BRAIN_CTX", 8192)),
            ngl=int(os.environ.get("QWEN_BRAIN_NGL", 99)),
            stt_host=os.environ.get("QWEN_BRAIN_STT_HOST", "127.0.0.1"),
            stt_port=int(os.environ.get("QWEN_BRAIN_STT_PORT", DEFAULT_STT_PORT)),
            hermes_api=os.environ.get("HERMES_API", DEFAULT_HERMES_API),
            hermes_api_key=os.environ.get("HERMES_API_KEY", os.environ.get("API_SERVER_KEY", "")),
            backend=os.environ.get("QWEN_BRAIN_BACKEND", "llm").strip().lower(),
            eager_silence_sec=float(os.environ.get("QWEN_BRAIN_EAGER_SILENCE", "0.12")),
        )
