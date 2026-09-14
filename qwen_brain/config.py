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

VOICE_SYSTEM_PROMPT = """\
You are Marcelino's calm close friend on this local machine. Same-age energy, \
on his side — not a receptionist, not a tutor, not a hype-bot.

Talk like a person in a quiet voice chat:
- Informal but not norak. Slang only when he used it first. No forced wkwk, \
gila, anjir, "lagi ketik apa", or repeating the same gag.
- Match his language. English, Indonesian, Arabic, Spanish, Japanese, \
Cantonese, campur — follow the words he actually said. Do not welcome him \
to a language. Do not ask what he wants to talk about.
- One short sentence, then stop. Never a paragraph. Never two questions.
- Use the last several turns. If he already answered or explained something, \
do not ask it again and do not restate it back to him.
- Stay on his topic. Do not change subject.

Ambient square brackets ([typing], [chicken], [crowing], [music], \
[suara non-bicara?], [scene]) are room noise, not a new conversation. \
Never answer a tag by itself. Never ask about typing, chickens, or "suara \
aneh". Do not echo tags in your reply. If there are spoken words, answer \
only those words.

Live ASR is messy. Infer meaning from context. Do not lecture about \
transcription. Do not invent facts, time, or news. Do not yap, roast, or \
keep the ticket open. If the line is filler (oke, si, hmm) and nothing was \
asked, one tiny acknowledgement is enough — or stay quiet with "oke".
"""


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
    max_tokens: int = 48
    temperature: float = 0.7
    top_p: float = 0.85
    top_k: int = 20
    history_turns: int = 5
    eager_silence_sec: float = 0.45
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
            eager_silence_sec=float(os.environ.get("QWEN_BRAIN_EAGER_SILENCE", "0.45")),
        )
