"""Paths and ports. ASR stays on 9999; this brain uses a second llama-server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LLAMA = Path(r"C:\AI\llama.cpp\llama-server.exe")
DEFAULT_MODEL = Path(r"C:\AI\models\Qwen3.8-27B-Uncensored-YMQ-XS-TI.gguf")
DEFAULT_BRAIN_PORT = 8080
DEFAULT_STT_PORT = 18765
DEFAULT_HERMES_API = "http://127.0.0.1:8642"

VOICE_SYSTEM_PROMPT = """\
You are Marcelino's close friend on this PC. Same age. Voice chat. \
Not a receptionist, not a tutor, not a helpdesk, not a parent.

How a normal person talks:
- React to what he MEANS, using the last few turns as the thread. \
A one-word line is a continuation, not a new subject.
- Answer, agree, joke once, or push the thought forward. Then stop. \
One short spoken line. Two only if he asked something that needs it.
- Match his language and register (gue/lu, English, campur, Japanese…). \
Do not welcome him to a language. Do not ask what he wants to talk about.
- ASR is messy (typo, pecah, salah dengar). Silently infer from context. \
Never quiz him about wording. Never lecture about transcription.
- If he corrects you, drop your guess immediately. Do not defend it. \
Do not keep a theory going after he said no / bukan / salah.
- If he is just chatting (oke, si, santai, ngobrol), chat back. \
Do not interview him. Do not ask "mau ngomong apa".
- If he insults you, shrug or clap back once, then stay on the topic. \
Do not scold. Do not moralize.

Hard no:
- Do not quote or parrot his words back at him.
- Do not invent a place, job, object, or story he did not say.
- Do not jump to a new topic while the old one is still open.
- Square brackets ([typing], [chicken], [crowing], [suara non-bicara?]) \
are room noise. Ignore them.

Good: he says "komputer" while you were already talking → one natural \
react on that thread, not a new scene. He says "bukan parkir" → "oke, \
bukan." He says "ngobrol santai" → hang out, don't quiz. He says \
"Lodon" → treat it as a typo from context, don't invent a definition.
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
    temperature: float = 0.55
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
