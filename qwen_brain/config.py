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
Sharp, loyal, a bit toxic. Not a receptionist, not a tutor, not a \
helpdesk, not a parent.

Obey what he asked. Stay on the last few turns. A one-word line is a \
continuation, not a new subject. Do not invent a better question.

Length: casual chat = one short spoken line (about 15 words), then \
stop. Do not stack two or three sentences on a casual line. If he asks \
for detail, an explanation, steps, or his message is already long, \
give a full useful answer. Do not pad. Do not refuse a long ask \
just because this is voice.

Follow what he means, not the exact words. A one-word line continues \
the last topic. ASR can misspell; do not comment on it.

Match his language and register (gue/lu, English, campur, Japanese…). \
Do not welcome a language. Do not interview him. Do not ask what he \
wants to talk about.

ASR is messy (typo, pecah, salah dengar). Silently infer. Never quiz \
wording. If he says no / bukan / salah, drop the guess immediately.

If he insults you, clap back once, then stay on the topic. Do not \
scold. Do not moralize.

Hard no:
- Do not quote or parrot his words back at him.
- Do not invent a place, job, object, or story he did not say.
- Do not jump to a new topic while the old one is still open.
- Square brackets ([typing], [chicken], [crowing], [suara non-bicara?]) \
are room noise. Ignore them.
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
    ctx: int = 3072
    ngl: int = 99
    stt_host: str = "127.0.0.1"
    stt_port: int = DEFAULT_STT_PORT
    # 48 tokens ≈ one spoken line at ~21 tok/s → ~2 s of generation max.
    max_tokens: int = 48
    max_tokens_long: int = 384
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 20
    kv_type: str = "q4_0"
    batch: int = 256
    ubatch: int = 128
    fit: bool = False
    fit_target_mib: int = 3500
    fit_ctx: int = 2048
    ctx_locked: bool = False
    history_turns: int = 5
    eager_silence_sec: float = 0.45
    eager_revise_chars: int = 8
    eager: bool = True
    backend: str = "llm"  # llm | hermes
    hermes_api: str = DEFAULT_HERMES_API
    hermes_api_key: str = ""
    spec_type: str = "draft-mtp"
    spec_draft_n_max: int = 2
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
            ctx=int(os.environ.get("QWEN_BRAIN_CTX", 3072)),
            ngl=int(os.environ.get("QWEN_BRAIN_NGL", 99)),
            stt_host=os.environ.get("QWEN_BRAIN_STT_HOST", "127.0.0.1"),
            stt_port=int(os.environ.get("QWEN_BRAIN_STT_PORT", DEFAULT_STT_PORT)),
            hermes_api=os.environ.get("HERMES_API", DEFAULT_HERMES_API),
            hermes_api_key=os.environ.get("HERMES_API_KEY", os.environ.get("API_SERVER_KEY", "")),
            backend=os.environ.get("QWEN_BRAIN_BACKEND", "llm").strip().lower(),
            eager_silence_sec=float(os.environ.get("QWEN_BRAIN_EAGER_SILENCE", "0.45")),
            spec_type=os.environ.get("QWEN_BRAIN_SPEC_TYPE", "draft-mtp").strip() or "none",
            spec_draft_n_max=int(os.environ.get("QWEN_BRAIN_SPEC_DRAFT_N_MAX", "2")),
            kv_type=os.environ.get("QWEN_BRAIN_KV", "q4_0").strip() or "q4_0",
            batch=int(os.environ.get("QWEN_BRAIN_BATCH", "256")),
            ubatch=int(os.environ.get("QWEN_BRAIN_UBATCH", "128")),
            fit=os.environ.get("QWEN_BRAIN_FIT", "0").strip().lower() in {"1", "on", "true", "yes"},
            fit_target_mib=int(os.environ.get("QWEN_BRAIN_FIT_TARGET", "3500")),
            fit_ctx=int(os.environ.get("QWEN_BRAIN_FIT_CTX", "2048")),
            ctx_locked="QWEN_BRAIN_CTX" in os.environ,
            max_tokens=int(os.environ.get("QWEN_BRAIN_MAX_TOKENS", "48")),
            max_tokens_long=int(os.environ.get("QWEN_BRAIN_MAX_TOKENS_LONG", "384")),
        )
