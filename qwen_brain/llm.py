"""Streaming chat client for local llama-server (OpenAI-compatible)."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import BrainConfig, VOICE_SYSTEM_PROMPT
from .server import LlamaServerError

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


@dataclass
class ChatTurn:
    role: str
    content: str


@dataclass
class StreamStats:
    first_token_ms: float = 0.0
    total_ms: float = 0.0
    tokens: int = 0


class LlamaBrain:
    def __init__(self, cfg: BrainConfig):
        self.cfg = cfg
        self.base_url = cfg.url.rstrip("/")
        self.history: list[ChatTurn] = []
        self._gen = 0

    def cancel(self) -> None:
        self._gen += 1

    def reset(self) -> None:
        self.history.clear()

    def ready(self) -> bool:
        from .server import health

        return health(self.base_url)

    def ask(
        self,
        user_text: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, StreamStats]:
        self._gen += 1
        gen = self._gen
        self.history.append(ChatTurn("user", user_text.strip()))
        self._trim()
        stats = StreamStats()
        t0 = time.perf_counter()
        pieces: list[str] = []
        first = True
        try:
            for delta in self._stream(self._messages(), gen):
                if gen != self._gen:
                    break
                if not delta:
                    continue
                if first:
                    stats.first_token_ms = (time.perf_counter() - t0) * 1000.0
                    first = False
                pieces.append(delta)
                stats.tokens += 1
                if on_token is not None:
                    on_token(delta)
        except Exception:
            if not pieces:
                if self.history and self.history[-1].role == "user":
                    self.history.pop()
                raise
        text = strip_think("".join(pieces)).strip()
        stats.total_ms = (time.perf_counter() - t0) * 1000.0
        cancelled = gen != self._gen
        if cancelled or not text:
            if self.history and self.history[-1].role == "user":
                self.history.pop()
            return text, stats
        self.history.append(ChatTurn("assistant", text))
        self._trim()
        return text, stats

    def _trim(self) -> None:
        cap = max(2, int(self.cfg.history_turns) * 2)
        if len(self.history) > cap:
            self.history = self.history[-cap:]

    def _messages(self) -> list[dict]:
        msgs = [{"role": "system", "content": self.cfg.system_prompt or VOICE_SYSTEM_PROMPT}]
        for turn in self.history:
            msgs.append({"role": turn.role, "content": turn.content})
        return msgs

    def _stream(self, messages: list[dict], gen: int = 0) -> Iterator[str]:
        payload = {
            "messages": messages,
            "temperature": self.cfg.temperature,
            "top_p": 0.9,
            "max_tokens": self.cfg.max_tokens,
            "stream": True,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_budget": 0,
        }
        data = json.dumps(payload).encode("utf-8")
        req = Request(
            self.base_url + "/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        buf = ""
        hold = ""
        try:
            with urlopen(req, timeout=120.0) as resp:
                while gen == self._gen:
                    chunk = resp.read(256)
                    if not chunk:
                        break
                    buf += chunk.decode("utf-8", errors="replace")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        payload_s = line[5:].strip()
                        if payload_s == "[DONE]":
                            if hold:
                                yield hold
                            return
                        try:
                            evt = json.loads(payload_s)
                        except json.JSONDecodeError:
                            continue
                        choices = evt.get("choices") or []
                        if not choices:
                            continue
                        delta = (choices[0].get("delta") or {}).get("content") or ""
                        if not delta:
                            continue
                        hold += delta
                        hold, emit = split_think_stream(hold)
                        if emit:
                            yield emit
                if hold:
                    yield strip_think(hold)
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise LlamaServerError(f"/v1/chat/completions HTTP {e.code}: {body[:800]}") from e
        except URLError as e:
            raise LlamaServerError(f"Cannot reach brain llama-server at {self.base_url}: {e.reason}") from e


def strip_think(text: str) -> str:
    t = THINK_RE.sub("", text or "")
    t = t.replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
    return t.strip()


def split_think_stream(buf: str) -> tuple[str, str]:
    """Hold tokens that are inside an unclosed <think> block."""
    if THINK_OPEN not in buf and THINK_CLOSE not in buf:
        return "", buf
    out: list[str] = []
    rest = buf
    while True:
        start = rest.find(THINK_OPEN)
        if start < 0:
            out.append(rest)
            return "", "".join(out)
        out.append(rest[:start])
        end = rest.find(THINK_CLOSE, start)
        if end < 0:
            return rest[start:], "".join(out)
        rest = rest[end + len(THINK_CLOSE) :]
