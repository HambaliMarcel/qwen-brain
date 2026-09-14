"""Streaming chat client for local llama-server (OpenAI-compatible)."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import BrainConfig, VOICE_SYSTEM_PROMPT
from .metrics import tok_per_sec
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
    decode_ms: float = 0.0
    tokens: int = 0
    chunks: int = 0
    chars: int = 0
    prompt_tokens: int = 0
    prompt_ms: float = 0.0
    predicted_ms: float = 0.0
    tok_s: float = 0.0
    cancelled: bool = False


class LlamaBrain:
    def __init__(self, cfg: BrainConfig):
        self.cfg = cfg
        self.base_url = cfg.url.rstrip("/")
        self.history: list[ChatTurn] = []
        self._gen = 0
        self._resp = None
        self._resp_lock = threading.Lock()

    def cancel(self) -> None:
        """Abort the in-flight HTTP stream so the GPU slot is freed immediately."""
        self._gen += 1
        with self._resp_lock:
            resp = self._resp
            self._resp = None
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass

    def reset(self) -> None:
        self.history.clear()

    def ready(self) -> bool:
        from .server import health

        return health(self.base_url)

    def ask(
        self,
        user_text: str,
        on_token: Optional[Callable[[str, StreamStats], None]] = None,
        *,
        remember: bool = True,
    ) -> tuple[str, StreamStats]:
        self._gen += 1
        gen = self._gen
        prompt = user_text.strip()
        if remember:
            self.history.append(ChatTurn("user", prompt))
            self._trim()
        stats = StreamStats()
        t0 = time.perf_counter()
        t_first: float | None = None
        pieces: list[str] = []
        first = True
        try:
            messages = self._messages() if remember else self._messages(prompt)
            for delta in self._stream(messages, gen, stats):
                if gen != self._gen:
                    break
                if not delta:
                    continue
                now = time.perf_counter()
                if first:
                    stats.first_token_ms = (now - t0) * 1000.0
                    t_first = now
                    first = False
                pieces.append(delta)
                stats.chunks += 1
                stats.chars += len(delta)
                if stats.tokens < stats.chunks:
                    stats.tokens = stats.chunks
                if t_first is not None:
                    decode_s = now - t_first
                    if decode_s > 0.05:
                        stats.decode_ms = decode_s * 1000.0
                        if stats.tok_s <= 0 or not stats.predicted_ms:
                            stats.tok_s = stats.tokens / decode_s
                if on_token is not None:
                    on_token(delta, stats)
        except Exception:
            if gen != self._gen:
                stats.cancelled = True
                stats.total_ms = (time.perf_counter() - t0) * 1000.0
                if remember and self.history and self.history[-1].role == "user":
                    self.history.pop()
                return "", stats
            if not pieces:
                if remember and self.history and self.history[-1].role == "user":
                    self.history.pop()
                raise
        text = strip_think("".join(pieces)).strip()
        stats.total_ms = (time.perf_counter() - t0) * 1000.0
        if stats.decode_ms <= 0 and stats.first_token_ms > 0:
            stats.decode_ms = max(0.0, stats.total_ms - stats.first_token_ms)
        if stats.tok_s <= 0:
            stats.tok_s = tok_per_sec(stats.tokens, stats.decode_ms or stats.total_ms)
        cancelled = gen != self._gen
        stats.cancelled = cancelled
        if cancelled or not text:
            if remember and self.history and self.history[-1].role == "user":
                self.history.pop()
            return text, stats
        if remember:
            self.history.append(ChatTurn("assistant", text))
            self._trim()
        return text, stats

    def remember_turn(self, user_text: str, assistant_text: str) -> None:
        """Commit a previously speculative result to conversation history."""
        user = (user_text or "").strip()
        assistant = (assistant_text or "").strip()
        if not user or not assistant:
            return
        self.history.extend((ChatTurn("user", user), ChatTurn("assistant", assistant)))
        self._trim()

    def _trim(self) -> None:
        cap = max(2, int(self.cfg.history_turns) * 2)
        if len(self.history) > cap:
            self.history = self.history[-cap:]

    def _messages(self, pending_user: str = "") -> list[dict]:
        msgs = [{"role": "system", "content": self.cfg.system_prompt or VOICE_SYSTEM_PROMPT}]
        for turn in self.history:
            msgs.append({"role": turn.role, "content": turn.content})
        if pending_user:
            msgs.append({"role": "user", "content": pending_user})
        return msgs

    def _stream(self, messages: list[dict], gen: int = 0, stats: StreamStats | None = None) -> Iterator[str]:
        payload = {
            "messages": messages,
            "temperature": self.cfg.temperature,
            "top_p": float(getattr(self.cfg, "top_p", 0.8)),
            "top_k": int(getattr(self.cfg, "top_k", 20)),
            "min_p": 0.0,
            "max_tokens": self.cfg.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
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
        resp = None
        try:
            resp = urlopen(req, timeout=45.0)
            with self._resp_lock:
                self._resp = resp
            while gen == self._gen:
                chunk = resp.read(128)
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
                    if stats is not None:
                        apply_llama_timings(stats, evt)
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
            if hold and gen == self._gen:
                yield strip_think(hold)
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise LlamaServerError(f"/v1/chat/completions HTTP {e.code}: {body[:800]}") from e
        except URLError as e:
            if gen != self._gen:
                return
            raise LlamaServerError(f"Cannot reach brain llama-server at {self.base_url}: {e.reason}") from e
        except (OSError, ValueError):
            if gen != self._gen:
                return
            raise
        finally:
            with self._resp_lock:
                if self._resp is resp:
                    self._resp = None
            if resp is not None:
                try:
                    resp.close()
                except Exception:
                    pass


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


def apply_llama_timings(stats: StreamStats, evt: dict) -> None:
    usage = evt.get("usage") or {}
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens")
        comp = usage.get("completion_tokens")
        if prompt:
            stats.prompt_tokens = int(prompt)
        if comp:
            stats.tokens = int(comp)
    timings = evt.get("timings") or {}
    if not isinstance(timings, dict):
        return
    if timings.get("prompt_n"):
        stats.prompt_tokens = int(timings["prompt_n"])
    if timings.get("prompt_ms"):
        stats.prompt_ms = float(timings["prompt_ms"])
    if timings.get("predicted_ms"):
        stats.predicted_ms = float(timings["predicted_ms"])
    n = timings.get("predicted_n") or timings.get("predicted_tokens")
    if n:
        stats.tokens = int(n)
    tps = (
        timings.get("predicted_per_second")
        or timings.get("predicted_n_per_second")
        or timings.get("tokens_predicted_per_second")
    )
    if tps:
        stats.tok_s = float(tps)
