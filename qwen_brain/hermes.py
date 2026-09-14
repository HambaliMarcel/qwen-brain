"""Future Hermes Agent backend. MVP talks to the 4B llama-server directly.

When Hermes gateway API server is running (localhost:8642), `--backend hermes`
forwards the committed utterance as a user chat turn so the 4B can drive
agentic tools. Point Hermes `model.base_url` at this project's llama-server
(`http://127.0.0.1:8080/v1`) so the local 4B is the agent core.
"""

from __future__ import annotations

import json
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import BrainConfig
from .llm import LlamaBrain, apply_llama_timings
from .server import LlamaServerError


class HermesBrain(LlamaBrain):
    """OpenAI-compatible client aimed at Hermes Agent's API server, not llama-server."""

    def __init__(self, cfg: BrainConfig):
        super().__init__(cfg)
        self.base_url = (cfg.hermes_api or "http://127.0.0.1:8642").rstrip("/")

    def _stream(self, messages: list[dict], gen: int = 0, stats=None) -> Iterator[str]:
        payload = {
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": True,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.cfg.hermes_api_key:
            headers["Authorization"] = f"Bearer {self.cfg.hermes_api_key}"
        req = Request(
            self.base_url + "/v1/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )
        buf = ""
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
                    if delta:
                        yield delta
        except HTTPError as e:
            if gen != self._gen:
                return
            body = e.read().decode("utf-8", errors="replace")
            raise LlamaServerError(
                f"Hermes API {self.base_url} HTTP {e.code}: {body[:800]}. "
                "Leave --backend llm for the local 4B, or start Hermes gateway."
            ) from e
        except URLError as e:
            if gen != self._gen:
                return
            raise LlamaServerError(
                f"Hermes API not reachable at {self.base_url}: {e.reason}. "
                "MVP uses --backend llm (local Qwen3.5-4B)."
            ) from e
        except OSError:
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


def make_brain(cfg: BrainConfig) -> LlamaBrain:
    if (cfg.backend or "llm").strip().lower() == "hermes":
        return HermesBrain(cfg)
    return LlamaBrain(cfg)
