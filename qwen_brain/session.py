"""Turn policy + live loop: STT commit (or eager pause) → streamed 4B reply."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from time import monotonic
from typing import Optional

from .bus import SttBusClient
from .config import BrainConfig
from .events import SttEvent, is_command_text, same_turn
from .hermes import make_brain
from .llm import LlamaBrain, StreamStats
from .server import fetch_context
from .tts import NullTts, TtsSink
from .ui import BrainUI


@dataclass
class TurnPolicy:
    """Fire the brain as soon as a command is usable.

    Primary trigger is ASR LAST commit. Eager trigger uses the integrator's
    silence_sec so the 4B can start ~1s earlier than the official pause.
    """

    eager: bool = True
    eager_silence_sec: float = 0.55
    sent_uid: int = -1
    sent_text: str = ""
    inflight_uid: int = -1

    def should_ask(self, ev: SttEvent) -> Optional[str]:
        text = (ev.text or "").strip()
        if not is_command_text(text):
            return None
        uid = ev.utterance_id
        if ev.type == "commit":
            if uid == self.sent_uid and same_turn(self.sent_text, text):
                if len(text) <= len(self.sent_text) + 2:
                    return None
            self.sent_uid = uid
            self.sent_text = text
            self.inflight_uid = uid
            return text
        if not self.eager or ev.type != "live":
            return None
        if ev.speaking or ev.decoding:
            return None
        if ev.silence_sec < self.eager_silence_sec:
            return None
        if uid == self.sent_uid and same_turn(self.sent_text, text):
            return None
        self.sent_uid = uid
        self.sent_text = text
        self.inflight_uid = uid
        return text


class AssistantSession:
    def __init__(self, cfg: BrainConfig, brain: Optional[LlamaBrain] = None, tts: Optional[TtsSink] = None):
        self.cfg = cfg
        self.brain = brain or make_brain(cfg)
        self.tts = tts or NullTts()
        self.ui = BrainUI()
        self.ui.ctx_max = int(cfg.ctx)
        self.policy = TurnPolicy(eager=cfg.eager, eager_silence_sec=cfg.eager_silence_sec)
        self._gen_lock = threading.Lock()
        self._gen_thread: Optional[threading.Thread] = None
        self._latest_prompt = ""
        self._fire_at = 0.0
        self._trigger = ""

    def run(self) -> int:
        self.ui.backend = self.cfg.backend
        self.ui.model = self.cfg.model.name.replace(".gguf", "")
        self.ui.stt_endpoint = f"{self.cfg.stt_host}:{self.cfg.stt_port}"
        self.ui.llm_endpoint = self.cfg.url if self.cfg.backend == "llm" else self.cfg.hermes_api
        self.ui.banner(
            "Qwen brain",
            f"STT {self.ui.stt_endpoint}  ·  LLM {self.ui.llm_endpoint}  ·  {self.cfg.backend}",
        )
        self._refresh_context()
        client = SttBusClient(
            self.cfg.stt_host,
            self.cfg.stt_port,
            on_status=self.ui.note_bus,
        )
        try:
            for ev in client.iter_events():
                self._on_event(ev)
        except KeyboardInterrupt:
            self.brain.cancel()
            self.tts.cancel()
        finally:
            client.stop()
            self.ui.close()
        return 0

    def _refresh_context(self) -> None:
        if self.cfg.backend != "llm":
            return
        try:
            used, n_ctx = fetch_context(self.cfg.url, fallback_ctx=int(self.cfg.ctx))
            if used <= 0:
                used = int(self.ui.prompt_tokens) + int(self.ui.tokens)
            self.ui.set_context(used, n_ctx)
        except Exception:
            pass

    def _should_barge(self, ev: SttEvent) -> bool:
        if self.policy.inflight_uid < 0:
            return False
        if ev.utterance_id != self.policy.inflight_uid:
            return True
        live = (ev.text or "").strip()
        if not is_command_text(live):
            return False
        return not same_turn(self.policy.sent_text, live)

    def _on_event(self, ev: SttEvent) -> None:
        m = self.ui.metrics
        m.utterance_id = ev.utterance_id
        m.silence_sec = ev.silence_sec
        m.speaking = ev.speaking
        m.decoding = ev.decoding
        if ev.language:
            m.last_language = ev.language
            self.ui.language = ev.language
        busy = self.ui.status in {"THINKING", "ANSWERING"}
        if ev.type == "live":
            self.ui.set_live(ev.text)
            if ev.speaking and self._should_barge(ev):
                self.brain.cancel()
                self.tts.cancel()
                self.policy.inflight_uid = -1
                self.ui.note_cut("barge-in")
            elif not busy:
                if ev.speaking:
                    self.ui.set_status("SPEAKING", ev.language or "mix")
                elif ev.decoding:
                    self.ui.set_status("LISTENING", "STT decoding")
                else:
                    self.ui.set_status("LISTENING", ev.language or "ears on")
            else:
                self.ui._paint_header()
        elif ev.type == "commit":
            self.ui.set_live(ev.text)
        prompt = self.policy.should_ask(ev)
        if prompt:
            trigger = "commit" if ev.type == "commit" else "eager"
            self._start_reply(prompt, trigger)

    def _start_reply(self, prompt: str, trigger: str) -> None:
        self._latest_prompt = prompt
        self._trigger = trigger
        self._fire_at = monotonic()
        self.brain.cancel()
        self.tts.cancel()
        self.ui.trigger = trigger
        self.ui.set_you(prompt)
        self.ui.set_brain("")
        self.ui.ttft_ms = 0.0
        self.ui.total_ms = 0.0
        self.ui.decode_ms = 0.0
        self.ui.e2e_ms = 0.0
        self.ui.tok_s = 0.0
        self.ui.set_status("THINKING", f"{trigger} · first token…")

        def run() -> None:
            with self._gen_lock:
                if prompt != self._latest_prompt:
                    return
                acc: list[str] = []

                def on_token(delta: str, stats: StreamStats) -> None:
                    acc.append(delta)
                    self.tts.on_token(delta)
                    e2e = (monotonic() - self._fire_at) * 1000.0
                    self.ui.on_stream(delta, stats, e2e_ms=e2e)

                try:
                    text, stats = self.brain.ask(prompt, on_token=on_token)
                except Exception as e:
                    self.ui.note_error(str(e))
                    self.ui.set_status("LISTENING", "error")
                    self.policy.inflight_uid = -1
                    self._fire_at = 0.0
                    return
                if prompt != self._latest_prompt:
                    return
                e2e = (monotonic() - self._fire_at) * 1000.0
                self.ui.metrics.record_turn(
                    trigger=trigger,
                    ttft_ms=stats.first_token_ms,
                    gen_ms=stats.total_ms,
                    e2e_ms=e2e,
                    tok_s=stats.tok_s,
                    cancelled=stats.cancelled,
                )
                if text and not acc:
                    self.ui.set_brain(text)
                self.ui.note_reply(text, stats, e2e_ms=e2e, trigger=trigger)
                self._refresh_context()
                self.tts.flush()
                self.policy.inflight_uid = -1
                self._fire_at = 0.0
                if stats.cancelled:
                    self.ui.set_status("LISTENING", "cancelled")
                else:
                    self.ui.set_status(
                        "LISTENING",
                        f"ttft {stats.first_token_ms:.0f}ms  {stats.tok_s:.0f} tok/s",
                    )

        self._gen_thread = threading.Thread(target=run, name="brain-reply", daemon=True)
        self._gen_thread.start()
