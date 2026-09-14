"""Turn policy + live loop: STT commit (or eager pause) → streamed 4B reply."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

from .bus import SttBusClient
from .config import BrainConfig
from .events import SttEvent, is_command_text, same_turn
from .hermes import make_brain
from .llm import LlamaBrain
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
        self.policy = TurnPolicy(eager=cfg.eager, eager_silence_sec=cfg.eager_silence_sec)
        self._gen_lock = threading.Lock()
        self._gen_thread: Optional[threading.Thread] = None
        self._latest_prompt = ""

    def run(self) -> int:
        self.ui.backend = self.cfg.backend
        self.ui.banner(
            "Qwen brain  ·  live STT → 4B",
            f"stt {self.cfg.stt_host}:{self.cfg.stt_port}   llm {self.cfg.url}   {self.cfg.backend}",
        )
        client = SttBusClient(
            self.cfg.stt_host,
            self.cfg.stt_port,
            on_status=lambda msg: self.ui.set_status("WAITING", msg),
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

    def _on_event(self, ev: SttEvent) -> None:
        if ev.type == "live":
            self.ui.set_live(ev.text)
            if ev.speaking:
                self.ui.set_status("SPEAKING", ev.language or "mix")
                if self.policy.inflight_uid >= 0:
                    self.brain.cancel()
                    self.tts.cancel()
                    self.policy.inflight_uid = -1
            elif ev.decoding:
                self.ui.set_status("LISTENING", "STT decoding")
            else:
                self.ui.set_status("LISTENING", ev.language or "ears on")
        elif ev.type == "commit":
            self.ui.set_live(ev.text)
        prompt = self.policy.should_ask(ev)
        if prompt:
            self._start_reply(prompt)

    def _start_reply(self, prompt: str) -> None:
        self._latest_prompt = prompt
        self.brain.cancel()
        self.tts.cancel()
        self.ui.set_you(prompt)
        self.ui.set_brain("")
        self.ui.ttft_ms = 0.0
        self.ui.total_ms = 0.0
        self.ui.set_status("THINKING", "first token…")

        def run() -> None:
            with self._gen_lock:
                if prompt != self._latest_prompt:
                    return
                acc = []

                def on_token(delta: str) -> None:
                    acc.append(delta)
                    self.tts.on_token(delta)
                    self.ui.append_brain(delta)
                    if self.ui.status != "ANSWERING":
                        self.ui.set_status("ANSWERING", "streaming")

                try:
                    text, stats = self.brain.ask(prompt, on_token=on_token)
                except Exception as e:
                    self.ui.set_brain(f"(brain error) {e}")
                    self.ui.set_status("LISTENING", "error")
                    self.policy.inflight_uid = -1
                    return
                if prompt != self._latest_prompt:
                    return
                self.ui.ttft_ms = stats.first_token_ms
                self.ui.total_ms = stats.total_ms
                if text and not acc:
                    self.ui.set_brain(text)
                self.tts.flush()
                self.policy.inflight_uid = -1
                self.ui.set_status("LISTENING", f"ttft {stats.first_token_ms:.0f}ms")

        self._gen_thread = threading.Thread(target=run, name="brain-reply", daemon=True)
        self._gen_thread.start()
