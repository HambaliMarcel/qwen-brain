"""Turn policy + live loop: STT commit (or eager pause) → streamed 27B reply."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from time import monotonic
from typing import Optional

from .bus import SttBusClient
from .config import BrainConfig
from .events import (
    SttEvent,
    collapse_loops,
    is_command_text,
    is_degenerate,
    is_short_fragment,
    is_sound_tag,
    join_fragments,
    looks_complete,
    meaningfully_longer,
    same_turn,
    scene_prompt,
    strip_already_sent,
    strip_language_leak,
)
from .hermes import make_brain
from .llm import LlamaBrain, StreamStats
from .server import fetch_context
from .tts import NullTts, TtsSink
from .ui import BrainUI

STUCK_THINKING_SEC = 6.0
STUCK_ANSWER_SEC = 8.0
STALE_LIVE_SEC = 2.5
LIVE_STABLE_SEC = 0.12
FINALIZE_PAUSE_SEC = 0.45
HOLD_PAUSE_SEC = 1.50
STITCH_SEC = 1.35
EVENT_COOLDOWN_SEC = 2.2


@dataclass
class TurnPolicy:
    """Fire the brain when speech actually ends, not on syllable gaps.

    Hidden speculation may start after a short silence on a complete line.
    A visible YOU is only printed on LAST, or after a real hold with ASR idle.
    Speech and in-flight hops never start a turn.
    """

    eager: bool = True
    eager_silence_sec: float = 0.45
    revise_extra_chars: int = 8
    speculative: bool = True
    hold_silence_sec: float = HOLD_PAUSE_SEC
    sent_uid: int = -1
    sent_text: str = ""
    inflight_uid: int = -1

    def should_ask(self, ev: SttEvent) -> Optional[str]:
        if ev.type == "sound":
            return None
        text = strip_language_leak(ev.text or "")
        if not is_command_text(text):
            return None
        uid = ev.utterance_id
        if ev.type == "commit":
            if uid == self.sent_uid and same_turn(self.sent_text, text):
                if not meaningfully_longer(self.sent_text, text, extra=2):
                    return None
            return self._accept(uid, text)
        if not self.eager or ev.type != "live":
            return None
        if ev.speaking or ev.decoding:
            return None
        pause = float(ev.silence_sec)
        complete = looks_complete(text)
        if complete:
            if pause < self.eager_silence_sec:
                return None
        elif pause < self.hold_silence_sec:
            return None
        if uid == self.sent_uid and same_turn(self.sent_text, text):
            if not meaningfully_longer(self.sent_text, text, extra=self.revise_extra_chars):
                return None
        return self._accept(uid, text)

    def _accept(self, uid: int, text: str) -> str:
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
        self.policy = TurnPolicy(
            eager=cfg.eager,
            eager_silence_sec=cfg.eager_silence_sec,
            revise_extra_chars=int(getattr(cfg, "eager_revise_chars", 8)),
            hold_silence_sec=HOLD_PAUSE_SEC,
        )
        self._latest_prompt = ""
        self._fire_at = 0.0
        self._trigger = ""
        self._stop = threading.Event()
        self._last_event = ""
        self._last_event_at = 0.0
        self._last_stt_at = 0.0
        self._last_token_at = 0.0
        self._stable_text = ""
        self._stable_at = 0.0
        self._job = 0
        self._done_job = 0
        self._job_uid = -1
        self._publish_job = 0
        self._visible_job = 0
        self._result_job = 0
        self._result_prompt = ""
        self._result_text = ""
        self._result_stats = StreamStats()
        self._result_e2e_ms = 0.0
        self._shown_uid = -1
        self._shown_text = ""
        self._shown_job = 0
        self._printed_job = 0
        self._last_sent_event = ""
        self._last_sent_event_at = 0.0
        self._held = ""
        self._held_at = 0.0
        self._held_uid = -1
        self._held_ev: Optional[SttEvent] = None
        self._state_lock = threading.RLock()

    def run(self) -> int:
        self.ui.backend = self.cfg.backend
        self.ui.model = self.cfg.model.name.replace(".gguf", "")
        self.ui.stt_endpoint = f"{self.cfg.stt_host}:{self.cfg.stt_port}"
        self.ui.llm_endpoint = self.cfg.url if self.cfg.backend == "llm" else self.cfg.hermes_api
        self.ui.banner(
            "brain",
            f"STT {self.ui.stt_endpoint}  ·  LLM {self.ui.llm_endpoint}  ·  {self.cfg.backend}",
        )
        threading.Thread(target=self._refresh_context, name="brain-ctx", daemon=True).start()
        threading.Thread(target=self._warm_brain, name="brain-warm", daemon=True).start()
        threading.Thread(target=self._watchdog, name="brain-watchdog", daemon=True).start()
        threading.Thread(target=self._worker, name="brain-worker", daemon=True).start()
        client = SttBusClient(
            self.cfg.stt_host,
            self.cfg.stt_port,
            on_status=self.ui.note_bus,
            on_idle=self.ui.heartbeat,
        )
        try:
            for ev in client.iter_events():
                if self._stop.is_set():
                    break
                self._on_event(ev)
        except KeyboardInterrupt:
            self.brain.cancel()
            self.tts.cancel()
        finally:
            self._stop.set()
            self.brain.cancel()
            client.stop()
            self.ui.close()
        return 0

    def _warm_brain(self) -> None:
        # First real turn otherwise pays ~1 s to prefill the system prompt.
        if self.cfg.backend != "llm":
            return
        try:
            if self.brain.ready() and self._job == 0:
                self.brain.warm()
        except Exception:
            pass

    def _watchdog(self) -> None:
        while not self._stop.wait(0.25):
            now = monotonic()
            self.ui.heartbeat()
            if self._last_stt_at and (now - self._last_stt_at) > STALE_LIVE_SEC:
                if self.ui.status not in {"THINKING", "ANSWERING"} and self.ui.live:
                    self.ui.set_live("")
            thinking = self.ui.status == "THINKING" and self._fire_at > 0
            answering = self.ui.status == "ANSWERING" and self._fire_at > 0
            if thinking and (now - self._fire_at) >= STUCK_THINKING_SEC:
                self._reset_stuck(f"thinking >{int(STUCK_THINKING_SEC)}s, reset")
            elif answering:
                last_tok = self._last_token_at or self._fire_at
                if (now - last_tok) >= STUCK_ANSWER_SEC:
                    self._reset_stuck(f"answer stalled >{int(STUCK_ANSWER_SEC)}s, reset")
            self._maybe_flush_held(now)

    def _reset_stuck(self, msg: str) -> None:
        self.brain.cancel()
        self.tts.cancel()
        self.policy.inflight_uid = -1
        self._job += 1
        self._fire_at = 0.0
        self.ui.note_stuck(msg)

    def _worker(self) -> None:
        while not self._stop.is_set():
            job = self._job
            prompt = self._latest_prompt
            trigger = self._trigger
            if job == self._done_job or not prompt:
                self._stop.wait(0.012)
                continue
            self._execute(prompt, trigger, job)
            if self._job == job:
                self._done_job = job

    def _note_sound(self, ev: SttEvent) -> None:
        event = (ev.event or "").strip().strip("[]")
        if not event:
            return
        self._last_event = event
        self._last_event_at = monotonic()
        self.ui.note_sound(event, score=ev.event_score)

    def _prompt_for(self, text: str) -> str:
        # Room noise stays on STATUS. Do not prefix the 27B turn with it.
        return strip_language_leak(text)

    def _clear_event_context(self) -> None:
        self._last_event = ""
        self._last_event_at = 0.0
        self.ui.event_label = ""

    def _ingest_fragment(self, text: str, ev: SttEvent) -> None:
        """Collect breath-sized LAST pieces until the singer actually stops."""
        body = strip_language_leak(text)
        if not self._held:
            if self._already_shown(ev.utterance_id, body):
                return
            if ev.utterance_id == self.policy.sent_uid and same_turn(self.policy.sent_text, body):
                if not meaningfully_longer(self.policy.sent_text, body, extra=2):
                    return
        joined = join_fragments(self._held, body) if self._held else body
        self._held = joined
        self._held_at = monotonic()
        self._held_uid = ev.utterance_id
        self._held_ev = ev
        if self.ui.status not in {"THINKING", "ANSWERING"}:
            # Live line only: set_you would print a YOU row per held piece.
            self.ui.set_live(joined)
            self.ui.set_status("LISTENING", "holding line")
        if joined and not is_short_fragment(joined) and looks_complete(joined):
            self._flush_held_turn()

    def _flush_held_turn(self) -> None:
        text = strip_language_leak(self._held)
        ev = self._held_ev
        self._held = ""
        self._held_at = 0.0
        self._held_uid = -1
        self._held_ev = None
        if not text or ev is None:
            return
        self._finalize_visible(text, ev, "commit")

    def _maybe_flush_held(self, now: float) -> None:
        if not self._held:
            return
        if self.ui.metrics.speaking:
            return
        if self.ui.status in {"THINKING", "ANSWERING"}:
            return
        if (now - self._held_at) < STITCH_SEC:
            return
        self._flush_held_turn()

    def _already_shown(self, uid: int, text: str) -> bool:
        if uid < 0 or uid != self._shown_uid:
            return False
        return same_turn(self._shown_text, text)

    def _mark_shown(self, uid: int, text: str, job: int) -> None:
        self._shown_uid = uid
        self._shown_text = text
        self._shown_job = job

    def _fire_scene(self, ev: SttEvent) -> None:
        event = (ev.event or "").strip().strip("[]")
        if not event and is_sound_tag(ev.text or ev.display):
            event = (ev.text or ev.display).strip()[1:-1].strip()
        if not event:
            return
        now = monotonic()
        if (
            event.lower() == self._last_sent_event.lower()
            and (now - self._last_sent_event_at) < EVENT_COOLDOWN_SEC
        ):
            return
        prompt = scene_prompt(event)
        if self._already_shown(ev.utterance_id, prompt):
            return
        # Ambient tags stay on STATUS and attach to the next spoken line.
        # They must not start a 27B turn of their own.
        return

    def _finalize_visible(self, text: str, ev: SttEvent, trigger: str) -> None:
        if not is_command_text(text) and not (text or "").lower().startswith("[scene]"):
            return
        if self._already_shown(ev.utterance_id, text):
            return
        prompt = self._prompt_for(text)
        same_speculation = (
            ev.utterance_id == self.policy.sent_uid
            and same_turn(self.policy.sent_text, text)
            and not meaningfully_longer(self.policy.sent_text, text, extra=2)
            and self._job_uid == ev.utterance_id
            and self._job > 0
        )
        self.policy._accept(ev.utterance_id, text)
        self._mark_shown(ev.utterance_id, text, self._job if same_speculation else self._job + 1)
        if same_speculation:
            self._promote_speculation(prompt)
        else:
            self._start_reply(prompt, trigger, uid=ev.utterance_id, publish=True)
        self._clear_event_context()

    def _trigger_name(self, ev: SttEvent) -> str:
        if ev.type == "commit":
            return "commit"
        if ev.speaking:
            return "live"
        return "eager"

    def _refresh_context(self) -> None:
        if self.cfg.backend != "llm":
            return
        try:
            used, n_ctx = fetch_context(self.cfg.url, fallback_ctx=max(2048, int(self.cfg.ctx or 3072)))
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
        live = strip_language_leak(ev.text or "")
        if not is_command_text(live):
            return False
        return not same_turn(self.policy.sent_text, live)

    def _on_event(self, ev: SttEvent) -> None:
        self._last_stt_at = monotonic()
        m = self.ui.metrics
        m.utterance_id = ev.utterance_id
        m.silence_sec = ev.silence_sec
        m.speaking = ev.speaking
        m.decoding = ev.decoding
        if ev.language:
            m.last_language = ev.language
            self.ui.language = ev.language
        if ev.event:
            m.last_event = ev.event
            self._note_sound(ev)
        elif ev.display:
            if is_sound_tag(ev.display) or is_sound_tag(ev.text):
                inner = (ev.display or ev.text).strip()[1:-1].strip()
                if inner:
                    ev.event = inner
                    m.last_event = inner
                    self._note_sound(ev)
        live_text = strip_language_leak(ev.text or "")
        collapsed, looped = collapse_loops(live_text)
        if looped:
            if is_degenerate(live_text):
                # ASR decoder spiral. Never a 27B turn; also never a barge-in.
                if self.ui.status not in {"THINKING", "ANSWERING"}:
                    self.ui.set_status("LISTENING", "asr loop dropped")
                return
            live_text = collapsed
            ev.text = collapsed
        if ev.type == "commit":
            # LAST re-sends the grown paragraph; answer only the unseen part.
            for prev in (self._shown_text, self.policy.sent_text):
                if same_turn(prev, live_text):
                    # Same line grown or repeated: promotion/dedupe below.
                    break
                tail = strip_already_sent(prev, live_text)
                if tail and tail != live_text:
                    live_text = tail
                    ev.text = tail
                    break
        if ev.type == "live" and live_text != self._stable_text:
            self._stable_text = live_text
            self._stable_at = monotonic()
        busy = self.ui.status in {"THINKING", "ANSWERING"}
        if ev.type == "live":
            if ev.speaking and self._should_barge(ev):
                self.brain.cancel()
                self.tts.cancel()
                self.policy.inflight_uid = -1
                self._job += 1
                self._fire_at = 0.0
                if self._visible_job:
                    self.ui.note_cut("barge-in")
            elif not busy:
                if ev.speaking:
                    self.ui.set_status("SPEAKING", ev.language or "mix")
                elif ev.decoding:
                    self.ui.set_status("LISTENING", "STT decoding")
                else:
                    self.ui.set_status("LISTENING", ev.language or "ears on")
        event_only = (
            (bool(ev.event) or is_sound_tag(live_text) or is_sound_tag(ev.display))
            and not is_command_text(live_text)
        )
        if event_only:
            if ev.type in {"sound", "commit"}:
                self._fire_scene(ev)
            return
        if ev.type == "commit" and is_command_text(live_text):
            if self._held or is_short_fragment(live_text):
                self._ingest_fragment(live_text, ev)
                return
            self._finalize_visible(live_text, ev, "commit")
            return
        if (
            ev.type == "live"
            and not ev.speaking
            and not ev.decoding
            and is_command_text(live_text)
            and (self._held or is_short_fragment(live_text))
        ):
            need = (
                self.policy.eager_silence_sec
                if looks_complete(live_text)
                else HOLD_PAUSE_SEC
            )
            if float(ev.silence_sec) >= need:
                self._ingest_fragment(live_text, ev)
                return
        if ev.type == "live" and ev.speaking and looks_complete(live_text):
            if (monotonic() - self._stable_at) < LIVE_STABLE_SEC:
                return
        prompt = self.policy.should_ask(ev)
        if prompt:
            if is_short_fragment(prompt):
                self._ingest_fragment(prompt, ev)
                return
            visible = ev.type == "commit" or (
                not ev.speaking
                and not ev.decoding
                and float(ev.silence_sec) >= HOLD_PAUSE_SEC
            )
            if visible and ev.type != "commit" and self._shown_uid == ev.utterance_id:
                return
            self._start_reply(
                self._prompt_for(prompt),
                self._trigger_name(ev),
                uid=ev.utterance_id,
                publish=visible,
            )
            return
        finished_hold = (
            ev.type == "live"
            and not ev.speaking
            and not ev.decoding
            and float(ev.silence_sec) >= HOLD_PAUSE_SEC
            and is_command_text(live_text)
        )
        if finished_hold:
            if self._held or is_short_fragment(live_text):
                self._ingest_fragment(live_text, ev)
                return
            self._finalize_visible(live_text, ev, "eager")

    def _start_reply(
        self,
        prompt: str,
        trigger: str,
        *,
        uid: int,
        publish: bool,
    ) -> None:
        self._latest_prompt = prompt
        self._trigger = trigger
        self._fire_at = monotonic()
        self._last_token_at = 0.0
        self._job += 1
        self._job_uid = uid
        self.brain.cancel()
        self.tts.cancel()
        self._publish_job = self._job if publish else 0
        self._visible_job = self._job if publish else 0
        self._result_job = 0
        if publish:
            self._mark_shown(uid, prompt, self._job)
        self.ui.trigger = trigger
        self.ui.brain = ""
        if publish:
            self.ui.set_you(prompt)
        self.ui.ttft_ms = 0.0
        self.ui.total_ms = 0.0
        self.ui.decode_ms = 0.0
        self.ui.e2e_ms = 0.0
        self.ui.tok_s = 0.0
        hint = "final · first token…" if publish else "speculating"
        self.ui.set_status("THINKING", hint)

    def _promote_speculation(self, final_prompt: str) -> None:
        """Make the latest hidden generation canonical once speech settles."""
        with self._state_lock:
            job = self._job
            if self._shown_job and self._shown_job != job:
                # A later visible job already owns the log line.
                if self._publish_job and self._publish_job != job:
                    return
            self._publish_job = job
            self._visible_job = job
            self._mark_shown(self._job_uid, final_prompt, job)
            self.ui.trigger = "commit"
            self.ui.set_you(final_prompt)
            if self._result_job == job and self._result_text:
                self._publish_result(
                    final_prompt,
                    self._result_text,
                    self._result_stats,
                    self._result_e2e_ms,
                    "commit",
                    remember=True,
                )
                return
        self.ui.set_status("THINKING", "final · using speculative result")

    def _execute(self, prompt: str, trigger: str, job: int) -> None:
        if job != self._job:
            return
        acc: list[str] = []
        visible = job == self._visible_job

        def on_token(delta: str, stats: StreamStats) -> None:
            if job != self._job:
                return
            acc.append(delta)
            self._last_token_at = monotonic()
            if job == self._visible_job:
                self.tts.on_token(delta)
                e2e = (monotonic() - self._fire_at) * 1000.0
                self.ui.on_stream(delta, stats, e2e_ms=e2e)

        try:
            text, stats = self.brain.ask(
                prompt,
                on_token=on_token,
                remember=visible,
            )
        except Exception as e:
            if job != self._job:
                return
            if visible or job == self._publish_job:
                self.ui.note_error(str(e))
            self.ui.set_status("LISTENING", "error")
            self.policy.inflight_uid = -1
            self._fire_at = 0.0
            return
        if job != self._job:
            return
        e2e = (monotonic() - self._fire_at) * 1000.0
        with self._state_lock:
            self._result_job = job
            self._result_prompt = prompt
            self._result_text = text
            self._result_stats = stats
            self._result_e2e_ms = e2e
            publish = job == self._publish_job
        if publish:
            final_prompt = self.ui.you or prompt
            self._publish_result(
                final_prompt,
                text,
                stats,
                e2e,
                "commit",
                remember=not visible,
            )
        else:
            self._fire_at = 0.0
            self.ui.set_status("LISTENING", "draft ready")

    def _publish_result(
        self,
        prompt: str,
        text: str,
        stats: StreamStats,
        e2e: float,
        trigger: str,
        *,
        remember: bool,
    ) -> None:
        with self._state_lock:
            job = self._job
            if self._shown_job and self._shown_job != job and self._publish_job != job:
                return
            printed = getattr(self, "_printed_job", 0)
            if printed == job and job:
                self._publish_job = 0
                self._visible_job = 0
                self.policy.inflight_uid = -1
                self._fire_at = 0.0
                return
            self._printed_job = job
        if remember and text:
            self.brain.remember_turn(prompt, text)
        self.ui.metrics.record_turn(
            trigger=trigger,
            ttft_ms=stats.first_token_ms,
            gen_ms=stats.total_ms,
            e2e_ms=e2e,
            tok_s=stats.tok_s,
            cancelled=stats.cancelled,
        )
        if text and not self.ui.brain:
            self.ui.set_brain(text)
        self.ui.note_reply(text, stats, e2e_ms=e2e, trigger=trigger)
        threading.Thread(target=self._refresh_context, name="brain-ctx", daemon=True).start()
        if remember and text:
            self.tts.on_token(text)
        self.tts.flush()
        self.policy.inflight_uid = -1
        self._fire_at = 0.0
        self._publish_job = 0
        self._visible_job = 0
        if stats.cancelled:
            self.ui.set_status("LISTENING", "cancelled")
        else:
            self.ui.set_status(
                "LISTENING",
                f"ttft {stats.first_token_ms:.0f}ms  {stats.tok_s:.0f} tok/s",
            )
