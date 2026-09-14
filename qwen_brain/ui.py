"""Sticky STATUS header + Cindy-style timestamped discussion trail.

LIVE/BRAIN stream in place. YOU/BRAIN/LAT/CUT append below and scroll.
The full screen is never cleared after the first paint.
"""

from __future__ import annotations

import shutil
import sys
import textwrap
import threading
import time
from datetime import datetime

from .llm import StreamStats
from .metrics import SessionMetrics
from .window import apply_light_terminal, console_width

CSI = "\x1b["
RESET = f"{CSI}0m"
PAPER = f"{CSI}107;34m"
NAVY = f"{CSI}34m"
NAVY_B = f"{CSI}1;34m"
LIVE_C = f"{CSI}1;94m"
YOU_C = f"{CSI}1;34m"
BRAIN_C = f"{CSI}34m"
WARN = f"{CSI}31m"
DIM = f"{CSI}2;34m"

ALT_ON = f"{CSI}?1049h"
ALT_OFF = f"{CSI}?1049l"
HIDE = f"{CSI}?25l"
SHOW = f"{CSI}?25h"
HOME = f"{CSI}H"
CLEAR = f"{CSI}2J"
EL = f"{CSI}2K"
SAVE = f"{CSI}s"
RESTORE = f"{CSI}u"

HEADER_ROWS = 11  # title + status box
LIVE_ROW = 12
BRAIN_ROW = 13
RULE_ROW = 14
SCROLL_TOP = 15

LABELS = {
    "YOU": YOU_C,
    "BRAIN": BRAIN_C,
    "LIVE": LIVE_C,
    "LAT": DIM,
    "MODE": DIM,
    "CUT": WARN,
    "BUS": DIM,
    "ERR": WARN,
    "ERROR": WARN,
    "CTX": DIM,
    "STUCK": WARN,
}


def enable_windows_vt() -> None:
    apply_light_terminal("Qwen brain")


def _ms(v: float) -> str:
    if v <= 0:
        return "—"
    if v < 1000:
        return f"{v:.0f}ms"
    return f"{v / 1000.0:.2f}s"


def _num(v: float, unit: str = "") -> str:
    if v <= 0:
        return "—"
    body = f"{v:.0f}" if v >= 10 else f"{v:.1f}"
    return body + unit


def _ctx_bar(used: int, total: int, width: int = 12) -> str:
    if total <= 0:
        return "—"
    frac = min(1.0, max(0.0, used / float(total)))
    n = int(round(width * frac))
    return ("█" * n) + ("░" * (width - n))


class BrainUI:
    def __init__(self) -> None:
        self._started = False
        self.title = "QWEN BRAIN"
        self.detail = "live STT → local 4B"
        self.status = "WAITING"
        self.hint = "waiting for STT bus"
        self.live = ""
        self.you = ""
        self.brain = ""
        self.backend = "llm"
        self.model = "Qwen3.5-4B"
        self.stt_endpoint = ""
        self.llm_endpoint = ""
        self.language = ""
        self.trigger = ""
        self.ttft_ms = 0.0
        self.total_ms = 0.0
        self.decode_ms = 0.0
        self.e2e_ms = 0.0
        self.prompt_ms = 0.0
        self.tok_s = 0.0
        self.tokens = 0
        self.prompt_tokens = 0
        self.chars = 0
        self.ctx_used = 0
        self.ctx_max = 8192
        self.metrics = SessionMetrics()
        self._lock = threading.RLock()
        self._last_header = 0.0
        self._last_live = 0.0
        self._last_brain_row = 0.0
        self._header_sig = ""
        self._live_sig = ""
        self._rows = 42
        self._cols = 56
        self._brain_streaming = False

    def banner(self, title: str, detail: str) -> None:
        apply_light_terminal("Qwen brain")
        self.title = title.split("·")[0].strip().upper() or "QWEN BRAIN"
        self.detail = detail
        try:
            self._cols, self._rows = shutil.get_terminal_size((console_width(), 42))
        except Exception:
            self._cols, self._rows = console_width(), 42
        self._cols = max(48, min(88, self._cols))
        sys.stdout.write(PAPER + ALT_ON + HIDE + CLEAR + HOME)
        self._started = True
        self._paint_header(force=True)
        self._paint_live(force=True)
        self._paint_brain_row("")
        self._paint_rule()
        self._set_scroll()
        self._trail("MODE", "waiting for STT bus")
        sys.stdout.flush()

    def _width(self) -> int:
        return max(48, min(88, self._cols - 1))

    def _goto(self, row: int, col: int = 1) -> None:
        sys.stdout.write(f"{CSI}{row};{col}H")

    def _set_scroll(self) -> None:
        bottom = max(SCROLL_TOP + 4, self._rows)
        sys.stdout.write(f"{CSI}{SCROLL_TOP};{bottom}r")
        self._goto(SCROLL_TOP, 1)

    def _with_fixed(self, fn) -> None:
        sys.stdout.write(SAVE)
        fn()
        sys.stdout.write(RESTORE)
        sys.stdout.flush()

    def _line(self, text: str, color: str = NAVY) -> str:
        w = self._width()
        raw = (text or "").replace("\n", " ")
        if len(raw) < w:
            raw = raw + (" " * (w - len(raw)))
        else:
            raw = raw[: w - 1] + "…"
        return f"{EL}{PAPER}{color}{raw}{RESET}{PAPER}"

    def _paint_header(self, force: bool = False) -> None:
        if not self._started:
            return
        now = time.perf_counter()
        if not force and (now - self._last_header) < 0.12:
            return
        m = self.metrics
        pct = (100.0 * self.ctx_used / self.ctx_max) if self.ctx_max else 0.0
        sig = (
            self.status,
            self.hint,
            self.language,
            m.utterance_id,
            round(m.silence_sec, 1),
            round(self.ttft_ms),
            round(self.total_ms),
            round(self.tok_s),
            round(self.e2e_ms),
            self.tokens,
            self.prompt_tokens,
            self.ctx_used,
            self.ctx_max,
            m.turns,
            m.barge_ins,
            m.errors,
            int(m.uptime_s()),
        )
        if not force and sig == self._header_sig:
            return
        self._header_sig = sig
        self._last_header = now
        st = (self.status or "WAITING").upper()
        w = self._width()
        inner = w - 2

        def box_top(title: str) -> str:
            label = f" {title} "
            fill = max(0, inner - len(label) - 1)
            return f"{NAVY_B}┌{label}{'─' * fill}┐{RESET}{PAPER}"

        def box_bot() -> str:
            return f"{NAVY}└{'─' * inner}┘{RESET}{PAPER}"

        def row(content: str) -> str:
            vis = content
            if len(vis) < inner:
                vis = vis + (" " * (inner - len(vis)))
            else:
                vis = vis[: inner - 1] + "…"
            return f"{NAVY}│{RESET}{PAPER}{vis}{NAVY}│{RESET}{PAPER}"

        lines = [
            f"{PAPER}{NAVY_B}  {self.title}{RESET}{PAPER}  {DIM}{self.detail}{RESET}{PAPER}",
            box_top("STATUS"),
            row(f" MODE  {st:<10}  {self.hint}"),
            row(f" LLM   {self.backend} {self.model}  {self.llm_endpoint}"),
            row(
                f" STT   {self.language or 'mix'}  uid {m.utterance_id}  "
                f"sil {m.silence_sec:.2f}s  {self.stt_endpoint}"
            ),
            row(
                f" TTFT  {_ms(self.ttft_ms)}  GEN {_ms(self.total_ms)}  "
                f"DEC {_ms(self.decode_ms)}  {_num(self.tok_s, ' tok/s')}"
            ),
            row(
                f" E2E   {_ms(self.e2e_ms)}  PREFILL {_ms(self.prompt_ms)}  "
                f"TOK {self.tokens or '—'} / {self.prompt_tokens or '—'}"
            ),
            row(
                f" CTX   {self.ctx_used}/{self.ctx_max}  {pct:.0f}%  "
                f"{_ctx_bar(self.ctx_used, self.ctx_max)}"
            ),
            row(
                f" TURNS {m.turns}  eager {m.eager_turns}  commit {m.commit_turns}  "
                f"cut {m.barge_ins}  err {m.errors}  up {m.uptime_s():.0f}s"
            ),
            row(
                f" AVG   ttft {_ms(m.avg_ttft_ms())}  gen {_ms(m.avg_gen_ms())}  "
                f"e2e {_ms(m.avg_e2e_ms())}  {_num(m.avg_tok_s(), ' tok/s')}"
            ),
            box_bot(),
        ]
        while len(lines) < HEADER_ROWS:
            lines.append(" " * w)

        def paint() -> None:
            self._goto(1, 1)
            for i, ln in enumerate(lines[:HEADER_ROWS], start=1):
                self._goto(i, 1)
                sys.stdout.write(EL + ln)

        self._with_fixed(paint)

    def _paint_live(self, force: bool = False) -> None:
        if not self._started:
            return
        now = time.perf_counter()
        body = (self.live or "").strip()
        sig = body
        if not force and sig == self._live_sig and (now - self._last_live) < 0.03:
            return
        self._live_sig = sig
        self._last_live = now
        stamp = datetime.now().strftime("%H:%M:%S")
        if body:
            text = f"{stamp}  LIVE    {body}▌"
            color = LIVE_C
        else:
            text = f"{stamp}  LIVE    listening…"
            color = DIM

        def paint() -> None:
            self._goto(LIVE_ROW, 1)
            sys.stdout.write(self._line(text, color))

        self._with_fixed(paint)

    def _paint_brain_row(self, text: str) -> None:
        now = time.perf_counter()
        if self._brain_streaming and (now - self._last_brain_row) < 0.04:
            # still update but not faster than 25 Hz
            pass
        if (now - self._last_brain_row) < 0.04 and text:
            return
        self._last_brain_row = now
        stamp = datetime.now().strftime("%H:%M:%S")
        body = (text or "").replace("\n", " ").strip()
        if body:
            line = f"{stamp}  BRAIN   {body}▌"
            color = BRAIN_C
        else:
            line = f"{stamp}  BRAIN   "
            color = DIM

        def paint() -> None:
            self._goto(BRAIN_ROW, 1)
            sys.stdout.write(self._line(line, color))

        self._with_fixed(paint)

    def _paint_rule(self) -> None:
        w = self._width()
        rule = f"── discussion ── {DIM}timestamped trail{RESET}{PAPER} "
        fill = max(0, w - 22)
        text = rule + ("─" * fill)

        def paint() -> None:
            self._goto(RULE_ROW, 1)
            sys.stdout.write(f"{EL}{PAPER}{NAVY}{text[:w]}{RESET}{PAPER}")

        self._with_fixed(paint)

    def _trail(self, label: str, body: str) -> None:
        if not self._started:
            return
        text = (body or "").strip().replace("\n", " ")
        if not text:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        color = LABELS.get(label.upper(), NAVY)
        prefix = f"{stamp}  {label:<6}  "
        width = max(24, self._width() - len(prefix))
        wrapped = textwrap.wrap(
            text,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [text]
        with self._lock:
            sys.stdout.write(f"{CSI}{SCROLL_TOP};{max(SCROLL_TOP + 4, self._rows)}r")
            for i, chunk in enumerate(wrapped):
                lead = prefix if i == 0 else " " * len(prefix)
                sys.stdout.write(f"{PAPER}{color}{lead}{chunk}{RESET}{PAPER}\n")
            sys.stdout.flush()

    def set_status(self, status: str, hint: str = "") -> None:
        prev = self.status
        self.status = status
        if hint:
            self.hint = hint
        if status != prev and status in {"WAITING", "THINKING", "ANSWERING"}:
            self._trail("MODE", f"{status.lower()}  {hint}".strip())
        self._paint_header()

    def set_live(self, text: str) -> None:
        incoming = text or ""
        if incoming == self.live:
            return
        self.live = incoming
        self.metrics.stt_live_updates += 1
        self._paint_live()

    def set_you(self, text: str) -> None:
        self.you = text or ""
        self._trail("YOU", self.you)
        self._paint_header(force=True)

    def set_brain(self, text: str) -> None:
        self.brain = text or ""
        self._paint_brain_row(self.brain)

    def on_stream(self, delta: str, stats: StreamStats, *, e2e_ms: float = 0.0) -> None:
        self.brain += delta
        self._brain_streaming = True
        if self.status != "ANSWERING":
            self.status = "ANSWERING"
            self.hint = "streaming"
        self.ttft_ms = stats.first_token_ms
        self.total_ms = stats.total_ms
        self.decode_ms = stats.decode_ms
        self.prompt_ms = stats.prompt_ms
        self.tok_s = stats.tok_s
        self.tokens = stats.tokens
        self.prompt_tokens = stats.prompt_tokens
        self.chars = stats.chars
        if e2e_ms:
            self.e2e_ms = e2e_ms
        self._paint_brain_row(self.brain)
        self._paint_header()

    def update_gen(self, stats: StreamStats, *, e2e_ms: float = 0.0) -> None:
        self.ttft_ms = stats.first_token_ms
        self.total_ms = stats.total_ms
        self.decode_ms = stats.decode_ms
        self.prompt_ms = stats.prompt_ms
        self.tok_s = stats.tok_s
        self.tokens = stats.tokens
        self.prompt_tokens = stats.prompt_tokens
        self.chars = stats.chars
        if e2e_ms:
            self.e2e_ms = e2e_ms
        if stats.prompt_tokens or stats.tokens:
            used = int(stats.prompt_tokens) + int(stats.tokens)
            if used > self.ctx_used:
                self.ctx_used = used
        self._paint_header(force=True)

    def set_context(self, used: int, n_ctx: int) -> None:
        if n_ctx > 0:
            self.ctx_max = n_ctx
        if used >= 0:
            self.ctx_used = used
        self._paint_header()

    def note_reply(self, text: str, stats: StreamStats, *, e2e_ms: float, trigger: str) -> None:
        self._brain_streaming = False
        if text:
            self.brain = text
        self.update_gen(stats, e2e_ms=e2e_ms)
        self.trigger = trigger
        self._paint_brain_row("")
        if text:
            self._trail("BRAIN", text)
        lat = (
            f"ttft {_ms(stats.first_token_ms)}  gen {_ms(stats.total_ms)}  "
            f"{_num(stats.tok_s, ' tok/s')}  e2e {_ms(e2e_ms)}  "
            f"tok {stats.tokens}/{stats.prompt_tokens or '—'}  "
            f"ctx {self.ctx_used}/{self.ctx_max}  {trigger}"
        )
        self._trail("LAT", lat)

    def note_cut(self, reason: str = "barge-in") -> None:
        self.metrics.barge_ins += 1
        self._brain_streaming = False
        self._paint_brain_row("")
        self._trail("CUT", reason)
        self._paint_header(force=True)

    def note_error(self, msg: str) -> None:
        self.metrics.errors += 1
        self._brain_streaming = False
        self._trail("ERROR", msg)
        self._paint_header(force=True)

    def note_stuck(self, msg: str) -> None:
        self.metrics.errors += 1
        self._brain_streaming = False
        self._trail("STUCK", msg)
        self.status = "LISTENING"
        self.hint = "watchdog reset"
        self._paint_header(force=True)

    def note_bus(self, msg: str) -> None:
        if "waiting" in (msg or "").lower():
            self.metrics.bus_reconnects += 1
        self._trail("BUS", msg)
        self.status = "WAITING"
        self.hint = msg
        self._paint_header(force=True)

    def close(self, final: str = "") -> None:
        if self._started:
            sys.stdout.write(f"{CSI}r")  # reset scroll region
            sys.stdout.write(SHOW + ALT_OFF + RESET)
            self._started = False
            sys.stdout.flush()
        text = final or self.brain
        if text:
            print()
            print(f"YOU    {self.you}")
            print(f"BRAIN  {text}")
            if self.ttft_ms:
                print(
                    f"LAT    ttft {_ms(self.ttft_ms)}  gen {_ms(self.total_ms)}  "
                    f"{_num(self.tok_s, ' tok/s')}  e2e {_ms(self.e2e_ms)}  "
                    f"ctx {self.ctx_used}/{self.ctx_max}"
                )
            print()
