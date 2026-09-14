"""Sticky STATUS header + Cindy-style YOU / BRAIN turns.

LIVE is inlined on the YOU line (yellow draft, black final). BRAIN streams
on the next line. LAT is always the last screen row, rewritten in place.
The log region is painted from a turn buffer so lines never collide.
"""

from __future__ import annotations

import shutil
import sys
import textwrap
import threading
import time
from datetime import datetime

from .events import with_sound_context
from .llm import StreamStats
from .metrics import SessionMetrics
from .window import apply_light_terminal, console_width

CSI = "\x1b["
RESET = f"{CSI}0m"
PAPER = f"{CSI}107;34m"
NAVY = f"{CSI}34m"
NAVY_B = f"{CSI}1;34m"
LIVE_C = f"{CSI}1;33m"  # yellow live draft
YOU_C = f"{CSI}30m"  # black committed words on paper
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

HEADER_ROWS = 11
SCROLL_TOP = 12
MAX_TURNS = 80


def enable_windows_vt() -> None:
    apply_light_terminal("Qwen brain")


def _ms(v: float) -> str:
    if v <= 0:
        return "-"
    if v < 1000:
        return f"{v:.0f}ms"
    return f"{v / 1000.0:.2f}s"


def _num(v: float, unit: str = "") -> str:
    if v <= 0:
        return "-"
    body = f"{v:.0f}" if v >= 10 else f"{v:.1f}"
    return body + unit


def _ctx_bar(used: int, total: int, width: int = 12) -> str:
    if total <= 0:
        return "-"
    frac = min(1.0, max(0.0, used / float(total)))
    n = int(round(width * frac))
    return ("█" * n) + ("░" * (width - n))


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def you_parts(final: str, live: str, event: str = "") -> tuple[str, str]:
    """Black committed words sent to the brain, plus yellow live remainder."""
    black = with_sound_context(final, event) if (final or "").strip() else ""
    full = with_sound_context((live or final or "").strip(), event)
    if not black:
        return "", full
    if full.lower().startswith(black.lower()):
        return black, full[len(black) :].strip()
    if black.lower().startswith(full.lower()) and full:
        return full, ""
    return black, ""


class BrainUI:
    def __init__(self) -> None:
        self._started = False
        self.title = "QWEN BRAIN"
        self.detail = "live STT -> local 4B"
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
        self.event_label = ""
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
        self._last_brain_row = 0.0
        self._lat_stamp = ""
        self._header_sig = ""
        self._frame_sig = ""
        self._rows = 42
        self._cols = 56
        self._brain_streaming = False
        self._last_beat = 0.0
        self._you_open = False
        self._brain_open = False
        self._you_stamp = ""
        self._brain_stamp = ""
        self._turns: list[tuple[str, str, str]] = []
        self._lat_text = "ttft -  gen -  tok/s -  e2e -  ctx -"

    def banner(self, title: str, detail: str) -> None:
        apply_light_terminal("Qwen brain")
        self.title = title.split("·")[0].strip().upper() or "QWEN BRAIN"
        self.detail = detail
        self._refresh_geom()
        sys.stdout.write(PAPER + ALT_ON + HIDE + f"{CSI}r" + CLEAR + HOME)
        self._started = True
        self._paint_header(force=True)
        self._paint_frame(force=True)
        sys.stdout.flush()

    def _refresh_geom(self) -> None:
        try:
            cols, rows = shutil.get_terminal_size((console_width(), 42))
        except Exception:
            cols, rows = console_width(), 42
        self._cols = max(48, min(88, cols))
        self._rows = max(HEADER_ROWS + 4, rows)

    def _width(self) -> int:
        return max(48, min(88, self._cols - 1))

    def _lat_row(self) -> int:
        return self._rows

    def _log_bottom(self) -> int:
        return max(SCROLL_TOP, self._lat_row() - 1)

    def _log_height(self) -> int:
        return max(1, self._log_bottom() - SCROLL_TOP + 1)

    def _goto(self, row: int, col: int = 1) -> None:
        sys.stdout.write(f"{CSI}{row};{col}H")

    def _with_fixed(self, fn) -> None:
        with self._lock:
            fn()
            sys.stdout.flush()

    def _blank(self) -> str:
        return f"{EL}{PAPER}{' ' * self._width()}{RESET}{PAPER}"

    def _line(self, text: str, color: str = NAVY) -> str:
        w = self._width()
        raw = (text or "").replace("\n", " ")
        if len(raw) < w:
            raw = raw + (" " * (w - len(raw)))
        else:
            raw = raw[: w - 1] + "…"
        return f"{EL}{PAPER}{color}{raw}{RESET}{PAPER}"

    def _prefix(self, stamp: str, label: str) -> str:
        return f"{stamp}  {label:<6}  "

    def _paint_header(self, force: bool = False) -> None:
        if not self._started:
            return
        now = time.perf_counter()
        if not force and (now - self._last_header) < 0.12:
            return
        self._refresh_geom()
        m = self.metrics
        pct = (100.0 * self.ctx_used / self.ctx_max) if self.ctx_max else 0.0
        sig = (
            self.status,
            self.hint,
            self.language,
            self.event_label,
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
            self._rows,
            self._cols,
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

        ev = f"  [{self.event_label}]" if self.event_label else ""
        lines = [
            f"{PAPER}{NAVY_B}  {self.title}{RESET}{PAPER}  {DIM}{self.detail}{RESET}{PAPER}",
            box_top("STATUS"),
            row(f" MODE  {st:<10}  {self.hint}"),
            row(f" LLM   {self.backend} {self.model}  {self.llm_endpoint}"),
            row(
                f" STT   {self.language or 'mix'}  uid {m.utterance_id}  "
                f"sil {m.silence_sec:.2f}s{ev}"
            ),
            row(
                f" TTFT  {_ms(self.ttft_ms)}  GEN {_ms(self.total_ms)}  "
                f"DEC {_ms(self.decode_ms)}  {_num(self.tok_s, ' tok/s')}"
            ),
            row(
                f" E2E   {_ms(self.e2e_ms)}  PREFILL {_ms(self.prompt_ms)}  "
                f"TOK {self.tokens or '-'} / {self.prompt_tokens or '-'}"
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
            for i, ln in enumerate(lines[:HEADER_ROWS], start=1):
                self._goto(i, 1)
                sys.stdout.write(EL + ln)

        self._with_fixed(paint)

    def _wrap_turn(self, stamp: str, label: str, text: str) -> list[tuple[str, str, str]]:
        prefix = self._prefix(stamp, label)
        body = (text or "").replace("\n", " ").strip()
        width = max(12, self._width() - len(prefix))
        chunks = textwrap.wrap(
            body,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        rows: list[tuple[str, str, str]] = []
        for i, chunk in enumerate(chunks):
            lead = prefix if i == 0 else " " * len(prefix)
            rows.append((lead, chunk, label))
        return rows

    def _color_body(self, label: str, body: str) -> str:
        if label == "YOU":
            return f"{YOU_C}{body}{RESET}{PAPER}"
        if label == "BRAIN":
            return f"{BRAIN_C}{body}{RESET}{PAPER}"
        if label == "ERROR":
            return f"{WARN}{body}{RESET}{PAPER}"
        return f"{NAVY}{body}{RESET}{PAPER}"

    def _open_you_row(self) -> tuple[str, str] | None:
        black, draft = you_parts(self.you, self.live, self.event_label)
        if not black and not draft:
            return None
        if not self._you_stamp:
            self._you_stamp = _now()
        prefix = self._prefix(self._you_stamp, "YOU")
        w = self._width()
        room = max(8, w - len(prefix) - 1)
        cursor = "▌" if draft or not black else ""
        if len(black) + (1 if black and draft else 0) + len(draft) + len(cursor) > room:
            draft = ""
            cursor = ""
            if len(black) > room:
                black = black[: room - 1] + "…"
        colored = f"{NAVY_B}{prefix}{RESET}{PAPER}"
        if black:
            colored += f"{YOU_C}{black}{RESET}{PAPER}"
            if draft:
                colored += " "
        if draft:
            colored += f"{LIVE_C}{draft}{RESET}{PAPER}"
        if cursor:
            colored += f"{LIVE_C}▌{RESET}{PAPER}"
        plain = prefix + (black + (" " if black and draft else "") + draft) + cursor
        return colored, plain

    def _open_brain_row(self) -> tuple[str, str] | None:
        body = (self.brain or "").replace("\n", " ").strip()
        if not body and not self._brain_streaming:
            return None
        if not self._brain_stamp:
            self._brain_stamp = _now()
        prefix = self._prefix(self._brain_stamp, "BRAIN")
        w = self._width()
        room = max(8, w - len(prefix) - 1)
        cursor = self._brain_streaming and bool(body)
        shown = body
        if len(shown) > room:
            shown = "…" + shown[-(room - 1) :]
        if cursor and len(shown) >= room:
            shown = shown[: room - 1]
        colored = (
            f"{NAVY_B}{prefix}{RESET}{PAPER}"
            f"{BRAIN_C}{shown}{RESET}{PAPER}"
        )
        if cursor:
            colored += f"{BRAIN_C}▌{RESET}{PAPER}"
        plain = prefix + shown + ("▌" if cursor else "")
        return colored, plain

    def _frame_rows(self) -> list[str]:
        visual: list[str] = []
        for kind, stamp, text in self._turns:
            for lead, chunk, label in self._wrap_turn(stamp, kind, text):
                colored = f"{NAVY_B}{lead}{RESET}{PAPER}{self._color_body(label, chunk)}"
                visual.append(self._pad(colored, lead + chunk))
        you_row = self._open_you_row() if self._you_open else None
        if you_row:
            visual.append(self._pad(you_row[0], you_row[1]))
        brain_row = self._open_brain_row() if self._brain_open else None
        if brain_row:
            visual.append(self._pad(brain_row[0], brain_row[1]))
        height = self._log_height()
        if len(visual) > height:
            visual = visual[-height:]
        while len(visual) < height:
            visual.append(self._blank())
        return visual

    def _pad(self, colored: str, plain: str) -> str:
        w = self._width()
        vis = len(plain.replace("\n", " "))
        extra = max(0, w - vis)
        return f"{EL}{PAPER}{colored}{' ' * extra}{RESET}{PAPER}"

    def _paint_frame(self, force: bool = False) -> None:
        if not self._started:
            return
        lat = self._lat_body()
        you = you_parts(self.you, self.live, self.event_label) if self._you_open else ("", "")
        sig = (
            tuple(self._turns[-12:]),
            you,
            self.brain if self._brain_open else "",
            self._brain_streaming,
            self._you_open,
            self._brain_open,
            lat,
            self._rows,
            self._cols,
        )
        if not force and sig == self._frame_sig:
            return
        self._frame_sig = sig
        rows = self._frame_rows()
        if lat != self._lat_text or not self._lat_stamp:
            self._lat_stamp = _now()
            self._lat_text = lat
        lat_line = self._line(f"{self._lat_stamp}  LAT     {lat}", DIM)

        def paint() -> None:
            for i, ln in enumerate(rows):
                self._goto(SCROLL_TOP + i, 1)
                sys.stdout.write(ln)
            self._goto(self._lat_row(), 1)
            sys.stdout.write(lat_line)

        self._with_fixed(paint)

    def _commit_open_you(self) -> None:
        if not self._you_open:
            return
        black, _draft = you_parts(self.you, self.live, self.event_label)
        body = black or with_sound_context(self.you or self.live, self.event_label)
        if body:
            self._turns.append(("YOU", self._you_stamp or _now(), body))
            if len(self._turns) > MAX_TURNS:
                self._turns = self._turns[-MAX_TURNS:]
        self._you_open = False
        self._you_stamp = ""

    def _commit_open_brain(self, text: str = "") -> None:
        body = (text or self.brain or "").replace("\n", " ").strip()
        if self._brain_open and body:
            self._turns.append(("BRAIN", self._brain_stamp or _now(), body))
            if len(self._turns) > MAX_TURNS:
                self._turns = self._turns[-MAX_TURNS:]
        self._brain_open = False
        self._brain_stamp = ""
        self._brain_streaming = False

    def _lat_body(self) -> str:
        return (
            f"ttft {_ms(self.ttft_ms)}  gen {_ms(self.total_ms)}  "
            f"{_num(self.tok_s, ' tok/s')}  e2e {_ms(self.e2e_ms)}  "
            f"tok {self.tokens or '-'}/{self.prompt_tokens or '-'}  "
            f"ctx {self.ctx_used}/{self.ctx_max}"
            + (f"  {self.trigger}" if self.trigger else "")
        )

    def _paint_lat(self, force: bool = False) -> None:
        self._paint_frame(force=force)

    def set_status(self, status: str, hint: str = "") -> None:
        self.status = status
        if hint:
            self.hint = hint
        self._paint_header()

    def set_live(self, text: str) -> None:
        incoming = text or ""
        if incoming == self.live:
            return
        self.live = incoming
        self.metrics.stt_live_updates += 1
        if self._brain_open:
            return
        if incoming and not self._you_open:
            self._you_open = True
            self._you_stamp = _now()
            self.you = ""
        elif not incoming and not self.you:
            self._you_open = False
            self._you_stamp = ""
        self._paint_frame()

    def set_you(self, text: str) -> None:
        self.you = text or ""
        if self._brain_open:
            self._commit_open_brain(self.brain)
        if not self._you_open:
            self._you_stamp = _now()
            self._you_open = True
        self._paint_frame(force=True)
        self._paint_header(force=True)

    def set_brain(self, text: str) -> None:
        body = (text or "").strip()
        if not body:
            return
        self.brain = body
        self._commit_open_you()
        if not self._brain_open:
            self._brain_stamp = _now()
            self._brain_open = True
        self._paint_frame(force=True)

    def on_stream(self, delta: str, stats: StreamStats, *, e2e_ms: float = 0.0) -> None:
        now = time.perf_counter()
        first = not self._brain_open
        self._brain_streaming = True
        if first:
            self._commit_open_you()
            self._brain_stamp = _now()
            self._brain_open = True
            self.brain = delta
        else:
            self.brain += delta
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
        if first or (now - self._last_brain_row) >= 0.04:
            self._last_brain_row = now
            self._paint_frame()
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
        self._paint_frame(force=True)

    def set_context(self, used: int, n_ctx: int) -> None:
        if n_ctx > 0:
            self.ctx_max = n_ctx
        if used >= 0:
            self.ctx_used = used
        self._paint_header()
        self._paint_frame()

    def note_reply(self, text: str, stats: StreamStats, *, e2e_ms: float, trigger: str) -> None:
        if text:
            self.brain = text
        self.trigger = trigger
        self.update_gen(stats, e2e_ms=e2e_ms)
        self._commit_open_you()
        self._commit_open_brain(text or self.brain)
        self.you = ""
        self.live = ""
        self._paint_frame(force=True)

    def note_sound(self, label: str, score: float = 0.0) -> None:
        event = (label or "").strip().strip("[]")
        if not event:
            return
        self.event_label = event
        if not self.live:
            self.live = f"[{event}]"
        if not self._you_open and not self._brain_open:
            self._you_open = True
            if not self._you_stamp:
                self._you_stamp = _now()
        self._paint_frame(force=True)
        self._paint_header(force=True)

    def note_cut(self, reason: str = "barge-in") -> None:
        self.metrics.barge_ins += 1
        self._commit_open_brain(self.brain)
        self.brain = ""
        if not self._you_open:
            self.you = ""
        self.hint = reason
        self._paint_header(force=True)
        self._paint_frame(force=True)

    def note_error(self, msg: str) -> None:
        self.metrics.errors += 1
        self._brain_streaming = False
        self._commit_open_you()
        self._commit_open_brain(self.brain)
        body = (msg or "").strip()
        if body:
            self._turns.append(("ERROR", _now(), body))
            if len(self._turns) > MAX_TURNS:
                self._turns = self._turns[-MAX_TURNS:]
        self._paint_frame(force=True)
        self._paint_header(force=True)

    def note_stuck(self, msg: str) -> None:
        self.note_error("STUCK " + (msg or ""))
        self.status = "LISTENING"
        self.hint = "watchdog reset"
        self._paint_header(force=True)

    def heartbeat(self) -> None:
        if not self._started:
            return
        now = time.perf_counter()
        if now - self._last_beat < 0.45:
            return
        self._last_beat = now
        self._paint_header()
        self._paint_frame()

    def note_bus(self, msg: str) -> None:
        waiting = "waiting" in (msg or "").lower()
        if waiting:
            self.metrics.bus_reconnects += 1
            if self.status in {"WAITING", ""}:
                self.status = "WAITING"
                self.hint = msg
        elif self.status == "WAITING":
            self.status = "LISTENING"
            self.hint = "ears on"
        self._paint_header(force=True)

    def close(self, final: str = "") -> None:
        if self._started:
            sys.stdout.write(f"{CSI}r")
            sys.stdout.write(SHOW + ALT_OFF + RESET)
            self._started = False
            sys.stdout.flush()
        text = final or self.brain
        if text or self.you:
            print()
            print(f"YOU    {self.you}")
            print(f"BRAIN  {text}")
            print(f"LAT    {self._lat_body()}")
            print()
