"""Fixed-panel dashboard: LIVE STT, committed YOU, streaming BRAIN."""

from __future__ import annotations

import ctypes
import re
import shutil
import sys
import time

CSI = "\x1b["
RESET = f"{CSI}0m"
BOLD = f"{CSI}1m"
DIM = f"{CSI}2m"
WHITE = f"{CSI}97m"
GRAY = f"{CSI}90m"
CYAN = f"{CSI}36m"
CYAN_B = f"{CSI}96m"
GREEN = f"{CSI}32m"
GREEN_B = f"{CSI}92m"
YELLOW = f"{CSI}33m"
YELLOW_B = f"{CSI}93m"
MAGENTA = f"{CSI}35m"

ALT_ON = f"{CSI}?1049h"
ALT_OFF = f"{CSI}?1049l"
HIDE = f"{CSI}?25l"
SHOW = f"{CSI}?25h"
HOME = f"{CSI}H"
CLEAR = f"{CSI}2J"


def enable_windows_vt() -> None:
    if sys.platform != "win32":
        return
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleOutputCP(65001)
    kernel32.SetConsoleCP(65001)
    handle = kernel32.GetStdHandle(-11)
    mode = ctypes.c_uint32()
    if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _width() -> int:
    return max(72, min(110, shutil.get_terminal_size((100, 28)).columns - 1))


def _visible_len(s: str) -> int:
    return len(re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s))


def _pad_row(inner: str, width: int) -> str:
    vis = _visible_len(inner)
    if vis < width:
        inner = inner + (" " * (width - vis))
    elif vis > width:
        inner = inner[: width - 1] + "…"
        inner += " " * max(0, width - _visible_len(inner))
    return inner


def _wrap(text: str, width: int, limit: int) -> list[str]:
    if not text:
        return [""] * max(1, limit)
    lines: list[str] = []
    buf = ""
    for ch in text:
        if ch == "\n":
            lines.append(buf)
            buf = ""
            continue
        buf += ch
        if len(buf) >= width:
            lines.append(buf)
            buf = ""
    if buf:
        lines.append(buf)
    lines = lines[-limit:]
    while len(lines) < limit:
        lines.append("")
    return lines


class BrainUI:
    def __init__(self) -> None:
        enable_windows_vt()
        self._started = False
        self.title = "Qwen brain  ·  live STT → 4B"
        self.detail = ""
        self.status = "WAITING"
        self.hint = "waiting for STT bus"
        self.live = ""
        self.you = ""
        self.brain = ""
        self.ttft_ms = 0.0
        self.total_ms = 0.0
        self.backend = "llm"
        self._last_draw = 0.0
        self._force = True

    def banner(self, title: str, detail: str) -> None:
        self.title = title
        self.detail = detail
        if not self._started:
            sys.stdout.write(ALT_ON + HIDE + CLEAR + HOME)
            self._started = True
        self._force = True
        self.draw()

    def set_status(self, status: str, hint: str = "") -> None:
        self.status = status
        if hint:
            self.hint = hint
        self.draw()

    def set_live(self, text: str) -> None:
        self.live = text
        self.draw()

    def set_you(self, text: str) -> None:
        self.you = text
        self.draw(force=True)

    def set_brain(self, text: str) -> None:
        self.brain = text
        self.draw()

    def append_brain(self, delta: str) -> None:
        self.brain += delta
        self.draw()

    def draw(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and not self._force and (now - self._last_draw) < 0.04:
            return
        self._last_draw = now
        self._force = False
        if not self._started:
            return
        w = _width()
        inner = w - 2

        def box_top(title: str) -> str:
            label = f" {title} "
            fill = max(0, inner - len(label) - 1)
            return f"{CYAN}┌{label}{'─' * fill}┐{RESET}"

        def box_bot() -> str:
            return f"{CYAN}└{'─' * inner}┘{RESET}"

        def row(content: str) -> str:
            return f"{CYAN}│{RESET}{_pad_row(content, inner)}{CYAN}│{RESET}"

        st = self.status.upper()
        if st in {"ANSWERING", "THINKING"}:
            badge = f"{YELLOW_B}{BOLD} {st} {RESET}"
        elif st in {"LISTENING", "SPEAKING"}:
            badge = f"{GREEN_B}{BOLD} {st} {RESET}"
        elif st in {"WAITING"}:
            badge = f"{GRAY}{BOLD} {st} {RESET}"
        else:
            badge = f"{CYAN_B}{BOLD} {st} {RESET}"
        ttft = f"{self.ttft_ms:.0f}ms" if self.ttft_ms else "—"
        tot = f"{self.total_ms:.0f}ms" if self.total_ms else "—"
        top = (
            f" {badge}  {GRAY}backend{RESET} {WHITE}{self.backend}{RESET}  "
            f"{GRAY}ttft{RESET} {WHITE}{ttft}{RESET}  "
            f"{GRAY}reply{RESET} {WHITE}{tot}{RESET}  "
            f"{GRAY}{self.hint}{RESET}"
        )
        try:
            rows = shutil.get_terminal_size((100, 28)).lines
        except Exception:
            rows = 28
        body = max(12, rows - 10)
        live_n = max(3, min(6, body // 4))
        you_n = max(3, min(5, body // 4))
        brain_n = max(5, body - live_n - you_n)

        live_lines = _wrap(self.live or "listening…", inner - 2, live_n)
        you_lines = _wrap(self.you or "", inner - 2, you_n)
        brain_lines = _wrap(self.brain or "", inner - 2, brain_n)
        if not self.you:
            you_lines = _wrap_ansi(
                f"{DIM}committed command appears here{RESET}", inner - 2, you_n
            )
        if not self.brain:
            brain_lines = _wrap_ansi(
                f"{DIM}streamed 4B reply appears here{RESET}", inner - 2, brain_n
            )

        out = [
            f"{BOLD}{CYAN_B}  {self.title}{RESET}",
            f"  {GRAY}{self.detail}{RESET}" if self.detail else "",
            box_top("SESSION"),
            row(top),
            box_bot(),
            box_top("LIVE  ·  STT draft"),
            *[row(" " + (f"{YELLOW}{line}{RESET}" if self.live else line)) for line in live_lines],
            box_bot(),
            box_top("YOU  ·  command"),
            *[row(" " + (f"{GREEN}{line}{RESET}" if self.you else line)) for line in you_lines],
            box_bot(),
            box_top("BRAIN  ·  Qwen3.5-4B"),
            *[row(" " + (f"{WHITE}{line}{RESET}" if self.brain else line)) for line in brain_lines],
            box_bot(),
            f" {GRAY}Ctrl+C stop{RESET}   {GRAY}ears = qwen3-asr-stream integrator · core = local 4B{RESET}",
        ]
        out = [ln for ln in out if ln is not None]
        sys.stdout.write(HOME + CLEAR)
        sys.stdout.write("\n".join(out))
        sys.stdout.write("\n")
        sys.stdout.flush()

    def close(self, final: str = "") -> None:
        if self._started:
            sys.stdout.write(SHOW + ALT_OFF)
            self._started = False
            sys.stdout.flush()
        text = final or self.brain
        if text:
            print()
            print(f"{BOLD}{GREEN}you{RESET}    {self.you}")
            print(f"{BOLD}{CYAN}brain{RESET}  {text}")
            print()


def _wrap_ansi(text: str, width: int, limit: int) -> list[str]:
    if not text:
        return [""] * limit
    lines: list[str] = []
    cur = ""
    vis = 0
    i = 0
    while i < len(text):
        if text[i] == "\x1b":
            m = re.match(r"\x1b\[[0-9;]*[A-Za-z]", text[i:])
            if m:
                cur += m.group(0)
                i += len(m.group(0))
                continue
        cur += text[i]
        vis += 1
        i += 1
        if vis >= width:
            lines.append(cur + RESET)
            cur = ""
            vis = 0
    if vis > 0 or not lines:
        lines.append(cur)
    lines = lines[-limit:]
    while len(lines) < limit:
        lines.append("")
    return lines
