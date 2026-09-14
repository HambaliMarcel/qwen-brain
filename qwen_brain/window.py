"""Cindy-style left-third console: light paper, dark blue, Consolas."""

from __future__ import annotations

import ctypes
import sys
import threading
import time

_WRAP_WIDTH = 56


class _COORD(ctypes.Structure):
    _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]


class _SMALL_RECT(ctypes.Structure):
    _fields_ = [
        ("Left", ctypes.c_short),
        ("Top", ctypes.c_short),
        ("Right", ctypes.c_short),
        ("Bottom", ctypes.c_short),
    ]


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class _CONSOLE_FONT_INFOEX(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_ulong),
        ("nFont", ctypes.c_ulong),
        ("dwFontSize", _COORD),
        ("FontFamily", ctypes.c_uint),
        ("FontWeight", ctypes.c_uint),
        ("FaceName", ctypes.c_wchar * 32),
    ]


def console_width() -> int:
    return _WRAP_WIDTH


def size_live_window(*, fraction: float = 1.0 / 3.0) -> int:
    """Snap the console to the left `fraction` of the monitor work area."""
    global _WRAP_WIDTH, _BUFFER_SIZED
    cols = 56
    rows = 42
    if sys.platform == "win32":
        try:
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            font = _CONSOLE_FONT_INFOEX()
            font.cbSize = ctypes.sizeof(_CONSOLE_FONT_INFOEX)
            font.dwFontSize = _COORD(0, 16)
            font.FontFamily = 54
            font.FontWeight = 400
            font.FaceName = "Consolas"
            kernel32.SetCurrentConsoleFontEx(handle, False, ctypes.byref(font))
            kernel32.GetCurrentConsoleFontEx(handle, False, ctypes.byref(font))
            cell_w = int(font.dwFontSize.X) or max(8, int(font.dwFontSize.Y) // 2 or 8)
            cell_h = int(font.dwFontSize.Y) or 16
            work = _RECT()
            if not user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(work), 0):
                work.left = 0
                work.top = 0
                work.right = int(user32.GetSystemMetrics(0) or 1920)
                work.bottom = int(user32.GetSystemMetrics(1) or 1080)
            work_w = max(360, int(work.right - work.left))
            work_h = max(420, int(work.bottom - work.top))
            target_w = max(420, int(work_w * fraction))
            target_h = work_h
            cols = max(48, min(88, target_w // cell_w))
            rows = max(32, min(56, target_h // cell_h))
            hwnd = kernel32.GetConsoleWindow()
            if hwnd:
                current = _RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(current))
                already = (
                    abs(current.left - work.left) <= 8
                    and abs(current.top - work.top) <= 8
                    and abs((current.right - current.left) - target_w) <= 16
                    and abs((current.bottom - current.top) - target_h) <= 16
                )
                if not already:
                    user32.MoveWindow(hwnd, work.left, work.top, target_w, target_h, True)
            global _BUFFER_SIZED
            if not _BUFFER_SIZED:
                kernel32.SetConsoleScreenBufferSize(handle, _COORD(max(cols, 80), max(rows, 400)))
                rect = _SMALL_RECT(0, 0, cols - 1, rows - 1)
                kernel32.SetConsoleWindowInfo(handle, True, ctypes.byref(rect))
                kernel32.SetConsoleScreenBufferSize(handle, _COORD(cols, max(rows, 400)))
                kernel32.SetConsoleWindowInfo(handle, True, ctypes.byref(rect))
                _BUFFER_SIZED = True
        except Exception:
            cols = 56
    _WRAP_WIDTH = cols
    return cols


_SNAP_STARTED = False
_BUFFER_SIZED = False


def keep_snapped_left(*, fraction: float = 1.0 / 3.0) -> None:
    """Reposition only if the window drifted. Never resize the buffer again."""
    global _SNAP_STARTED
    if _SNAP_STARTED or sys.platform != "win32":
        return
    _SNAP_STARTED = True

    def _loop() -> None:
        while True:
            time.sleep(3.0)
            try:
                _reposition_only(fraction=fraction)
            except Exception:
                pass

    threading.Thread(target=_loop, name="qwen-snap", daemon=True).start()


def _reposition_only(*, fraction: float = 1.0 / 3.0) -> None:
    if sys.platform != "win32":
        return
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    hwnd = kernel32.GetConsoleWindow()
    if not hwnd:
        return
    work = _RECT()
    if not user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(work), 0):
        return
    work_w = max(360, int(work.right - work.left))
    work_h = max(420, int(work.bottom - work.top))
    target_w = max(420, int(work_w * fraction))
    target_h = work_h
    current = _RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(current))
    already = (
        abs(current.left - work.left) <= 12
        and abs(current.top - work.top) <= 12
        and abs((current.right - current.left) - target_w) <= 24
        and abs((current.bottom - current.top) - target_h) <= 24
    )
    if not already:
        user32.MoveWindow(hwnd, work.left, work.top, target_w, target_h, True)


def apply_light_terminal(title: str = "brain") -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if sys.platform == "win32":
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
            handle = kernel32.GetStdHandle(-11)
            kernel32.SetConsoleTextAttribute(handle, 0xF1)
            kernel32.SetConsoleTitleW(title)
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass
    cols = size_live_window()
    keep_snapped_left()
    try:
        sys.stdout.write("\x1b[107;34m\x1b[2J\x1b[H")
        sys.stdout.flush()
    except Exception:
        pass
    return cols
