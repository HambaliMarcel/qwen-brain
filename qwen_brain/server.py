"""Launch llama-server for the Qwen3.8-27B chat brain (not the ASR GGUF)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from .config import BrainConfig


class LlamaServerError(RuntimeError):
    pass


def resolve_paths(cfg: BrainConfig) -> tuple[Path, Path]:
    exe = cfg.llama_server
    if exe.is_dir():
        exe = exe / "llama-server.exe"
    model = cfg.model
    if not model.is_file():
        alt = Path(r"C:\AI\models") / model.name
        if alt.is_file():
            model = alt
    return exe, model


def build_server_cmd(cfg: BrainConfig, extra: Optional[list[str]] = None) -> list[str]:
    exe, model = resolve_paths(cfg)
    if not exe.is_file():
        raise FileNotFoundError(f"llama-server not found: {exe}")
    if not model.is_file():
        raise FileNotFoundError(f"brain GGUF not found: {model}")
    cmd = [
        str(exe),
        "-m",
        str(model),
        "-ngl",
        str(cfg.ngl),
        "-c",
        str(cfg.ctx),
        "-np",
        "1",
        "--port",
        str(cfg.port),
        "--host",
        cfg.host if not cfg.host.startswith("http") else "127.0.0.1",
        "-fa",
        "on",
        "--jinja",
        "--cache-prompt",
        "--no-webui",
        "--reasoning-budget",
        "0",
        "--chat-template-kwargs",
        '{"enable_thinking":false}',
        "-ctk",
        "q8_0",
        "-ctv",
        "q8_0",
    ]
    spec = (cfg.spec_type or "").strip().lower()
    if spec and spec not in {"none", "off", "0"}:
        cmd.extend(
            [
                "--spec-type",
                cfg.spec_type,
                "--spec-draft-n-max",
                str(cfg.spec_draft_n_max),
                "--spec-draft-ngl",
                str(cfg.ngl),
            ]
        )
    if extra:
        cmd.extend(extra)
    return cmd


def health(url: str, timeout: float = 2.0) -> bool:
    try:
        req = Request(url.rstrip("/") + "/health", method="GET")
        with urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (URLError, OSError, TimeoutError):
        return False


def wait_until_ready(url: str, timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    last = "not ready"
    while time.time() < deadline:
        if health(url):
            return
        time.sleep(0.4)
        last = "health not ready"
    raise LlamaServerError(f"brain llama-server did not become ready at {url}: {last}")


def start_server(cfg: BrainConfig, new_console: bool = True) -> subprocess.Popen:
    cmd = build_server_cmd(cfg)
    kwargs: dict = {}
    if sys.platform == "win32" and new_console:
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
    cwd = str(Path(cmd[0]).parent)
    env = os.environ.copy()
    return subprocess.Popen(cmd, cwd=cwd, env=env, **kwargs)


def ensure_server(cfg: BrainConfig, start: bool) -> Optional[subprocess.Popen]:
    if health(cfg.url):
        return None
    if not start:
        raise LlamaServerError(
            f"No brain llama-server at {cfg.url}. Run `python -m qwen_brain serve` "
            "in another terminal, or add --start-server."
        )
    proc = start_server(cfg)
    wait_until_ready(cfg.url, timeout=240.0)
    return proc


def fetch_context(url: str, fallback_ctx: int = 8192) -> tuple[int, int]:
    """Return (tokens_used, n_ctx) from llama-server slots/props."""
    base = url.rstrip("/")
    try:
        req = Request(base + "/slots", method="GET")
        with urlopen(req, timeout=0.35) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if isinstance(data, list) and data:
            slot = data[0] if isinstance(data[0], dict) else {}
        elif isinstance(data, dict):
            slots = data.get("slots") or data.get("data") or []
            slot = slots[0] if slots else data
        else:
            slot = {}
        n_ctx = int(slot.get("n_ctx") or fallback_ctx)
        used = slot.get("n_past")
        if used is None:
            used = int(slot.get("n_prompt_tokens") or 0) + int(slot.get("n_decoded") or 0)
        return max(0, int(used)), max(1, n_ctx)
    except Exception:
        pass
    try:
        req = Request(base + "/props", method="GET")
        with urlopen(req, timeout=0.35) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        n_ctx = int(
            (data.get("default_generation_settings") or {}).get("n_ctx")
            or data.get("n_ctx")
            or fallback_ctx
        )
        return 0, max(1, n_ctx)
    except Exception:
        return 0, max(1, fallback_ctx)
