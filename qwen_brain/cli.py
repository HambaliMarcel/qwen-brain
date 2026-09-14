"""CLI: python -m qwen_brain <serve|listen|chat>"""

from __future__ import annotations

import argparse
import os
import sys
import time

from .config import BrainConfig
from .hermes import make_brain
from .server import LlamaServerError, build_server_cmd, ensure_server, start_server
from .session import AssistantSession
from .ui import enable_windows_vt


def _cfg_from_args(args: argparse.Namespace) -> BrainConfig:
    cfg = BrainConfig.from_env()
    if getattr(args, "host", None):
        cfg.host = args.host
    if getattr(args, "port", None) is not None:
        cfg.port = int(args.port)
    if getattr(args, "model", None):
        from pathlib import Path

        cfg.model = Path(args.model)
    if getattr(args, "llama_server", None):
        from pathlib import Path

        cfg.llama_server = Path(args.llama_server)
    if getattr(args, "stt_host", None):
        cfg.stt_host = args.stt_host
    if getattr(args, "stt_port", None) is not None:
        cfg.stt_port = int(args.stt_port)
    if getattr(args, "backend", None):
        cfg.backend = str(args.backend).strip().lower()
    if getattr(args, "no_eager", False):
        cfg.eager = False
    if getattr(args, "eager_silence", None) is not None:
        cfg.eager_silence_sec = float(args.eager_silence)
    if getattr(args, "max_tokens", None) is not None:
        cfg.max_tokens = int(args.max_tokens)
    return cfg


def cmd_serve(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    cmd = build_server_cmd(cfg)
    print("Starting brain llama-server:")
    print(" ", " ".join(cmd))
    proc = start_server(cfg, new_console=not args.same_console)
    if args.same_console:
        return int(proc.wait() or 0)
    from .server import wait_until_ready

    wait_until_ready(cfg.url, timeout=240.0)
    print(f"Ready at {cfg.url}  (PID {proc.pid})")
    print("Leave this process running. ASR stays on port 9999.")
    try:
        while proc.poll() is None:
            time.sleep(0.5)
    except KeyboardInterrupt:
        proc.terminate()
    return 0


def cmd_listen(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    if args.start_server:
        ensure_server(cfg, start=True)
    else:
        from .server import health

        if cfg.backend == "llm" and not health(cfg.url):
            raise LlamaServerError(
                f"No brain llama-server at {cfg.url}. Run `python -m qwen_brain serve` "
                "or add --start-server."
            )
    session = AssistantSession(cfg)
    return session.run()


def cmd_chat(args: argparse.Namespace) -> int:
    enable_windows_vt()
    cfg = _cfg_from_args(args)
    if args.start_server:
        ensure_server(cfg, start=True)
    brain = make_brain(cfg)
    print(f"chat  {cfg.url if cfg.backend == 'llm' else cfg.hermes_api}  ({cfg.backend})")
    print("Type a line, empty line to quit.")
    while True:
        try:
            line = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return 0
        if not line:
            return 0
        acc: list[str] = []
        print("brain> ", end="", flush=True)
        def on_token(delta: str, stats=None) -> None:
            acc.append(delta)
            sys.stdout.write(delta)
            sys.stdout.flush()

        text, stats = brain.ask(line, on_token=on_token)
        if text and not acc:
            sys.stdout.write(text)
        print(
            f"\n  ttft {stats.first_token_ms:.0f}ms  gen {stats.total_ms:.0f}ms  "
            f"{stats.tok_s:.0f} tok/s  tok {stats.tokens}"
        )
    return 0


def _shared_flags() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--host", default=os.environ.get("QWEN_BRAIN_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("QWEN_BRAIN_PORT", 8080)))
    p.add_argument("--model", default=os.environ.get("QWEN_BRAIN_MODEL"))
    p.add_argument("--llama-server", default=os.environ.get("LLAMA_SERVER"))
    p.add_argument("--stt-host", default=os.environ.get("QWEN_BRAIN_STT_HOST", "127.0.0.1"))
    p.add_argument("--stt-port", type=int, default=int(os.environ.get("QWEN_BRAIN_STT_PORT", 18765)))
    p.add_argument("--start-server", action="store_true")
    p.add_argument("--backend", choices=("llm", "hermes"), default=os.environ.get("QWEN_BRAIN_BACKEND", "llm"))
    p.add_argument("--no-eager", action="store_true", help="Wait for official ASR LAST commit only")
    p.add_argument("--eager-silence", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    return p


def build_parser() -> argparse.ArgumentParser:
    shared = _shared_flags()
    p = argparse.ArgumentParser(
        prog="qwen_brain",
        description="Local Qwen3.8-27B brain driven by live Qwen3-ASR.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="Start llama-server with the 27B chat GGUF", parents=[shared])
    s.add_argument("--same-console", action="store_true")
    s.set_defaults(func=cmd_serve)

    m = sub.add_parser("listen", help="Consume live STT and stream 27B text replies", parents=[shared])
    m.set_defaults(func=cmd_listen)

    c = sub.add_parser("chat", help="Type to the 27B without the mic (debug)", parents=[shared])
    c.set_defaults(func=cmd_chat)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except LlamaServerError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
