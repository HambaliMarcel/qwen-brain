"""Fan-out TCP JSONL hub and subscriber. Stdlib only, TCP_NODELAY."""

from __future__ import annotations

import socket
import threading
import time
from typing import Callable, Iterator, Optional

from .events import SttEvent, dump_event, parse_event


class JsonlHub:
    """Producer-side bus. ASR integrator publishes; brain(s) subscribe."""

    def __init__(self, host: str = "127.0.0.1", port: int = 18765):
        self.host = host
        self.port = port
        self._sock: Optional[socket.socket] = None
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._alive = False

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(16)
        sock.settimeout(0.5)
        self._sock = sock
        self._alive = True
        threading.Thread(target=self._accept, name="stt-bus-accept", daemon=True).start()

    def close(self) -> None:
        self._alive = False
        with self._lock:
            clients = list(self._clients)
            self._clients.clear()
        for c in clients:
            try:
                c.close()
            except OSError:
                pass
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def publish(self, payload: dict) -> None:
        data = (dump_event(payload) + "\n").encode("utf-8")
        with self._lock:
            clients = list(self._clients)
        dead: list[socket.socket] = []
        for c in clients:
            try:
                c.sendall(data)
            except OSError:
                dead.append(c)
        if dead:
            with self._lock:
                self._clients = [c for c in self._clients if c not in dead]
            for c in dead:
                try:
                    c.close()
                except OSError:
                    pass

    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def _accept(self) -> None:
        assert self._sock is not None
        while self._alive:
            try:
                conn, _addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
            with self._lock:
                self._clients.append(conn)


class SttBusClient:
    """Consumer-side bus. Reconnects until cancelled."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 18765,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self.host = host
        self.port = port
        self.on_status = on_status
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def iter_events(self) -> Iterator[SttEvent]:
        backoff = 0.2
        while not self._stop.is_set():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(3.0)
                sock.connect((self.host, self.port))
                sock.settimeout(0.4)
                if self.on_status:
                    self.on_status(f"stt bus {self.host}:{self.port}")
                backoff = 0.2
                yield from self._read_lines(sock)
            except OSError as e:
                if self.on_status:
                    self.on_status(f"waiting for STT bus {self.host}:{self.port} ({e})")
                time.sleep(backoff)
                backoff = min(2.0, backoff * 1.6)
            finally:
                try:
                    sock.close()
                except OSError:
                    pass

    def _read_lines(self, sock: socket.socket) -> Iterator[SttEvent]:
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    data = parse_event(line.decode("utf-8", errors="replace"))
                except (ValueError, UnicodeError):
                    continue
                if not data:
                    continue
                yield SttEvent.from_dict(data)
