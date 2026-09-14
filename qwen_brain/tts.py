"""TTS hook. MVP is text-only; a real engine will consume token deltas later."""

from __future__ import annotations

from typing import Protocol


class TtsSink(Protocol):
    def on_token(self, token: str) -> None: ...

    def flush(self) -> None: ...

    def cancel(self) -> None: ...


class NullTts:
    """Placeholder until a local / ElevenLabs TTS lane is wired."""

    def on_token(self, token: str) -> None:
        return None

    def flush(self) -> None:
        return None

    def cancel(self) -> None:
        return None
