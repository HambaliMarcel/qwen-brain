"""Live session counters for the terminal dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic


def tok_per_sec(tokens: int, duration_ms: float) -> float:
    if tokens <= 0 or duration_ms <= 0:
        return 0.0
    return tokens / (duration_ms / 1000.0)


def avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


@dataclass
class SessionMetrics:
    started_at: float = field(default_factory=monotonic)
    turns: int = 0
    eager_turns: int = 0
    commit_turns: int = 0
    barge_ins: int = 0
    errors: int = 0
    bus_reconnects: int = 0
    stt_live_updates: int = 0
    cancelled: int = 0
    ttft_ms: list[float] = field(default_factory=list)
    gen_ms: list[float] = field(default_factory=list)
    e2e_ms: list[float] = field(default_factory=list)
    tok_s: list[float] = field(default_factory=list)
    last_trigger: str = ""
    last_language: str = ""
    last_event: str = ""
    utterance_id: int = 0
    silence_sec: float = 0.0
    speaking: bool = False
    decoding: bool = False

    def uptime_s(self) -> float:
        return monotonic() - self.started_at

    def record_turn(
        self,
        *,
        trigger: str,
        ttft_ms: float,
        gen_ms: float,
        e2e_ms: float,
        tok_s: float,
        cancelled: bool = False,
    ) -> None:
        self.turns += 1
        self.last_trigger = trigger
        if trigger == "eager" or trigger == "live":
            self.eager_turns += 1
        elif trigger == "commit":
            self.commit_turns += 1
        if cancelled:
            self.cancelled += 1
            return
        if ttft_ms > 0:
            self.ttft_ms.append(ttft_ms)
            self.ttft_ms = self.ttft_ms[-20:]
        if gen_ms > 0:
            self.gen_ms.append(gen_ms)
            self.gen_ms = self.gen_ms[-20:]
        if e2e_ms > 0:
            self.e2e_ms.append(e2e_ms)
            self.e2e_ms = self.e2e_ms[-20:]
        if tok_s > 0:
            self.tok_s.append(tok_s)
            self.tok_s = self.tok_s[-20:]

    def avg_ttft_ms(self) -> float:
        return avg(self.ttft_ms)

    def avg_gen_ms(self) -> float:
        return avg(self.gen_ms)

    def avg_e2e_ms(self) -> float:
        return avg(self.e2e_ms)

    def avg_tok_s(self) -> float:
        return avg(self.tok_s)
