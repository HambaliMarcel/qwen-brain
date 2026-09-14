"""JSONL STT bus protocol (one JSON object per line, TCP localhost)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


PROTOCOL_VERSION = 1


def dump_event(payload: dict[str, Any]) -> str:
    import json

    body = dict(payload)
    body.setdefault("v", PROTOCOL_VERSION)
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"))


def parse_event(line: str) -> dict[str, Any]:
    import json

    raw = (line or "").strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("event must be a JSON object")
    return data


@dataclass
class SttEvent:
    type: str
    text: str = ""
    language: str = ""
    speaking: bool = False
    decoding: bool = False
    utterance_id: int = 0
    gap_sec: float = 0.0
    silence_sec: float = 0.0
    ts: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SttEvent":
        return cls(
            type=str(data.get("type") or ""),
            text=str(data.get("text") or ""),
            language=str(data.get("language") or ""),
            speaking=bool(data.get("speaking")),
            decoding=bool(data.get("decoding")),
            utterance_id=int(data.get("utterance_id") or 0),
            gap_sec=float(data.get("gap_sec") or 0.0),
            silence_sec=float(data.get("silence_sec") or 0.0),
            ts=float(data.get("ts") or 0.0),
            raw=data,
        )


def is_command_text(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if t.startswith("[") and t.endswith("]") and " " not in t.strip("[]"):
        return False
    lowered = t.lower().strip(" .!?，。！？")
    if lowered in {"uh", "um", "erm", "hmm", "ah", "eh", "oh", "ha", "haha"}:
        return False
    return any(ch.isalnum() for ch in t)


def same_turn(prev: Optional[str], new: str) -> bool:
    a = (prev or "").strip().lower()
    b = (new or "").strip().lower()
    if not a or not b:
        return False
    return a == b or b.startswith(a) or a.startswith(b)
