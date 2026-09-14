"""JSONL STT bus protocol (one JSON object per line, TCP localhost)."""

from __future__ import annotations

import re
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


_COMPLETE_END = set("?!。.？！…")
_TAG_RE = re.compile(r"\[[^\]]*\]")
_SCENE_RE = re.compile(r"\[scene\]\s*", re.I)
_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_LANG_LEAK_RE = re.compile(r"(?i)\blanguage\s+[A-Za-z]+\b(?:\s*<asr_text>)?")


@dataclass
class SttEvent:
    type: str
    text: str = ""
    display: str = ""
    language: str = ""
    speaking: bool = False
    decoding: bool = False
    utterance_id: int = 0
    gap_sec: float = 0.0
    silence_sec: float = 0.0
    ts: float = 0.0
    event: str = ""
    event_score: float = 0.0
    companion: bool = False
    non_speech_only: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SttEvent":
        text = str(data.get("text") or "")
        return cls(
            type=str(data.get("type") or ""),
            text=text,
            display=str(data.get("display") or text),
            language=str(data.get("language") or ""),
            speaking=bool(data.get("speaking")),
            decoding=bool(data.get("decoding")),
            utterance_id=int(data.get("utterance_id") or 0),
            gap_sec=float(data.get("gap_sec") or 0.0),
            silence_sec=float(data.get("silence_sec") or 0.0),
            ts=float(data.get("ts") or 0.0),
            event=str(data.get("event") or ""),
            event_score=float(data.get("event_score") or 0.0),
            companion=bool(data.get("companion")),
            non_speech_only=bool(data.get("non_speech_only")),
            raw=data,
        )


def strip_language_leak(text: str) -> str:
    t = _LANG_LEAK_RE.sub(" ", text or "")
    t = _SCENE_RE.sub(" ", t)
    return _SPACE_RE.sub(" ", t).strip()


def is_sound_tag(text: str) -> bool:
    t = (text or "").strip()
    return bool(t) and t.startswith("[") and t.endswith("]") and "[" not in t[1:-1]


def is_command_text(text: str) -> bool:
    t = strip_language_leak(text or "")
    if not t or is_sound_tag(t):
        return False
    lowered = t.lower().strip(" .!?，。！？")
    if lowered in {"uh", "um", "erm", "hmm", "ah", "eh", "oh", "ha", "haha"}:
        return False
    if "suara non-bicara" in lowered:
        rest = _TAG_RE.sub(" ", lowered)
        rest = _SPACE_RE.sub(" ", rest).strip()
        if not rest or rest in {"suara non-bicara", "suara non-bicara?"}:
            return False
    return any(ch.isalnum() for ch in t)


def looks_complete(text: str) -> bool:
    """True when a live draft already looks like a finished utterance.

    Short fragments ("I'm not", "mas") must wait for a real pause or LAST.
    """
    t = strip_language_leak(text or "")
    t = _TAG_RE.sub(" ", t)
    t = _SPACE_RE.sub(" ", t).strip()
    if not t:
        return False
    if t[-1] in _COMPLETE_END and len(t) >= 2:
        return True
    cjk = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", t))
    if cjk >= 4 and not re.search(r"[A-Za-zÀ-ÿ]{3,}", t):
        return True
    return len(t.split()) >= 5


def scene_prompt(event: str) -> str:
    ev = (event or "").strip().strip("[]")
    if not ev:
        return ""
    return f"[scene] [{ev}]"


def is_scene_prompt(text: str) -> bool:
    t = (text or "").strip()
    if not t.lower().startswith("[scene]"):
        return False
    rest = _SCENE_RE.sub("", t).strip()
    return bool(rest) and is_sound_tag(rest)


def canonical_turn(text: str) -> str:
    t = _SCENE_RE.sub(" ", text or "")
    t = _TAG_RE.sub(" ", t)
    t = _PUNCT_RE.sub(" ", t.lower())
    return _SPACE_RE.sub(" ", t).strip()


def with_sound_context(text: str, event: str) -> str:
    body = (text or "").strip()
    ev = (event or "").strip().strip("[]")
    if not ev:
        return body
    tag = f"[{ev}]"
    if not body:
        return tag
    if body.lower().startswith("[scene]"):
        return body
    if body.lower().startswith(tag.lower()):
        return body
    return f"{tag} {body}"


def meaningfully_longer(prev: Optional[str], new: str, extra: int = 8) -> bool:
    a = (prev or "").strip()
    b = (new or "").strip()
    if not b:
        return False
    if not a:
        return True
    if not same_turn(a, b):
        return False
    return len(b) >= len(a) + extra


def same_turn(prev: Optional[str], new: str) -> bool:
    ca = canonical_turn(prev or "")
    cb = canonical_turn(new or "")
    a = ca or (prev or "").strip().lower()
    b = cb or (new or "").strip().lower()
    if not a or not b:
        return False
    return a == b or b.startswith(a) or a.startswith(b)
