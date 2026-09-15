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


_FILLER_WORDS = {
    "uh",
    "um",
    "erm",
    "hmm",
    "hm",
    "ah",
    "aah",
    "ahh",
    "eh",
    "oh",
    "ooh",
    "oooh",
    "ha",
    "haha",
    "mm",
    "mmm",
}


def _content_words(text: str) -> list[str]:
    t = strip_language_leak(text or "")
    t = _TAG_RE.sub(" ", t)
    t = _SPACE_RE.sub(" ", t).strip()
    return [w for w in t.split() if any(ch.isalnum() for ch in w)]


def is_command_text(text: str) -> bool:
    t = strip_language_leak(text or "")
    if not t or is_sound_tag(t):
        return False
    lowered = t.lower().strip(" .!?，。！？")
    if lowered in _FILLER_WORDS:
        return False
    if "suara non-bicara" in lowered:
        rest = _TAG_RE.sub(" ", lowered)
        rest = _SPACE_RE.sub(" ", rest).strip()
        if not rest or rest in {"suara non-bicara", "suara non-bicara?"}:
            return False
    return any(ch.isalnum() for ch in t)


def looks_complete(text: str) -> bool:
    """True when a live draft already looks like a finished utterance.

    One or two words plus a period ("What.", "You.") are ASR punctuation,
    not a finished line. Those must wait for more audio or a real hold.
    """
    t = strip_language_leak(text or "")
    t = _TAG_RE.sub(" ", t)
    t = _SPACE_RE.sub(" ", t).strip()
    if not t:
        return False
    words = _content_words(t)
    cjk = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", t))
    if t[-1] in "?？！" and (len(words) >= 3 or cjk >= 4):
        return True
    if t[-1] in _COMPLETE_END and len(words) >= 5:
        return True
    if cjk >= 4 and not re.search(r"[A-Za-zÀ-ÿ]{3,}", t):
        return True
    return len(words) >= 5


def is_short_fragment(text: str) -> bool:
    """True for breath-sized LAST pieces that should be stitched, not answered."""
    t = strip_language_leak(text or "")
    t = _TAG_RE.sub(" ", t)
    t = _SPACE_RE.sub(" ", t).strip()
    if not t:
        return True
    words = _content_words(t)
    cjk = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", t))
    if cjk >= 8:
        return False
    if len(words) >= 5:
        return False
    if t[-1] in "?？！" and len(words) >= 3:
        return False
    return len(words) <= 3


def join_fragments(prev: str, new: str) -> str:
    a = strip_language_leak(prev or "").strip()
    b = strip_language_leak(new or "").strip()
    if not a:
        return b
    if not b:
        return a
    if same_turn(a, b):
        return b if len(b) >= len(a) else a
    # ASR LAST re-sends the whole paragraph; if the held piece is already
    # inside it, the paragraph is the turn, not "piece + paragraph".
    if _find_span(_norm_words(b), _norm_words(a)) >= 0:
        return b
    return f"{a} {b}".strip()


_LOOP_MAX_NGRAM = 8
_LOOP_MIN_REPEATS = 4
# A single word sung 4-5 times ("na na na na na") is normal; only a longer
# run of one word is a decoder spiral.
_LOOP_MIN_REPEATS_1GRAM = 6
_WORD_NORM_RE = re.compile(r"[^\w]+", re.UNICODE)


def _norm_words(text: str) -> list[str]:
    return [_WORD_NORM_RE.sub("", w.lower()) for w in (text or "").split()]


def _find_span(hay: list[str], needle: list[str]) -> int:
    n = len(needle)
    if not n or n > len(hay):
        return -1
    for i in range(len(hay) - n + 1):
        if hay[i : i + n] == needle:
            return i
    return -1


def collapse_loops(text: str, keep: int = 2, min_repeats: int = _LOOP_MIN_REPEATS) -> tuple[str, bool]:
    """Collapse an ASR decoder spiral ("black on black on black on …").

    Hooks sung 2–3 times stay. Only an n-gram repeated `min_repeats`+ times
    back-to-back is cut to `keep` copies. Returns (text, looped).
    """
    words = (text or "").split()
    n = len(words)
    if n < min_repeats:
        return text or "", False
    norm = _norm_words(text)
    out: list[str] = []
    looped = False
    i = 0
    while i < n:
        best_len = 0
        best_reps = 0
        for length in range(1, min(_LOOP_MAX_NGRAM, (n - i) // min_repeats) + 1):
            unit = norm[i : i + length]
            if not any(unit):
                continue
            reps = 1
            j = i + length
            while j + length <= n and norm[j : j + length] == unit:
                reps += 1
                j += length
            need = max(min_repeats, _LOOP_MIN_REPEATS_1GRAM) if length == 1 else min_repeats
            if reps >= need and reps * length > best_reps * best_len:
                best_len, best_reps = length, reps
        if best_len:
            out.extend(words[i : i + best_len * keep])
            i += best_len * best_reps
            looped = True
        else:
            out.append(words[i])
            i += 1
    return " ".join(out), looped


def is_degenerate(text: str) -> bool:
    """True when most of the line is a decoder loop — not something to answer."""
    words = (text or "").split()
    if len(words) < 8:
        return False
    collapsed, looped = collapse_loops(text)
    return looped and len(collapsed.split()) <= int(0.6 * len(words))


def strip_already_sent(prev: Optional[str], new: str) -> str:
    """Drop the part of `new` that was already answered as `prev`.

    ASR LAST grows as a paragraph and re-sends it whole. Only the words the
    brain has not seen yet should become the next turn — a shorter prompt is
    faster to prefill and does not answer the same lyric twice.
    """
    a = strip_language_leak(prev or "").strip()
    b = strip_language_leak(new or "").strip()
    if not a or not b:
        return b
    pw = _norm_words(a)
    if len(pw) < 4:
        return b
    raw = b.split()
    at = _find_span(_norm_words(b), pw)
    if at < 0:
        return b
    rest = raw[:at] + raw[at + len(pw) :]
    return " ".join(rest).strip()


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
