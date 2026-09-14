from __future__ import annotations

import unittest

from qwen_brain.config import BrainConfig
from qwen_brain.events import (
    SttEvent,
    is_command_text,
    looks_complete,
    parse_event,
    same_turn,
    strip_language_leak,
    with_sound_context,
)
from qwen_brain.llm import LlamaBrain, split_think_stream, strip_think
from qwen_brain.session import TurnPolicy


def ev(**kwargs) -> SttEvent:
    data = {
        "type": "live",
        "text": "",
        "language": "Indonesian",
        "speaking": False,
        "decoding": False,
        "utterance_id": 1,
        "silence_sec": 0.0,
    }
    data.update(kwargs)
    return SttEvent.from_dict(data)


class EventTests(unittest.TestCase):
    def test_parse_commit(self):
        line = '{"v":1,"type":"commit","text":"halo","utterance_id":3}'
        data = parse_event(line)
        self.assertEqual(data["type"], "commit")
        self.assertEqual(SttEvent.from_dict(data).text, "halo")

    def test_command_filter(self):
        self.assertTrue(is_command_text("jam berapa sekarang"))
        self.assertFalse(is_command_text(""))
        self.assertFalse(is_command_text("[musik]"))
        self.assertFalse(is_command_text("[finger snapping]"))
        self.assertFalse(is_command_text("language Canton"))
        self.assertFalse(is_command_text("[suara non-bicara?]"))
        self.assertEqual(strip_language_leak("language Canton 唔該"), "唔該")
        self.assertEqual(
            with_sound_context("halo pak", "finger snapping"),
            "[finger snapping] halo pak",
        )
        self.assertEqual(with_sound_context("", "batuk?"), "[batuk?]")
        self.assertEqual(with_sound_context("[batuk?]", "batuk?"), "[batuk?]")

    def test_same_turn(self):
        self.assertTrue(same_turn("jam berapa", "jam berapa sekarang"))
        self.assertTrue(same_turn("[batuk?] apa kabar", "apa kabar"))
        self.assertFalse(same_turn("jam berapa", "buka notepad"))


class ThinkStripTests(unittest.TestCase):
    def test_strip_block(self):
        self.assertEqual(strip_think("<think>secret</think>Halo"), "Halo")

    def test_hold_open_think(self):
        hold, emit = split_think_stream("Hi <think>abc")
        self.assertEqual(emit, "Hi ")
        self.assertTrue(hold.startswith("<think>"))

    def test_closed_think(self):
        hold, emit = split_think_stream("A<think>x</think>B")
        self.assertEqual(hold, "")
        self.assertEqual(emit, "AB")


class SpeculativeHistoryTests(unittest.TestCase):
    def test_hidden_ask_does_not_pollute_history(self):
        class StubBrain(LlamaBrain):
            def _stream(self, messages, gen=0, stats=None):
                self.seen = messages
                yield "ready"

        brain = StubBrain(BrainConfig())
        text, _stats = brain.ask("draft words", remember=False)
        self.assertEqual(text, "ready")
        self.assertEqual(brain.history, [])
        self.assertEqual(brain.seen[-1]["role"], "user")
        self.assertTrue(brain.seen[-1]["content"].startswith("draft words"))
        self.assertIn("Jangan ganti topik", brain.seen[-1]["content"])

        brain.remember_turn("final words", text)
        self.assertEqual(
            [(turn.role, turn.content) for turn in brain.history],
            [("user", "final words"), ("assistant", "ready")],
        )


class PersonalityTests(unittest.TestCase):
    def test_prompt_stays_on_thread_and_does_not_quote(self):
        from qwen_brain.config import VOICE_SYSTEM_PROMPT

        text = VOICE_SYSTEM_PROMPT.lower()
        self.assertIn("continuation", text)
        self.assertIn("do not quote", text)
        self.assertIn("typo", text)


class TurnPolicyTests(unittest.TestCase):
    def test_commit_fires_once(self):
        p = TurnPolicy(eager=False)
        first = p.should_ask(ev(type="commit", text="halo dunia", utterance_id=2))
        again = p.should_ask(ev(type="commit", text="halo dunia", utterance_id=2))
        self.assertEqual(first, "halo dunia")
        self.assertIsNone(again)

    def test_live_does_not_fire_while_speaking(self):
        p = TurnPolicy(eager=True, eager_silence_sec=0.5)
        out = p.should_ask(
            ev(type="live", text="halo", speaking=True, silence_sec=2.0, utterance_id=1)
        )
        self.assertIsNone(out)

    def test_eager_fires_after_pause(self):
        p = TurnPolicy(eager=True, eager_silence_sec=0.5)
        out = p.should_ask(
            ev(
                type="live",
                text="jam berapa sekarang ya?",
                speaking=False,
                decoding=False,
                silence_sec=0.6,
                utterance_id=4,
            )
        )
        self.assertEqual(out, "jam berapa sekarang ya?")

    def test_incomplete_draft_does_not_fire_on_short_pause(self):
        p = TurnPolicy(eager=True, eager_silence_sec=0.45)
        out = p.should_ask(
            ev(
                type="live",
                text="I'm not",
                speaking=False,
                silence_sec=0.5,
                gap_sec=2.0,
                utterance_id=4,
            )
        )
        self.assertIsNone(out)

    def test_incomplete_draft_fires_after_hold(self):
        p = TurnPolicy(eager=True, eager_silence_sec=0.45, hold_silence_sec=0.9)
        out = p.should_ask(
            ev(
                type="live",
                text="I'm not",
                speaking=False,
                silence_sec=0.95,
                utterance_id=4,
            )
        )
        self.assertEqual(out, "I'm not")

    def test_commit_after_eager_same_text_skipped(self):
        p = TurnPolicy(eager=True, eager_silence_sec=0.5)
        p.should_ask(
            ev(
                type="live",
                text="jam berapa sekarang ya?",
                speaking=False,
                silence_sec=0.7,
                utterance_id=4,
            )
        )
        out = p.should_ask(ev(type="commit", text="jam berapa sekarang ya?", utterance_id=4))
        self.assertIsNone(out)

    def test_commit_longer_text_restarts(self):
        p = TurnPolicy(eager=True, eager_silence_sec=0.5)
        p.should_ask(
            ev(
                type="live",
                text="jam berapa sekarang ya?",
                speaking=False,
                silence_sec=0.7,
                utterance_id=4,
            )
        )
        out = p.should_ask(ev(type="commit", text="jam berapa sekarang ya ini", utterance_id=4))
        self.assertEqual(out, "jam berapa sekarang ya ini")

    def test_decoding_does_not_fire_a_complete_line(self):
        p = TurnPolicy(eager=True, eager_silence_sec=0.45)
        out = p.should_ask(
            ev(
                type="live",
                text="jam berapa sekarang ya?",
                speaking=False,
                decoding=True,
                silence_sec=0.6,
                utterance_id=4,
            )
        )
        self.assertIsNone(out)

    def test_speculative_live_does_not_fire_while_speaking(self):
        p = TurnPolicy(eager=True, eager_silence_sec=0.16)
        out = p.should_ask(
            ev(
                type="live",
                text="jam berapa sekarang ya?",
                speaking=True,
                silence_sec=0.0,
                utterance_id=5,
            )
        )
        self.assertIsNone(out)

    def test_live_revision_on_longer_draft(self):
        p = TurnPolicy(eager=True, eager_silence_sec=0.16, revise_extra_chars=8)
        first = p.should_ask(
            ev(
                type="live",
                text="oke jadi ini bagus sekali",
                speaking=False,
                silence_sec=0.5,
                utterance_id=6,
            )
        )
        again = p.should_ask(
            ev(
                type="live",
                text="oke jadi ini bagus atau enggak ya",
                speaking=False,
                silence_sec=0.5,
                utterance_id=6,
            )
        )
        self.assertEqual(first, "oke jadi ini bagus sekali")
        self.assertEqual(again, "oke jadi ini bagus atau enggak ya")

    def test_sound_event_does_not_ask(self):
        p = TurnPolicy()
        out = p.should_ask(
            ev(type="sound", text="[finger snapping]", event="finger snapping", utterance_id=7)
        )
        self.assertIsNone(out)


class CompletenessTests(unittest.TestCase):
    def test_fragments_are_not_complete(self):
        self.assertFalse(looks_complete("I'm not"))
        self.assertFalse(looks_complete("mas"))
        self.assertFalse(looks_complete("masih makan"))

    def test_finished_lines_are_complete(self):
        self.assertTrue(looks_complete("jam berapa sekarang ya?"))
        self.assertTrue(looks_complete("oke jadi ini bagus sekali"))
        self.assertTrue(looks_complete("唔該，啲咩事啊"))


if __name__ == "__main__":
    unittest.main()
