from __future__ import annotations

import unittest

from qwen_brain.events import SttEvent, is_command_text, parse_event, same_turn
from qwen_brain.llm import split_think_stream, strip_think
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
        self.assertFalse(is_command_text("um"))

    def test_same_turn(self):
        self.assertTrue(same_turn("jam berapa", "jam berapa sekarang"))
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
                text="jam berapa",
                speaking=False,
                decoding=False,
                silence_sec=0.6,
                utterance_id=4,
            )
        )
        self.assertEqual(out, "jam berapa")

    def test_commit_after_eager_same_text_skipped(self):
        p = TurnPolicy(eager=True, eager_silence_sec=0.5)
        p.should_ask(
            ev(type="live", text="jam berapa", speaking=False, silence_sec=0.7, utterance_id=4)
        )
        out = p.should_ask(ev(type="commit", text="jam berapa", utterance_id=4))
        self.assertIsNone(out)

    def test_commit_longer_text_restarts(self):
        p = TurnPolicy(eager=True, eager_silence_sec=0.5)
        p.should_ask(
            ev(type="live", text="jam berapa", speaking=False, silence_sec=0.7, utterance_id=4)
        )
        out = p.should_ask(ev(type="commit", text="jam berapa sekarang", utterance_id=4))
        self.assertEqual(out, "jam berapa sekarang")


if __name__ == "__main__":
    unittest.main()
