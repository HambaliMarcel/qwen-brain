from __future__ import annotations

import unittest

from qwen_brain.config import BrainConfig
from qwen_brain.events import (
    SttEvent,
    is_command_text,
    is_short_fragment,
    join_fragments,
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
        self.assertFalse(is_command_text("ooh"))
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
            def _stream(self, messages, gen=0, stats=None, max_tokens=None):
                self.seen = messages
                yield "ready"

        brain = StubBrain(BrainConfig())
        text, _stats = brain.ask("draft words", remember=False)
        self.assertEqual(text, "ready")
        self.assertEqual(brain.history, [])
        self.assertEqual(brain.seen[-1]["role"], "user")
        # Plain user text: a per-turn steer suffix would bust the KV prefix
        # cache of the previous exchange on every request.
        self.assertEqual(brain.seen[-1]["content"], "draft words")

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
        self.assertIn("obey", text)
        self.assertIn("toxic", text)
        self.assertIn("detail", text)


class ReplyBudgetTests(unittest.TestCase):
    def test_casual_stays_short(self):
        from qwen_brain.config import BrainConfig
        from qwen_brain.llm import reply_token_budget

        cfg = BrainConfig()
        self.assertEqual(reply_token_budget("oke", cfg), 24)
        self.assertEqual(reply_token_budget("I'm in love with you", cfg), 48)


class HistoryTrimTests(unittest.TestCase):
    def test_history_is_dropped_in_blocks_and_starts_on_a_user_turn(self):
        class StubBrain(LlamaBrain):
            def _stream(self, messages, gen=0, stats=None, max_tokens=None):
                yield "ok"

        cfg = BrainConfig()
        cfg.history_turns = 5
        brain = StubBrain(cfg)
        for i in range(5):
            brain.ask(f"u{i}")
        self.assertEqual(len(brain.history), 10)
        brain.ask("u5")
        # 12 > cap 10 -> down to floor (10 - 2*2 = 6), so the prefix is now
        # stable for the next two turns instead of shifting every turn.
        self.assertEqual(len(brain.history), 6)
        self.assertEqual(brain.history[0].role, "user")
        self.assertEqual(brain.history[0].content, "u3")
        brain.ask("u6")
        self.assertEqual(len(brain.history), 8)
        self.assertEqual(brain.history[0].content, "u3")

    def test_warm_does_not_cancel_a_real_turn(self):
        class StubBrain(LlamaBrain):
            def _stream(self, messages, gen=0, stats=None, max_tokens=None):
                yield "ok"

        brain = StubBrain(BrainConfig())
        gen_before = brain._gen
        brain.warm()
        self.assertEqual(brain._gen, gen_before)
        self.assertEqual(brain.history, [])
        text, stats = brain.ask("halo")
        self.assertEqual(text, "ok")
        self.assertFalse(stats.cancelled)


class LoopGuardTests(unittest.TestCase):
    def test_decoder_spiral_is_degenerate(self):
        from qwen_brain.events import collapse_loops, is_degenerate

        loop = " ".join(["black on"] * 40)
        collapsed, looped = collapse_loops(loop)
        self.assertTrue(looped)
        self.assertEqual(collapsed, "black on black on")
        self.assertTrue(is_degenerate(loop))
        self.assertTrue(is_degenerate("Yeah, " + ", ".join(["baby"] * 30) + "."))

    def test_sung_hook_is_not_a_loop(self):
        from qwen_brain.events import collapse_loops, is_degenerate

        hook = "no love, no love, no love, we don't need it"
        self.assertEqual(collapse_loops(hook), (hook, False))
        self.assertFalse(is_degenerate(hook))
        self.assertFalse(is_degenerate("pesawat pesawat pesawat pesawat"))
        self.assertEqual(collapse_loops("na na na na na hey")[1], False)

    def test_only_the_unseen_tail_becomes_a_turn(self):
        from qwen_brain.events import strip_already_sent

        sent = "I still feel a shock through every bone when I hear an I love you."
        grown = "I'm not sure. " + sent + " How did I fall in love this time?"
        self.assertEqual(
            strip_already_sent(sent, grown),
            "I'm not sure. How did I fall in love this time?",
        )
        self.assertEqual(strip_already_sent(sent, sent), "")
        self.assertEqual(strip_already_sent("oke", "oke sip lanjut"), "oke sip lanjut")
        self.assertEqual(strip_already_sent(sent, "totally new line here"), "totally new line here")

    def test_detail_ask_gets_a_long_budget(self):
        from qwen_brain.config import BrainConfig
        from qwen_brain.llm import reply_token_budget

        cfg = BrainConfig()
        self.assertEqual(reply_token_budget("jelasin dong secara detail", cfg), 384)
        self.assertEqual(reply_token_budget("explain step by step how MTP works", cfg), 384)

    def test_long_prompt_gets_a_long_budget(self):
        from qwen_brain.config import BrainConfig
        from qwen_brain.llm import reply_token_budget

        cfg = BrainConfig()
        long = " ".join(["lagu"] * 45)
        self.assertEqual(reply_token_budget(long, cfg), 384)


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
        self.assertFalse(looks_complete("What."))
        self.assertFalse(looks_complete("You."))
        self.assertFalse(looks_complete("Every."))
        self.assertFalse(looks_complete("Okay."))

    def test_finished_lines_are_complete(self):
        self.assertTrue(looks_complete("jam berapa sekarang ya?"))
        self.assertTrue(looks_complete("oke jadi ini bagus sekali"))
        self.assertTrue(looks_complete("唔該，啲咩事啊"))

    def test_short_fragments_stitch(self):
        self.assertTrue(is_short_fragment("What."))
        self.assertTrue(is_short_fragment("Every day."))
        self.assertFalse(is_short_fragment("I can't wait a moment more"))
        self.assertEqual(join_fragments("Every.", "Every day."), "Every day.")
        self.assertEqual(join_fragments("Every day.", "To a joy."), "Every day. To a joy.")


if __name__ == "__main__":
    unittest.main()
