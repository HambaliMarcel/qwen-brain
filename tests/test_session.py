from __future__ import annotations

import unittest

from qwen_brain.config import BrainConfig
from qwen_brain.events import SttEvent
from qwen_brain.session import AssistantSession, FINALIZE_PAUSE_SEC, HOLD_PAUSE_SEC


class FakeBrain:
    def __init__(self) -> None:
        self.cancelled = 0
        self.remembered: list[tuple[str, str]] = []

    def cancel(self) -> None:
        self.cancelled += 1

    def remember_turn(self, user_text: str, assistant_text: str) -> None:
        self.remembered.append((user_text, assistant_text))


def event(**values) -> SttEvent:
    data = {
        "type": "live",
        "text": "",
        "display": "",
        "utterance_id": 1,
        "speaking": False,
    }
    data.update(values)
    return SttEvent.from_dict(data)


class CanonicalTurnTests(unittest.TestCase):
    def make_session(self) -> AssistantSession:
        return AssistantSession(BrainConfig(), brain=FakeBrain())

    def test_event_only_does_not_submit_a_scene_turn(self):
        session = self.make_session()
        session._on_event(
            event(
                type="sound",
                text="[cough]",
                display="[cough]",
                event="cough",
                non_speech_only=True,
            )
        )
        self.assertEqual(session._job, 0)
        self.assertEqual(session._publish_job, 0)
        self.assertEqual(session.ui.you, "")
        self.assertEqual(session._last_event, "cough")

    def test_event_cooldown_does_not_spam(self):
        session = self.make_session()
        session._on_event(event(type="sound", text="[cough]", event="cough"))
        session._on_event(event(type="sound", text="[cough]", event="cough"))
        self.assertEqual(session._job, 0)

    def test_spoken_turn_still_finalizes_without_waiting_for_a_tag(self):
        session = self.make_session()
        session._on_event(event(type="commit", text="are you listening", utterance_id=2))
        session._flush_held_turn()
        self.assertEqual(session._latest_prompt, "are you listening")
        self.assertEqual(session.ui.you, "are you listening")
        self.assertEqual(session._publish_job, session._job)

    def test_sound_during_speech_does_not_steal_the_turn(self):
        session = self.make_session()
        session._on_event(
            event(
                type="sound",
                text="[crowing]",
                display="[crowing] serangnya berapa?",
                event="crowing",
                speaking=True,
                utterance_id=8,
            )
        )
        self.assertEqual(session._job, 0)
        self.assertEqual(session._last_event, "crowing")

    def test_matching_last_promotes_hidden_speculation(self):
        session = self.make_session()
        session._job = 4
        session._job_uid = 7
        session.policy.sent_uid = 7
        session.policy.sent_text = "hello there"
        session._on_event(event(type="commit", text="hello there", utterance_id=7))
        self.assertEqual(session._job, 4)
        self.assertEqual(session._publish_job, 4)
        self.assertEqual(session.ui.you, "hello there")

    def test_duplicate_commit_is_not_a_second_turn(self):
        session = self.make_session()
        session._on_event(event(type="commit", text="apa kabar", utterance_id=3))
        session._flush_held_turn()
        first = session._job
        session._on_event(event(type="commit", text="apa kabar", utterance_id=3))
        self.assertEqual(session._job, first)

    def test_short_gap_does_not_finalize_a_draft(self):
        session = self.make_session()
        session._on_event(
            event(
                type="live",
                text="I'm not",
                speaking=False,
                silence_sec=0.08,
                gap_sec=0.08,
                utterance_id=9,
            )
        )
        self.assertEqual(session._job, 0)
        self.assertEqual(session.ui.you, "")

    def test_incomplete_waits_for_a_real_hold(self):
        session = self.make_session()
        session._on_event(
            event(
                type="live",
                text="I'm not",
                speaking=False,
                silence_sec=FINALIZE_PAUSE_SEC + 0.02,
                gap_sec=1.0,
                utterance_id=9,
            )
        )
        self.assertEqual(session._job, 0)
        session._on_event(
            event(
                type="live",
                text="I'm not",
                speaking=False,
                silence_sec=HOLD_PAUSE_SEC + 0.02,
                gap_sec=1.0,
                utterance_id=9,
            )
        )
        self.assertGreaterEqual(session._job, 1)
        self.assertEqual(session._publish_job, session._job)
        self.assertEqual(session.ui.you, "I'm not")

    def test_complete_sentence_stays_hidden_until_hold(self):
        session = self.make_session()
        session._on_event(
            event(
                type="live",
                text="jam berapa sekarang ya?",
                speaking=False,
                decoding=False,
                silence_sec=FINALIZE_PAUSE_SEC + 0.02,
                utterance_id=9,
            )
        )
        self.assertEqual(session._job, 0)
        session._flush_held_turn()
        self.assertGreaterEqual(session._job, 1)
        self.assertEqual(session._publish_job, 0)
        self.assertEqual(session.ui.you, "")
        session._on_event(
            event(
                type="live",
                text="jam berapa sekarang ya?",
                speaking=False,
                decoding=False,
                silence_sec=HOLD_PAUSE_SEC + 0.02,
                utterance_id=9,
            )
        )
        self.assertEqual(session._publish_job, session._job)
        self.assertEqual(session.ui.you, "jam berapa sekarang ya?")

    def test_language_leak_is_not_a_turn(self):
        session = self.make_session()
        session._on_event(
            event(
                type="live",
                text="language Canton",
                speaking=False,
                silence_sec=HOLD_PAUSE_SEC + 0.05,
                utterance_id=11,
            )
        )
        self.assertEqual(session._job, 0)

    def test_complete_while_speaking_does_not_start(self):
        session = self.make_session()
        session._stable_text = "jam berapa sekarang ya?"
        session._stable_at = 0.0
        session._on_event(
            event(
                type="live",
                text="jam berapa sekarang ya?",
                speaking=True,
                silence_sec=0.0,
                gap_sec=0.0,
                utterance_id=10,
            )
        )
        self.assertEqual(session._job, 0)
        self.assertEqual(session._publish_job, 0)

    def test_short_commits_are_stitched_before_the_brain(self):
        session = self.make_session()
        session._on_event(event(type="commit", text="Every.", utterance_id=20))
        session._on_event(event(type="commit", text="Every day.", utterance_id=21))
        self.assertEqual(session._job, 0)
        session._on_event(event(type="commit", text="To a joy.", utterance_id=22))
        self.assertGreaterEqual(session._job, 1)
        self.assertEqual(session._latest_prompt, "Every day. To a joy.")
        self.assertEqual(session.ui.you, "Every day. To a joy.")

    def test_long_commit_still_fires_immediately(self):
        session = self.make_session()
        session._on_event(
            event(type="commit", text="I can't wait a moment more", utterance_id=30)
        )
        self.assertGreaterEqual(session._job, 1)
        self.assertEqual(session._latest_prompt, "I can't wait a moment more")

    def test_live_growth_does_not_print_a_second_you(self):
        session = self.make_session()
        session._on_event(
            event(
                type="live",
                text="jam berapa sekarang ya?",
                speaking=False,
                decoding=False,
                silence_sec=HOLD_PAUSE_SEC + 0.02,
                utterance_id=12,
            )
        )
        first = session.ui.you
        job = session._job
        session._on_event(
            event(
                type="live",
                text="jam berapa sekarang ya ini",
                speaking=False,
                decoding=False,
                silence_sec=HOLD_PAUSE_SEC + 0.02,
                utterance_id=12,
            )
        )
        self.assertEqual(session.ui.you, first)
        self.assertEqual(session._publish_job, job)


if __name__ == "__main__":
    unittest.main()
