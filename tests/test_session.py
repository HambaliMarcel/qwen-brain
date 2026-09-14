from __future__ import annotations

import unittest

from qwen_brain.config import BrainConfig
from qwen_brain.events import SttEvent
from qwen_brain.session import AssistantSession


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

    def test_event_only_is_context_not_a_brain_turn(self):
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
        self.assertEqual(session._last_event, "cough")

    def test_event_attaches_to_next_final_spoken_turn(self):
        session = self.make_session()
        session._on_event(event(type="sound", text="[cough]", event="cough"))
        session._on_event(event(type="commit", text="are you listening", utterance_id=2))
        self.assertEqual(session._latest_prompt, "[cough] are you listening")
        self.assertEqual(session.ui.you, "[cough] are you listening")
        self.assertEqual(session._publish_job, session._job)
        self.assertEqual(session._last_event, "")

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


if __name__ == "__main__":
    unittest.main()
