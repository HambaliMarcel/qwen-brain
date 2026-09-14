from __future__ import annotations

import unittest

from qwen_brain.ui import you_parts


class YouPartsTests(unittest.TestCase):
    def test_live_only_is_draft(self):
        black, draft = you_parts("", "halo kamu", "")
        self.assertEqual(black, "")
        self.assertEqual(draft, "halo kamu")

    def test_committed_stays_black(self):
        black, draft = you_parts("halo kamu", "halo kamu", "")
        self.assertEqual(black, "halo kamu")
        self.assertEqual(draft, "")

    def test_live_grows_after_commit(self):
        black, draft = you_parts("halo", "halo kamu", "")
        self.assertEqual(black, "halo")
        self.assertEqual(draft, "kamu")

    def test_event_prefix_once_on_commit(self):
        black, draft = you_parts("halo", "halo", "batuk?")
        self.assertEqual(black, "[batuk?] halo")
        self.assertEqual(draft, "")

    def test_event_only_commit(self):
        black, draft = you_parts("[batuk?]", "[batuk?]", "batuk?")
        self.assertEqual(black, "[batuk?]")
        self.assertEqual(draft, "")

    def test_event_on_live_draft(self):
        black, draft = you_parts("", "kamu tahu", "typing")
        self.assertEqual(black, "")
        self.assertEqual(draft, "[typing] kamu tahu")


class TurnBufferTests(unittest.TestCase):
    def test_you_then_brain(self):
        from qwen_brain.ui import BrainUI

        ui = BrainUI()
        ui._scrollback = False
        ui.set_you("[batuk?]")
        ui._commit_open_you()
        ui.brain = "Kamu batuk ya?"
        ui._brain_open = True
        ui._brain_stamp = "00:00:02"
        ui._commit_open_brain("Kamu batuk ya?")
        self.assertEqual([kind for kind, _stamp, _text in ui._turns], ["YOU", "BRAIN"])
        self.assertEqual(ui._turns[0][2], "[batuk?]")
        self.assertFalse(ui._you_open)
        self.assertFalse(ui._brain_open)

    def test_stream_starts_fresh_brain(self):
        from qwen_brain.llm import StreamStats
        from qwen_brain.ui import BrainUI

        ui = BrainUI()
        ui.brain = "old answer"
        stats = StreamStats()
        ui.on_stream("Halo", stats)
        self.assertEqual(ui.brain, "Halo")
        ui.on_stream(" kamu", stats)
        self.assertEqual(ui.brain, "Halo kamu")


if __name__ == "__main__":
    unittest.main()
