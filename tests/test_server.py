from __future__ import annotations

import unittest

from qwen_brain.config import BrainConfig
from qwen_brain.server import build_server_cmd


class ServerCmdTests(unittest.TestCase):
    def test_default_speed_flags(self):
        cfg = BrainConfig()
        cmd = build_server_cmd(cfg)
        self.assertEqual(cfg.ctx, 3072)
        self.assertFalse(cfg.fit)
        self.assertIn("-c", cmd)
        self.assertEqual(cmd[cmd.index("-c") + 1], "3072")
        self.assertIn("-ub", cmd)
        self.assertEqual(cmd[cmd.index("-ub") + 1], "128")
        self.assertIn("-b", cmd)
        self.assertEqual(cmd[cmd.index("-b") + 1], "256")
        self.assertIn("--reasoning", cmd)
        self.assertEqual(cmd[cmd.index("--reasoning") + 1], "off")
        self.assertNotIn("--fit", cmd)
        self.assertIn("draft-mtp", cmd)
        self.assertIn("q4_0", cmd)

    def test_fit_only_when_enabled(self):
        cfg = BrainConfig(fit=True, ctx_locked=False)
        cmd = build_server_cmd(cfg)
        self.assertIn("--fit", cmd)
        self.assertNotIn("-c", cmd)
