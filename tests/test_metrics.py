from __future__ import annotations

import unittest

from qwen_brain.llm import StreamStats, apply_llama_timings
from qwen_brain.metrics import SessionMetrics, tok_per_sec


class MetricsTests(unittest.TestCase):
    def test_tok_per_sec(self):
        self.assertEqual(tok_per_sec(40, 1000), 40.0)
        self.assertEqual(tok_per_sec(0, 1000), 0.0)
        self.assertEqual(tok_per_sec(10, 0), 0.0)

    def test_record_turn_averages(self):
        m = SessionMetrics()
        m.record_turn(trigger="eager", ttft_ms=100, gen_ms=1000, e2e_ms=1200, tok_s=40)
        m.record_turn(trigger="commit", ttft_ms=200, gen_ms=2000, e2e_ms=2200, tok_s=20)
        self.assertEqual(m.turns, 2)
        self.assertEqual(m.eager_turns, 1)
        self.assertEqual(m.commit_turns, 1)
        self.assertEqual(m.avg_ttft_ms(), 150)
        self.assertEqual(m.avg_tok_s(), 30)

    def test_llama_timings(self):
        stats = StreamStats()
        apply_llama_timings(
            stats,
            {
                "timings": {
                    "prompt_n": 80,
                    "prompt_ms": 40.0,
                    "predicted_n": 32,
                    "predicted_ms": 800.0,
                    "predicted_per_second": 40.0,
                }
            },
        )
        self.assertEqual(stats.prompt_tokens, 80)
        self.assertEqual(stats.tokens, 32)
        self.assertEqual(stats.tok_s, 40.0)
        self.assertEqual(stats.prompt_ms, 40.0)


if __name__ == "__main__":
    unittest.main()
