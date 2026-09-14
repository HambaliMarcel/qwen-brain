from __future__ import annotations

import socket
import threading
import time
import unittest

from qwen_brain.bus import JsonlHub, SttBusClient
from qwen_brain.events import dump_event


class BusTests(unittest.TestCase):
    def test_roundtrip(self):
        hub = JsonlHub("127.0.0.1", 0)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        hub.port = port
        hub.start()
        time.sleep(0.05)
        got = []
        done = threading.Event()

        def consume():
            client = SttBusClient("127.0.0.1", port)
            for ev in client.iter_events():
                got.append(ev)
                client.stop()
                done.set()
                return

        t = threading.Thread(target=consume, daemon=True)
        t.start()
        deadline = time.time() + 2.0
        while hub.client_count() == 0 and time.time() < deadline:
            time.sleep(0.02)
        hub.publish({"v": 1, "type": "commit", "text": "tes", "utterance_id": 9})
        self.assertTrue(done.wait(2.0), dump_event({"missed": True}))
        hub.close()
        self.assertEqual(got[0].type, "commit")
        self.assertEqual(got[0].text, "tes")
        self.assertEqual(got[0].utterance_id, 9)


if __name__ == "__main__":
    unittest.main()
