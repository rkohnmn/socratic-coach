"""Regression coverage for the loopback HUD's HTTP security boundary."""
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from webui import HUD_HOST, HudHandler, WebBridge


class HttpSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = WebBridge()
        cls.httpd = ThreadingHTTPServer((HUD_HOST, 0), HudHandler)
        cls.httpd.daemon_threads = True
        cls.httpd.bridge = cls.bridge
        cls.origin = f"http://{HUD_HOST}:{cls.httpd.server_port}"
        cls.worker = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.worker.join(timeout=3)

    def request(self, path, *, method="GET", payload=None, headers=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.origin + path, data=data,
                                     method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def test_normal_local_get(self):
        status, content = self.request("/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(content)["ok"])

    def test_dns_rebinding_host_rejected_for_state_read(self):
        status, _ = self.request("/api/state", headers={"Host": "attacker.example:443"})
        self.assertEqual(status, 403)

    def test_cross_origin_get_rejected(self):
        status, _ = self.request("/api/state", headers={"Origin": "http://attacker.example"})
        self.assertEqual(status, 403)

    def test_cross_origin_action_rejected(self):
        status, _ = self.request(
            "/api/action", method="POST", payload={"action": "stop"},
            headers={"Origin": "http://attacker.example", "Content-Type": "application/json"})
        self.assertEqual(status, 403)
        self.assertIsNone(self.bridge.wait_action(timeout=0))

    def test_cross_origin_simple_content_type_rejected(self):
        status, _ = self.request(
            "/api/action", method="POST", payload={"action": "stop"},
            headers={"Content-Type": "text/plain"})
        self.assertEqual(status, 415)
        self.assertIsNone(self.bridge.wait_action(timeout=0))

    def test_same_origin_action_works(self):
        status, _ = self.request(
            "/api/action", method="POST", payload={"action": "dismiss"},
            headers={"Content-Type": "application/json", "Origin": self.origin})
        self.assertEqual(status, 200)
        self.assertEqual(self.bridge.wait_action(timeout=0.2), "dismiss")

    def test_native_vision_publisher_without_origin_works(self):
        status, _ = self.request(
            "/api/board", method="POST", payload={"equations": ["a=b"]},
            headers={"Content-Type": "application/json; charset=utf-8"})
        self.assertEqual(status, 200)
        self.assertEqual(self.bridge.snapshot()["board"]["equations"], ["a=b"])

    def test_cross_origin_board_publisher_rejected(self):
        original = self.bridge.snapshot()["board"]
        status, _ = self.request(
            "/api/board", method="POST", payload={"equations": ["tampered"]},
            headers={"Content-Type": "application/json",
                     "Origin": "http://attacker.example"})
        self.assertEqual(status, 403)
        self.assertEqual(self.bridge.snapshot()["board"], original)

    def test_malformed_json_shape_rejected(self):
        status, _ = self.request(
            "/api/action", method="POST", payload=["stop"],
            headers={"Content-Type": "application/json"})
        self.assertEqual(status, 400)
        self.assertIsNone(self.bridge.wait_action(timeout=0))

    def test_public_bind_rejected(self):
        from webui import HudServer
        server = HudServer(host="0.0.0.0", port=0)
        with self.assertRaisesRegex(ValueError, "127.0.0.1"):
            server.start()


if __name__ == "__main__":
    unittest.main()
