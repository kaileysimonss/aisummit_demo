"""
Local stand-in for Vercel: serves ui/ and /api/predict from one address.

    python dev_server.py [port]
"""
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from api import predict  # noqa: E402


class DevHandler(SimpleHTTPRequestHandler):
    def _is_api(self):
        return urlsplit(self.path).path == "/api/predict"

    def end_headers(self):
        # Makes the browser check for changes, so edits show up on a normal refresh.
        if not self._is_api():
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_POST(self):
        if self._is_api():
            predict.serve_post(self)
        else:
            self.send_error(404)

    def do_GET(self):
        if self._is_api():
            predict.send_json(self, 200, predict.status())
        else:
            super().do_GET()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    if predict.model_file().is_file():
        print(f"Model: {predict.model_file().name}")
    else:
        print(f"WARNING: no model at {predict.model_file()}\nThe page will load, but scoring returns an error.")
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(DevHandler, directory=str(ROOT / "ui")))
    print(f"Open http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
