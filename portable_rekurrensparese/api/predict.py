"""
Vercel Python function: POST /api/predict

Request JSON:  {"sex": "female" | "male", "recordings": {"a_n": "<base64 WAV>"}}
               (the recordings the model uses, default a_n only; 16-bit PCM WAV, any
               sample rate; the page in public/ converts browser recordings to WAV)
Response JSON: rp_model.predict(...) result, or {"error": "..."} with status 400.

Model file: model/<RP_MODEL> (environment variable; default the age-weighted a_n model).
GET /api/predict returns the model's recordings and validated performance for the page.
"""
import base64
import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rp_model  # noqa: E402

MODEL = rp_model.load_model(ROOT / "model" / os.environ.get("RP_MODEL", "rekurrensparese_an_lr_weighted.json"))
MAX_BYTES = 4_000_000   # Vercel limits request bodies to 4.5 MB


class handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"model": f"{MODEL['model']} ({MODEL['variant']})", "recordings": MODEL["audio"]["recordings"],
                         "validated_performance": MODEL.get("validated_performance", {}), "caveats": MODEL["caveats"]})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_BYTES:
                return self._send(413, {"error": "request too large; keep each recording under about 10 seconds"})
            req = json.loads(self.rfile.read(n))
            recs = {code: rp_model.read_wav(base64.b64decode(b64)) for code, b64 in req.get("recordings", {}).items()}
            for code, (x, sr) in recs.items():
                if len(x) < 0.3 * sr:
                    return self._send(400, {"error": f"{code} is shorter than 0.3 s"})
            self._send(200, rp_model.predict(recs, req.get("sex"), MODEL))
        except (ValueError, KeyError, json.JSONDecodeError, EOFError) as e:
            self._send(400, {"error": str(e)})
