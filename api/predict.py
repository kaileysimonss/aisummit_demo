"""
POST /api/predict: score one sustained /a/ with the Rekurrensparese logistic regression.

Implements docs/api_contract.md (multipart request, plain-English response) on top of the
unmodified portable_rekurrensparese package. Audio is only held in memory for the request.

The model file is portable_rekurrensparese/model/<RP_MODEL>. The default is the
age/sex-weighted a_n model that the package README recommends.
GET /api/predict reports whether the model is installed.
"""
import email
import json
import os
import sys
import wave
from functools import lru_cache
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "portable_rekurrensparese"
sys.path.insert(0, str(PACKAGE))
import rp_model  # noqa: E402

DEFAULT_MODEL = "rekurrensparese_an_lr_weighted.json"
MAX_BODY_BYTES = 4_400_000  # Vercel rejects request bodies above about 4.5 MB
MIN_SECONDS = 0.3           # same minimum as the package's own API
MAX_SECONDS = 30
MIN_PEAK = 0.005
# Deliberately low. Paralysis can reduce voicing, so this only rejects clips with almost none.
MIN_VOICED_FRACTION = 0.1

LABELS = {
    "shimmer_local": "Shimmer (loudness stability)",
    "hnr_db": "HNR (voice vs. breath noise)",
    "f0_mean_hz": "Average pitch",
    "f1_mean_hz": "F1 (first vowel resonance)",
    "f2_mean_hz": "F2 (second vowel resonance)",
    "f3_mean_hz": "F3 (third vowel resonance)",
    "mfcc1_mean": "MFCC 1 (recording level and tone)",
    **{f"mfcc{i}_mean": f"MFCC {i} (voice timbre)" for i in range(2, 14)},
    "cpp_db": "CPP (voice clarity)",
    "h1_h2_db": "H1-H2 (breathiness)",
    "spectral_slope_db_khz": "Spectral slope (voice brightness)",
    "alpha_ratio_db": "Alpha ratio (low vs. high energy)",
    "f0_p5_st": "Lowest pitch (5th percentile)",
    "f0_p95_st": "Highest pitch (95th percentile)",
    "f0_range_st": "Pitch range",
    "f0_step_st": "Pitch steadiness (moment to moment)",
    "voiced_fraction": "Voiced fraction (share of clip with voicing)",
    "sex_female": "Sex",
    "sex_male": "Sex",
}


class RequestError(Exception):
    """A problem to report to the user: status plus a plain-language message."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def model_file():
    return PACKAGE / "model" / os.environ.get("RP_MODEL", DEFAULT_MODEL)


@lru_cache(maxsize=1)
def _load_model(path):
    return rp_model.load_model(path)


def get_model():
    path = model_file()
    if not path.is_file():
        print(f"Model file not found: {path}", file=sys.stderr)
        raise RequestError(503, "The screening model isn't installed on the server yet, so no score can be given.")
    return _load_model(path)


def feature_label(name):
    return LABELS.get(name.split("__", 1)[-1], name)


def parse_form(content_type, body):
    """Fields of a multipart/form-data body, as {name: bytes}."""
    if not content_type or not content_type.lower().startswith("multipart/form-data"):
        raise RequestError(400, "Send the request as multipart/form-data with audio and sex fields.")
    message = email.message_from_bytes(b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + body)
    if not message.is_multipart():
        raise RequestError(400, "The request body couldn't be read. Please try again.")
    fields = {}
    for part in message.get_payload():
        name = part.get_param("name", header="content-disposition")
        if name:
            fields[name] = part.get_payload(decode=True) or b""
    return fields


def read_audio(data):
    """Signal and sample rate, after rejecting audio the model can't meaningfully score."""
    try:
        signal, rate = rp_model.read_wav(data)
    except (wave.Error, EOFError, ValueError) as error:
        print(f"Unreadable audio: {error}", file=sys.stderr)
        raise RequestError(422, "That file couldn't be read. Please use an uncompressed PCM WAV file.") from error
    if len(signal) < MIN_SECONDS * rate:
        raise RequestError(422, "That recording is too short. Please hold the sound for about 2 seconds.")
    if len(signal) > MAX_SECONDS * rate:
        raise RequestError(422, f"That recording is too long. Please use one held “ah” under {MAX_SECONDS} seconds.")
    if np.abs(signal).max() < MIN_PEAK:
        raise RequestError(422, "That recording is silent or nearly silent. Please check the microphone and try again.")
    return signal, rate


def require_voice(signal, rate, model):
    """Without this, a clip with no voice would get a score built from imputed training medians."""
    import parselmouth

    settings = model["extraction"]
    pitch = parselmouth.Sound(signal, sampling_frequency=rate).to_pitch(
        pitch_floor=settings["pitch_floor_hz"], pitch_ceiling=settings["pitch_ceiling_hz"])
    if (pitch.selected_array["frequency"] > 0).mean() < MIN_VOICED_FRACTION:
        raise RequestError(422, "We couldn’t detect a voice in that recording. Please hold a steady “ah” and try again.")


def to_response(result):
    """Converts the package's result to the shape in docs/api_contract.md."""
    return {
        "probability": result["probability_rekurrensparese"],
        "top_features": [{
            "name": item["feature"],
            "label": feature_label(item["feature"]),
            "direction": "up" if item["contribution"] >= 0 else "down",
            "size": abs(item["contribution"]),
        } for item in result["top_contributions"]],
    }


def score(content_type, body):
    """Returns (HTTP status, JSON-ready payload)."""
    try:
        model = get_model()
        fields = parse_form(content_type, body)
        sex = fields.get("sex", b"").decode("utf-8", "replace").strip()
        if sex not in ("female", "male"):
            raise RequestError(400, "Please choose female or male.")
        if not fields.get("audio"):
            raise RequestError(400, "No audio file was received. Please add a recording.")
        signal, rate = read_audio(fields["audio"])
        require_voice(signal, rate, model)
        return 200, to_response(rp_model.predict({"a_n": (signal, rate)}, sex, model))
    except RequestError as error:
        return error.status, {"error": error.message}
    except Exception as error:  # Details go to the server log, never to the page.
        print(f"Scoring failed: {error!r}", file=sys.stderr)
        return 500, {"error": "Something went wrong while scoring this recording. Please try again."}


def status():
    """Model status for GET requests, without any user data."""
    path = model_file()
    if not path.is_file():
        return {"model_installed": False, "expected_file": path.name}
    model = _load_model(path)
    return {"model_installed": True, "model": model["model"], "variant": model["variant"],
            "recordings": model["audio"]["recordings"],
            "validated_performance": model.get("validated_performance", {})}


def send_json(request, code, payload):
    data = json.dumps(payload).encode()
    request.send_response(code)
    request.send_header("Content-Type", "application/json")
    request.send_header("Content-Length", str(len(data)))
    request.send_header("Cache-Control", "no-store")
    request.end_headers()
    request.wfile.write(data)


def serve_post(request):
    try:
        length = int(request.headers.get("Content-Length", 0))
    except ValueError:
        length = 0
    if length > MAX_BODY_BYTES:
        request.close_connection = True  # The unread body can't be reused for another request.
        return send_json(request, 413, {"error": "That file is too large. Please keep it under 4 MB."})
    code, payload = score(request.headers.get("Content-Type", ""), request.rfile.read(length))
    send_json(request, code, payload)


class handler(BaseHTTPRequestHandler):  # Vercel looks for this exact class name.
    def do_POST(self):
        serve_post(self)

    def do_GET(self):
        send_json(self, 200, status())
