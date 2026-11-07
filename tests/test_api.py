"""
Tests for api/predict.py. They use a throwaway model with arbitrary coefficients, written to a
temporary folder, so they check the plumbing and say nothing about accuracy.
The package's own tests (portable_rekurrensparese/tests) check the features against training.
"""
import io
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api import predict  # noqa: E402
import rp_model  # noqa: E402

BOUNDARY = "testboundary123"
CONTENT_TYPE = f"multipart/form-data; boundary={BOUNDARY}"
EXTRACTION = {"sr": 44100, "resample_quality": "VHQ", "pitch_floor_hz": 75, "pitch_ceiling_hz": 600,
              "max_formants": 5, "max_formant_hz": 5500, "n_mfcc": 13,
              "cpp": {"frame_s": 0.04, "hop_s": 0.01, "f0_min_hz": 60, "f0_max_hz": 330, "fit_from_s": 0.001},
              "spectrum_max_hz": 4000}


def throwaway_model():
    features = [f"a_n__{name}" for name in rp_model.PER_RECORDING] + ["sex_female", "sex_male"]
    rng = np.random.default_rng(0)
    return {"model": "test_only", "variant": "test", "features": features,
            "impute_median": [0.0] * len(features), "scale_mean": [0.0] * len(features),
            "scale_std": [100.0] * len(features), "coef": rng.normal(size=len(features)).tolist(),
            "intercept": 0.0, "platt": {"a": 1.0, "b": 0.0}, "audio": {"recordings": ["a_n"]},
            "reference_medians": {}, "caveats": [], "extraction": EXTRACTION}


@pytest.fixture
def model(tmp_path, monkeypatch):
    path = tmp_path / "test_model.json"
    path.write_text(json.dumps(throwaway_model()))
    monkeypatch.setenv("RP_MODEL", str(path))  # An absolute path overrides the model folder.
    predict._load_model.cache_clear()
    yield rp_model.load_model(path)
    predict._load_model.cache_clear()


def wav_bytes(signal, rate=48000, channels=1):
    pcm = np.clip(signal * 32767, -32768, 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(channels), out.setsampwidth(2), out.setframerate(rate), out.writeframes(pcm.tobytes())
    return buffer.getvalue()


def vowel(seconds=2.5, f0=180.0, rate=48000):
    t = np.arange(int(seconds * rate)) / rate
    return 0.5 * sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12)) / 3


def form(**fields):
    body = b""
    for name, value in fields.items():
        if isinstance(value, bytes):
            head = f'name="{name}"; filename="a.wav"\r\nContent-Type: audio/wav'
        else:
            head, value = f'name="{name}"', value.encode()
        body += f'--{BOUNDARY}\r\nContent-Disposition: form-data; {head}\r\n\r\n'.encode() + value + b"\r\n"
    return body + f"--{BOUNDARY}--\r\n".encode()


def test_valid_recording_matches_contract(model):
    code, payload = predict.score(CONTENT_TYPE, form(audio=wav_bytes(vowel()), sex="female"))
    assert code == 200
    assert set(payload) == {"probability", "top_features"}
    assert 0 <= payload["probability"] <= 1
    assert len(payload["top_features"]) == 5
    for item in payload["top_features"]:
        assert set(item) == {"name", "label", "direction", "size"}
        assert item["direction"] in ("up", "down") and item["size"] >= 0
        assert item["label"] != item["name"]
    json.dumps(payload)


def test_response_agrees_with_the_package(model):
    data = wav_bytes(vowel())
    _, payload = predict.score(CONTENT_TYPE, form(audio=data, sex="male"))
    direct = rp_model.predict({"a_n": rp_model.read_wav(data)}, "male", model)
    assert payload["probability"] == direct["probability_rekurrensparese"]
    for ours, theirs in zip(payload["top_features"], direct["top_contributions"]):
        assert ours["name"] == theirs["feature"]
        assert ours["size"] == abs(theirs["contribution"])
        assert (ours["direction"] == "up") == (theirs["contribution"] >= 0)


def test_sex_changes_the_score(model):
    data = wav_bytes(vowel())
    _, female = predict.score(CONTENT_TYPE, form(audio=data, sex="female"))
    _, male = predict.score(CONTENT_TYPE, form(audio=data, sex="male"))
    assert female["probability"] != male["probability"]


def test_multipart_keeps_binary_audio_exact():
    # Bytes that a text-based parser could alter, including ones that look like line endings.
    data = bytes(range(256)) * 200 + b"\r\n\r\n--almost\r\n\n\r\r" + np.random.default_rng(1).bytes(50000) + b"\r\n"
    assert predict.parse_form(CONTENT_TYPE, form(audio=data, sex="male"))["audio"] == data


def test_stereo_and_other_sample_rates_work(model):
    mono = vowel(rate=22050)
    stereo = np.column_stack([mono, mono]).ravel()
    code, _ = predict.score(CONTENT_TYPE, form(audio=wav_bytes(stereo, rate=22050, channels=2), sex="female"))
    assert code == 200


@pytest.mark.parametrize("fields, code, text", [
    ({"sex": "female"}, 400, "No audio"),
    ({"audio": wav_bytes(vowel(0.5))}, 400, "female or male"),
    ({"audio": wav_bytes(vowel(0.5)), "sex": "other"}, 400, "female or male"),
    ({"audio": b"this is not a wav file", "sex": "male"}, 422, "PCM WAV"),
    ({"audio": wav_bytes(vowel(0.1)), "sex": "male"}, 422, "too short"),
    ({"audio": wav_bytes(vowel(31, rate=8000), rate=8000), "sex": "male"}, 422, "too long"),
    ({"audio": wav_bytes(np.zeros(48000)), "sex": "male"}, 422, "silent"),
    ({"audio": wav_bytes(np.random.default_rng(2).uniform(-0.4, 0.4, 96000)), "sex": "male"}, 422, "detect a voice"),
])
def test_bad_requests_get_plain_errors(model, fields, code, text):
    status, payload = predict.score(CONTENT_TYPE, form(**fields))
    assert status == code
    assert text in payload["error"]


def test_wrong_content_type_is_rejected(model):
    code, _ = predict.score("application/json", b"{}")
    assert code == 400


def test_missing_model_returns_error_not_score(tmp_path, monkeypatch):
    monkeypatch.setenv("RP_MODEL", str(tmp_path / "absent.json"))
    code, payload = predict.score(CONTENT_TYPE, form(audio=wav_bytes(vowel()), sex="female"))
    assert code == 503
    assert "isn't installed" in payload["error"]
    assert "probability" not in payload
    assert predict.status()["model_installed"] is False


def test_internal_errors_hide_details(model, monkeypatch):
    def fail(*args):
        raise RuntimeError("secret path /var/task/x")
    monkeypatch.setattr(rp_model, "predict", fail)
    code, payload = predict.score(CONTENT_TYPE, form(audio=wav_bytes(vowel()), sex="female"))
    assert code == 500
    assert "secret" not in payload["error"]


def test_every_model_feature_has_a_plain_label():
    for name in rp_model.PER_RECORDING + ["sex_female", "sex_male"]:
        assert predict.feature_label(f"a_n__{name}") != f"a_n__{name}"


def test_vercel_dependencies_match_the_package():
    read = lambda path: {line for line in path.read_text().splitlines() if line and not line.startswith("#")}
    assert read(ROOT / "requirements.txt") == read(ROOT / "portable_rekurrensparese" / "requirements.txt")


@pytest.mark.skipif(not predict.model_file().is_file(), reason="real model not installed")
def test_installed_model_is_the_a_n_model():
    model = rp_model.load_model(predict.model_file())
    assert model["audio"]["recordings"] == ["a_n"]
    assert len(model["features"]) == 30
    assert model["validated_performance"]["age_matched_pairs"]["auc"] > 0
