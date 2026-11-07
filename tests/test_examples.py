"""The committed example speakers: scores must match the model, and no session IDs may appear."""
import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "portable_rekurrensparese"))
import rp_model  # noqa: E402

EXAMPLES_DIR = ROOT / "ui" / "examples"
PAYLOAD = json.loads((EXAMPLES_DIR / "examples.json").read_text())
ALLOWED_FIELDS = {"file", "sex", "group", "validatedProbability", "result"}


def test_only_expected_fields():
    for example in PAYLOAD["examples"]:
        assert set(example) == ALLOWED_FIELDS


def test_wavs_hold_only_format_and_audio():
    wavs = list(EXAMPLES_DIR.glob("*.wav")) + list((ROOT / "drag_and_drop").glob("*.wav"))
    assert len(wavs) == 8
    for path in wavs:
        data = path.read_bytes()
        position, chunks = 12, []
        while position + 8 <= len(data):
            size = struct.unpack("<I", data[position + 4:position + 8])[0]
            chunks.append(data[position:position + 4])
            position += 8 + size + (size & 1)
        assert chunks == [b"fmt ", b"data"], path.name


@pytest.mark.parametrize("example", PAYLOAD["examples"], ids=lambda example: example["file"])
def test_saved_score_matches_model(example):
    model = rp_model.load_model(ROOT / "portable_rekurrensparese" / "model" / PAYLOAD["model"])
    audio = rp_model.read_wav((EXAMPLES_DIR / example["file"]).read_bytes())
    live = rp_model.predict({"a_n": audio}, example["sex"], model)
    assert live["probability_rekurrensparese"] == pytest.approx(example["result"]["probability"], abs=0.001)
