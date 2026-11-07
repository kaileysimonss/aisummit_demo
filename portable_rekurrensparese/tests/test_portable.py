"""The portable features and probabilities must equal the training code's (tests/golden.json,
computed in the training repository from the synthetic recordings in tests/fixtures/)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rp_model  # noqa: E402

GOLDEN = json.loads((ROOT / "tests" / "golden.json").read_text())


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["name"])
def test_matches_training_code(case):
    recs = {code: rp_model.read_wav(ROOT / "tests" / "fixtures" / f) for code, f in case["files"].items()}
    for variant, expected_p in case["probability"].items():
        model = rp_model.load_model(ROOT / "model" / GOLDEN["model_files"][variant])
        x = rp_model.feature_vector(recs, case["sex"], model)
        ref = np.array([np.nan if v is None else v for v in case["features"]], float)
        assert np.allclose(x, ref, rtol=GOLDEN["rtol"], atol=GOLDEN["atol"], equal_nan=True)
        assert float(rp_model.probability(x, model)) == pytest.approx(expected_p, abs=GOLDEN["prob_atol"])
        result = rp_model.predict(recs, case["sex"], model)
        assert result["recordings_missing_imputed"] == [c for c in model["audio"]["recordings"] if c not in case["files"]]
        assert result["validated_performance"]["age_matched_pairs"]["auc"] > 0


def test_wav_roundtrip(tmp_path):
    import wave
    x = (np.sin(np.arange(4800) / 10) * 20000).astype("<i2")
    with wave.open(str(tmp_path / "t.wav"), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(48000), w.writeframes(x.tobytes())
    y, sr = rp_model.read_wav(tmp_path / "t.wav")
    assert sr == 48000 and np.allclose(y, x / 32768)
    y2, _ = rp_model.read_wav((tmp_path / "t.wav").read_bytes())
    assert np.array_equal(y, y2)
