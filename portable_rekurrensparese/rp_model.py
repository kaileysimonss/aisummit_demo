"""
Portable inference for an exported logistic regression: healthy vs. Rekurrensparese
(vocal fold paralysis) from sustained /a/ ("aah") recordings. The model file says which
recordings it uses (model["audio"]["recordings"]; the default model: a_n, normal pitch,
only). Research prototype, not a diagnostic test.

Self-contained: numpy, praat-parselmouth and soxr only (no librosa / scipy / pandas,
so it fits a Vercel Python function). The features are the training features
(voice-ai-enabled-diagnosis-for-neuro-psych, rekurrensparese_v2.py): librosa's MFCC
and scipy's Welch spectrum are re-implemented here in numpy, and
tests/test_portable.py checks the whole path against values computed with the
training code (tests/golden.json). Extraction settings come from the model JSON.

    import rp_model
    model = rp_model.load_model("model/rekurrensparese_an_lr_weighted.json")
    result = rp_model.predict({"a_n": rp_model.read_wav("normal.wav")}, "female", model)
"""
import io
import json
import wave
from pathlib import Path

import numpy as np

RECORDINGS = ("a_n", "a_h", "a_l", "a_lhl")
MFCC = [f"mfcc{i}_mean" for i in range(1, 14)]
NUMERIC = ["shimmer_local", "hnr_db", "f0_mean_hz", "f1_mean_hz", "f2_mean_hz", "f3_mean_hz"] + MFCC
EXTRA = ["cpp_db", "h1_h2_db", "spectral_slope_db_khz", "alpha_ratio_db",
         "f0_p5_st", "f0_p95_st", "f0_range_st", "f0_step_st", "voiced_fraction"]
PER_RECORDING = NUMERIC + EXTRA


def load_model(path):
    m = json.loads(Path(path).read_text())
    if "extraction" not in m:
        raise ValueError("model JSON lacks the extraction settings (export it with the 'portable' step)")
    return m


# ---------- audio ----------

def read_wav(src):
    """(mono float64 signal in [-1, 1), sample rate) from a PCM WAV path or bytes (8/16/24/32-bit)."""
    with wave.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else str(src), "rb") as w:
        sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if width == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float64) - 128) / 128
    elif width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float64) / 32768
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(v >= 1 << 23, v - (1 << 24), v).astype(np.float64) / (1 << 23)
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float64) / (1 << 31)
    else:
        raise ValueError(f"unsupported sample width {width}")
    return x.reshape(-1, ch).mean(axis=1), sr


# ---------- numpy re-implementations of the librosa / scipy defaults used in training ----------

_F_SP, _MIN_LOG_HZ = 200.0 / 3, 1000.0
_MIN_LOG_MEL, _LOGSTEP = _MIN_LOG_HZ / _F_SP, np.log(6.4) / 27.0


def _hz_to_mel(f):
    f = np.asarray(f, float)
    return np.where(f >= _MIN_LOG_HZ, _MIN_LOG_MEL + np.log(np.maximum(f, 1e-10) / _MIN_LOG_HZ) / _LOGSTEP, f / _F_SP)


def _mel_to_hz(m):
    m = np.asarray(m, float)
    return np.where(m >= _MIN_LOG_MEL, _MIN_LOG_HZ * np.exp(_LOGSTEP * (m - _MIN_LOG_MEL)), _F_SP * m)


def mel_filterbank(sr, n_fft, n_mels=128):
    """librosa.filters.mel defaults (Slaney scale and norm, fmin 0, fmax sr/2, float32)."""
    fft_f = np.linspace(0, sr / 2, 1 + n_fft // 2)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2), n_mels + 2))
    fdiff, ramps = np.diff(mel_f), mel_f[:, None] - fft_f[None, :]
    w = np.zeros((n_mels, len(fft_f)))
    for i in range(n_mels):
        w[i] = np.maximum(0, np.minimum(-ramps[i] / fdiff[i], ramps[i + 2] / fdiff[i + 1]))
    w *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, None]
    return w.astype(np.float32)


def mfcc_means(y, sr, n_mfcc=13, n_fft=2048, hop=512, n_mels=128):
    """Mean over frames of librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc) with its defaults."""
    yp = np.pad(y, n_fft // 2, mode="constant")
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)          # periodic Hann
    frames = np.lib.stride_tricks.sliding_window_view(yp, n_fft)[::hop]
    power = np.abs(np.fft.rfft(frames * win, axis=1)).T ** 2
    S = mel_filterbank(sr, n_fft, n_mels) @ power
    db = 10 * np.log10(np.maximum(1e-10, S))
    db = np.maximum(db, db.max() - 80.0)
    N = db.shape[0]
    k, n = np.arange(n_mfcc)[:, None], np.arange(N)[None, :]
    dct = np.cos(np.pi / N * (n + 0.5) * k) * np.sqrt(2.0 / N)
    dct[0] /= np.sqrt(2.0)                                                   # DCT-II, norm="ortho"
    return (dct @ db).mean(axis=1)


def welch(x, fs, nperseg):
    """scipy.signal.welch defaults: periodic Hann, 50% overlap, constant detrend, one-sided density."""
    step = nperseg - nperseg // 2
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nperseg) / nperseg)
    segs = np.lib.stride_tricks.sliding_window_view(x, nperseg)[::step]
    segs = segs - segs.mean(axis=1, keepdims=True)
    P = np.abs(np.fft.rfft(segs * win, axis=1)) ** 2 / (fs * (win ** 2).sum())
    P[:, 1:-1 if nperseg % 2 == 0 else None] *= 2
    return np.fft.rfftfreq(nperseg, 1 / fs), P.mean(axis=0)


# ---------- features (identical to the training code) ----------

def _st(f):
    return 12 * np.log2(np.asarray(f, float) / 100.0)


def cpp(y, sr, c):
    n, hop = int(c["frame_s"] * sr), int(c["hop_s"] * sr)
    if len(y) < n:
        return np.nan
    frames = np.lib.stride_tricks.sliding_window_view(y, n)[::hop] * np.hanning(n)
    rms = np.sqrt((frames ** 2).mean(1))
    frames = frames[rms > 0.05 * rms.max()]
    nfft = int(2 ** np.ceil(np.log2(n)))
    spec_db = 10 * np.log10(np.abs(np.fft.rfft(frames, nfft)) ** 2 + 1e-12)
    ceps_db = 10 * np.log10(np.fft.irfft(spec_db, nfft) ** 2 + 1e-12)
    q = np.arange(nfft) / sr
    peak_zone = (q >= 1 / c["f0_max_hz"]) & (q <= 1 / c["f0_min_hz"])
    fit_zone = (q >= c["fit_from_s"]) & (q <= nfft / 2 / sr)
    vals = []
    for row in ceps_db:
        i = np.flatnonzero(peak_zone)[np.argmax(row[peak_zone])]
        slope, icpt = np.polyfit(q[fit_zone], row[fit_zone], 1)
        vals.append(row[i] - (slope * q[i] + icpt))
    return float(np.mean(vals)) if vals else np.nan


def recording_features(y, sr, ex):
    """The 28 per-recording features of one recording already at ex['sr']."""
    import parselmouth
    from parselmouth.praat import call

    def num(v):
        v = float(v)
        return v if np.isfinite(v) else np.nan

    snd = parselmouth.Sound(y, sampling_frequency=sr)
    lo, hi = ex["pitch_floor_hz"], ex["pitch_ceiling_hz"]
    pitch = snd.to_pitch(pitch_floor=lo, pitch_ceiling=hi)
    pp = call(snd, "To PointProcess (periodic, cc)", lo, hi)
    harm = call(snd, "To Harmonicity (cc)", 0.01, lo, 0.1, 1.0)
    form = call(snd, "To Formant (burg)", 0.0, ex["max_formants"], ex["max_formant_hz"], 0.025, 50)
    f = {"shimmer_local": num(call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)),
         "hnr_db": num(call(harm, "Get mean", 0, 0)),
         "f0_mean_hz": num(call(pitch, "Get mean", 0, 0, "Hertz"))}
    for i in (1, 2, 3):
        f[f"f{i}_mean_hz"] = num(call(form, "Get mean", i, 0, 0, "hertz"))
    f.update({c: float(v) for c, v in zip(MFCC, mfcc_means(y, sr, ex["n_mfcc"]))})
    # breathiness and F0 contour
    out = {k: np.nan for k in EXTRA}
    f0, times = pitch.selected_array["frequency"], pitch.xs()
    voiced = f0 > 0
    out["voiced_fraction"] = float(voiced.mean()) if len(f0) else np.nan
    if voiced.sum() >= 3:
        st = _st(f0[voiced])
        out["f0_p5_st"], out["f0_p95_st"] = float(np.percentile(st, 5)), float(np.percentile(st, 95))
        out["f0_range_st"] = out["f0_p95_st"] - out["f0_p5_st"]
        both = voiced[1:] & voiced[:-1]
        steps = np.abs(np.diff(_st(np.where(voiced, f0, 1.0))))[both]
        out["f0_step_st"] = float(np.median(steps)) if len(steps) else np.nan
        n = int(0.04 * sr)
        h = []
        for i in np.flatnonzero(voiced)[np.linspace(0, voiced.sum() - 1, min(50, voiced.sum())).astype(int)]:
            c0 = int(times[i] * sr) - n // 2
            if c0 < 0 or c0 + n > len(y):
                continue
            spec = 20 * np.log10(np.abs(np.fft.rfft(y[c0:c0 + n] * np.hanning(n), 8 * n)) + 1e-12)
            fr = np.fft.rfftfreq(8 * n, 1 / sr)
            amp = lambda fq: spec[(fr > 0.9 * fq) & (fr < 1.1 * fq)].max()
            h.append(amp(f0[i]) - amp(2 * f0[i]))
        out["h1_h2_db"] = float(np.median(h)) if h else np.nan
    out["cpp_db"] = cpp(y, sr, ex["cpp"])
    fr, pw = welch(y, sr, min(2048, len(y)))
    band = (fr > 0) & (fr <= ex["spectrum_max_hz"])
    out["spectral_slope_db_khz"] = float(np.polyfit(fr[band] / 1000, 10 * np.log10(pw[band] + 1e-20), 1)[0])
    lo_e, hi_e = pw[(fr >= 50) & (fr < 1000)].sum(), pw[(fr >= 1000) & (fr < 5000)].sum()
    out["alpha_ratio_db"] = float(10 * np.log10(lo_e / hi_e)) if lo_e > 0 and hi_e > 0 else np.nan
    return {**f, **out}


def resample(x, fs, ex):
    import soxr
    return soxr.resample(x, fs, ex["sr"], quality=ex["resample_quality"]) if fs != ex["sr"] else x


def feature_vector(recordings, sex, model):
    """The model's feature vector. recordings: {code: (signal, sample rate)}; missing codes -> NaN."""
    ex = model["extraction"]
    per = {code: recording_features(resample(np.asarray(x, float), sr, ex), ex["sr"], ex)
           for code, (x, sr) in recordings.items()}
    vals = {f"{code}__{feat}": per[code][feat] if code in per else np.nan
            for code in RECORDINGS for feat in PER_RECORDING}
    vals["sex_female"], vals["sex_male"] = float(sex == "female"), float(sex == "male")
    fh, fl = vals["a_h__f0_mean_hz"], vals["a_l__f0_mean_hz"]
    vals["pitch_range_a_h_minus_a_l_st"] = float(_st(fh) - _st(fl)) if np.isfinite(fh) and np.isfinite(fl) else np.nan
    return np.array([vals[c] for c in model["features"]], float)


# ---------- model ----------

def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def probability(x, model):
    x = np.where(np.isnan(x), np.asarray(model["impute_median"]), x)
    z = (x - np.asarray(model["scale_mean"])) / np.asarray(model["scale_std"])
    s = 1 / (1 + np.exp(-(z @ np.asarray(model["coef"]) + model["intercept"])))
    return 1 / (1 + np.exp(-(model["platt"]["a"] * _logit(s) + model["platt"]["b"])))


def predict(recordings, sex, model, top=5):
    """Probability, largest contributions, key features vs. training medians, caveats."""
    if sex not in ("female", "male"):
        raise ValueError("sex must be 'female' or 'male'")
    needed = model["audio"]["recordings"]
    unknown = set(recordings) - set(needed)
    if unknown or not recordings:
        raise ValueError(f"this model uses {needed}; got {sorted(recordings)}")
    x = feature_vector(recordings, sex, model)
    p = float(probability(x, model))
    xi = np.where(np.isnan(x), np.asarray(model["impute_median"]), x)
    contrib = np.asarray(model["coef"]) * (xi - np.asarray(model["scale_mean"])) / np.asarray(model["scale_std"])
    order = np.argsort(-np.abs(contrib))[:top]
    fin = lambda v: None if not np.isfinite(v) else round(float(v), 3)
    return {"model": f"{model['model']} ({model['variant']})", "probability_rekurrensparese": round(p, 3),
            "recordings_used": sorted(recordings), "recordings_missing_imputed": [c for c in needed if c not in recordings],
            "validated_performance": model.get("validated_performance", {}),
            "top_contributions": [{"feature": model["features"][i], "contribution": round(float(contrib[i]), 3),
                                   "value": fin(x[i])} for i in order],
            "key_features": {f: {"this_speaker": fin(x[model["features"].index(f)]), **{k: round(v, 3) for k, v in r.items()}}
                             for f, r in model["reference_medians"].items()},
            "caveats": model["caveats"]}
