# Vocal fold paralysis screen (portable, research demo)

Probability that a voice belongs to a speaker with **vocal fold paralysis (Rekurrensparese)**
rather than a healthy speaker, from **one sustained /a/ ("aah") at a comfortable pitch**.
**Research prototype, not a diagnostic test.**

Self-contained copy of the a_n audio logistic regression from
`voice-ai-enabled-diagnosis-for-neuro-psych` (branch `svd-egg-sensor`; exported by
`experiments/svd_audio_egg_disagreement/08_audio_model/rekurrensparese_lr_export.py`). Copy this folder as is into
another repository; it needs nothing else from the training repository.

## Model card

| | |
|---|---|
| Task | healthy vs. Rekurrensparese (patients whose only diagnosis is vocal fold paralysis) |
| Data | Saarbrücken Voice Database (SVD), 678 healthy speakers and 137 patients |
| Input | one sustained /a/ at a comfortable pitch (`a_n`, the SVD "normal pitch" vowel), about 3 s; sex |
| Features | 30: Praat shimmer, HNR, mean F0, F1-F3, MFCC 1-13 means, cepstral peak prominence, H1-H2, spectral slope, alpha ratio, F0-contour measures (5th/95th percentile, range, frame-to-frame step, voiced fraction); sex |
| Model | L2 logistic regression, Platt-calibrated; `model/rekurrensparese_an_lr_weighted.json` (age/sex-weighted training, recommended) and `..._main.json` |
| Validated performance | out-of-fold, 5-fold speaker-disjoint CV (stored in each model file under `validated_performance`): on the 69 age- and sex-matched pairs, AUC 0.781 (0.695-0.854) for the weighted model and 0.767 (0.682-0.843) for the main model; 0.846 (0.799-0.886) on all speakers (main). Age alone reaches 0.924 on all speakers, so the matched-pair figure is the meaningful one. The four-recording model of the training repo does slightly better on the pairs (0.818), with overlapping CIs |
| Training of this file | refit on all 815 speakers (C by 5-fold CV, Platt on out-of-fold scores) |

**Caveats** (also returned with every prediction):
- Not a diagnostic test.
- Healthy SVD speakers are mostly in their 20s and patients mostly over 50; the model partly reflects age.
- SVD was recorded in a clinic; other microphones and rooms shift the features (MFCC 1 depends on the
  recording level), so scores on new recordings are illustrative.
- This model has seen every SVD speaker; do not evaluate it on SVD recordings.

Data credit: Saarbrücken Voice Database (Pützer & Barry), Zenodo record 16874898, CC BY 4.0.

## Use from Python

```bash
pip install -r requirements.txt
```
```python
import rp_model
model = rp_model.load_model("model/rekurrensparese_an_lr_weighted.json")
print(rp_model.predict({"a_n": rp_model.read_wav("aah.wav")}, "female", model))
```
Input: PCM WAV, any sample rate, mono or stereo.

## Deploy on Vercel

The repository root must contain `api/`, `public/`, `model/`, `rp_model.py`, `requirements.txt`
and `vercel.json` (this folder's layout). Then `vercel deploy`, or import the repository in the
Vercel dashboard with framework preset "Other".

- `public/index.html`: records the "aah" in the browser (or takes an uploaded file), converts it to
  16-bit WAV in the browser, posts it to the function, and shows the validated AUC from the model file.
- `api/predict.py`: `POST /api/predict` with `{"sex": "female", "recordings": {"a_n": "<base64 WAV>"}}`;
  `GET /api/predict` returns the recordings the model needs and its validated performance.
  Set the environment variable `RP_MODEL=rekurrensparese_an_lr_main.json` to use the main variant.
- Dependencies are numpy, praat-parselmouth and soxr only (no librosa or scipy), to stay within the
  Vercel function size limit. One request takes a few seconds; `vercel.json` allows 60 s.

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests
```
`tests/golden.json` holds features and probabilities computed with the **training code** (librosa,
scipy and the original feature functions) for the synthetic recordings in `tests/fixtures/`. The test
recomputes them with `rp_model.py`; if it fails, the portable features no longer match the model.
