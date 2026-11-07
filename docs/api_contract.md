# API contract: `POST /api/predict`

Implemented by `api/predict.py`, which calls `portable_rekurrensparese/rp_model.py`.
This is what the UI in `ui/` sends and expects. The UI checks every response against this
shape and shows its error state if the response doesn't match.
`ui/mock/predict_response.json` is a valid example of the success response.

## Request

Same-origin `multipart/form-data` with two fields:

| Field   | Type | Notes |
|---------|------|-------|
| `audio` | file | One sustained /a/ at normal pitch, as a WAV file. |
| `sex`   | text | Exactly `female` or `male`. The API maps these to its model's own coding. |

About the audio:

- **In-browser recordings** arrive as mono 16-bit PCM WAV, about 2.5 s long, with
  browser echo cancellation, noise suppression and auto-gain turned off.
- **Sample rate is not fixed.** Browsers usually record at 44.1 or 48 kHz, and
  uploads can be any rate. The API must handle this itself.
- **Uploads may be stereo** and up to 4 MB. The UI rejects larger files because Vercel
  limits function request bodies to about 4.5 MB.

```bash
curl -X POST http://localhost:3000/api/predict \
  -F "audio=@voice.wav" \
  -F "sex=female"
```

## Success response: `200`

```json
{
  "probability": 0.42,
  "top_features": [
    { "name": "a_n__cpp_db", "label": "CPP (voice clarity)", "direction": "down", "size": 0.64 },
    { "name": "a_n__shimmer_local", "label": "Shimmer (loudness stability)", "direction": "up", "size": 0.52 }
  ]
}
```

| Field                     | Type   | Notes |
|---------------------------|--------|-------|
| `probability`             | number | Model output from 0 to 1. The UI shows it as a percentage. |
| `top_features`            | array  | At least 1 item, normally 5. The UI sorts by `size` and shows the top 5. |
| `top_features[].name`     | string | Stable machine name of the feature. |
| `top_features[].label`    | string | Plain-English name shown on screen. |
| `top_features[].direction`| string | `up` if it raised this score, `down` if it lowered it. |
| `top_features[].size`     | number | Strength of the effect, 0 or greater. Never signed; `direction` holds the sign. |

**`size`** is the absolute log-odds contribution, `|coefficient × standardized feature
value|`, before Platt calibration, with `direction` taken from its sign. The UI scales bars relative to
the largest `size` in each result, so only the proportions within a result matter.

## Error response: any `4xx` or `5xx`

```json
{ "error": "That recording is too short. Please hold the sound for about 2 seconds." }
```

`error` is shown to the user as written, so keep it in plain language and leave out
stack traces. If it's missing, the UI shows a generic message.

| Status | When |
|--------|------|
| `400`  | A field is missing, or `sex` is invalid. |
| `413`  | The file is too large. |
| `422`  | The audio can't be read or scored (not PCM WAV, under 0.3 s, over 30 s, silent, or almost no voicing). |
| `500`  | Unexpected server failure. |
| `503`  | The model file isn't installed. No score is returned. |

## Timing

The UI shows a "still working" note after 8 seconds and gives up after 30 seconds. Both
are set at the top of `ui/app.js`. The API must answer within that window, including
serverless cold starts.

## What stays in the UI

- **Screening cutoff:** `SCREEN_CUTOFF` in `ui/app.js` decides whether the result says a
  paralysis pattern was detected. It's `0.5`, the threshold used in the study's analysis.

## Example speakers (not part of the API)

The Example speakers tab reads `ui/examples/examples.json`, a static file that plays its audio
from the WAV files beside it. `scripts/build_examples.py` generates both.

| Field | Values |
|---|---|
| `file` | WAV file name in `ui/examples/` |
| `sex` | `female` or `male` |
| `group` | `healthy` or `paralysis` |
| `validatedProbability` | number from 0 to 1, from a model that never saw the speaker |
| `result` | the success response above, saved from the deployed model |
