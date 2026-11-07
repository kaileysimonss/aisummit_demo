# aisummit_demo

Research demo of a vocal fold paralysis voice screen. Not for diagnosis.

## Run locally

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python dev_server.py
```

Then open http://localhost:8000.

## The model

`portable_rekurrensparese/model/` holds the exported a_n logistic regression. The age/sex-weighted
variant is the default; set `RP_MODEL=rekurrensparese_an_lr_main.json` to use the main variant. To
update it, re-run the export in the training repository (branch `svd-egg-sensor`) and copy that
package's `model/` folder, `tests/golden.json` and `tests/fixtures/` here.

## Test

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests
(cd portable_rekurrensparese && ../.venv/bin/python -m pytest tests)
```

The package's golden test compares against values computed on x86, which is also what Vercel runs. It
passes there. On Apple Silicon it fails its strict tolerance because HNR and F3 differ slightly
(scores change by at most 0.00003). To run it on an Apple Silicon Mac, use an x86 Python under Rosetta.

## Layout

- `ui/`: the web page (no build step)
- `api/predict.py`: the API described in `docs/api_contract.md`
- `portable_rekurrensparese/`: the model package, unmodified
- `dev_server.py`: runs the page and API together locally
- `vercel.json`: Vercel settings for the same layout

## Example speakers

The Example speakers tab lists five recordings from the Saarbrücken Voice Database, in
`ui/examples/`. Clicking one plays its audio and shows its saved results without scoring it again.
Three more recordings are in `drag_and_drop/`, to drag into **Try a recording**. That folder isn't
part of the deployed site, but everything in `ui/examples/` is publicly downloadable once deployed.

Each example shows its live score, from the deployed model, which was trained on these speakers.
`examples.json` also keeps the validated score, from a model that never saw the speaker, but the
page doesn't show it.

To rebuild them, run `python scripts/build_examples.py <folder>`. The source `manifest.csv` holds
session IDs, so it stays out of the repo, and the script doesn't copy them.
