"""
Builds the demo's example speakers from a folder of WAV files and a manifest.csv.

    python scripts/build_examples.py ~/Desktop/svd_demo_examples

Listed examples go to ui/examples/ with their scores in examples.json. Three recordings go to
drag_and_drop/ instead, for live use. Session IDs and ages in the manifest are never copied.
"""
import csv
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api import predict  # noqa: E402

HELD_BACK_COUNT = 3


def held_back(rows):
    """The second speaker of the first few sex and group pairs."""
    seen, repeats = set(), []
    for row in rows:
        key = (row["sex"], row["group"])
        if key in seen:
            repeats.append(row)
        seen.add(key)
    return repeats[:HELD_BACK_COUNT]


def public_name(name):
    """The file name without the speaker's age, because file names appear on screen."""
    return re.sub(r"_age\d+", "", name)


def main(source):
    with open(source / "manifest.csv", newline="") as file:
        rows = list(csv.DictReader(file))
    missing = [row["file"] for row in rows if not (source / row["file"]).is_file()]
    if missing:
        sys.exit(f"Nothing was changed. These files are in manifest.csv but not in the folder: {', '.join(missing)}")
    drag_rows = held_back(rows)
    model = predict.get_model()
    examples_dir = ROOT / "ui" / "examples"
    drag_dir = ROOT / "drag_and_drop"
    examples_dir.mkdir(parents=True, exist_ok=True)
    drag_dir.mkdir(exist_ok=True)

    examples = []
    for row in rows:
        if row in drag_rows:
            shutil.copyfile(source / row["file"], drag_dir / public_name(row["file"]))
            continue
        shutil.copyfile(source / row["file"], examples_dir / public_name(row["file"]))
        signal, rate = predict.read_audio((source / row["file"]).read_bytes())
        result = predict.rp_model.predict({"a_n": (signal, rate)}, row["sex"], model)
        examples.append({
            "file": public_name(row["file"]),
            "sex": row["sex"],
            "group": row["group"],
            "validatedProbability": float(row["oof_probability_paralysis"]),
            "result": predict.to_response(result),
        })
    payload = {"model": predict.model_file().name, "examples": examples}
    (examples_dir / "examples.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{len(examples)} examples in {examples_dir}, {len(drag_rows)} recordings in {drag_dir}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser())
