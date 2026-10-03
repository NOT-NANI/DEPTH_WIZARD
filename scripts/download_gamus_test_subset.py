#!/usr/bin/env python3
"""Download a small, filename-matched subset of GAMUS test RGB/AGL HDF5 pairs."""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
SCENE_IDS = ["DC_03_26", "DC_05_28", "DC_05_30", "DC_07_21", "DC_07_29", "DC_08_27", "DC_09_18", "DC_09_29", "DC_09_32", "DC_10_20"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--count", type=int, default=10, choices=range(1, 11), help="number of scene pairs to download (1–10)")
parser.add_argument("--output-dir", type=Path, default=ROOT / "data/multiscene_test")
args = parser.parse_args()
output = args.output_dir if args.output_dir.is_absolute() else (ROOT / args.output_dir).resolve()
for scene_id in SCENE_IDS[:args.count]:
    for folder, suffix in [("images", "RGB"), ("heights", "AGL")]:
        name = f"{scene_id}_{suffix}.h5"
        path = hf_hub_download(repo_id="earthflow/GAMUS", repo_type="dataset", filename=f"{folder}/test/{name}", local_dir=str(output))
        print(f"{scene_id}: {folder}/test/{name} ({Path(path).stat().st_size:,} bytes)")
