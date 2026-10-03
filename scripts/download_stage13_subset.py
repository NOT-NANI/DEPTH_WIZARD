#!/usr/bin/env python3
"""Download only the predeclared new GAMUS Stage 13 RGB/AGL pairs."""
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/stage13_gamus"
SCENES = ["DC_10_29", "DC_11_15", "DC_11_19", "DC_11_31", "DC_11_32", "DC_12_15", "DC_12_20", "DC_12_34", "DC_12_35", "DC_13_19"]

for scene in SCENES:
    for folder, suffix in (("images", "RGB"), ("heights", "AGL")):
        filename = f"{folder}/test/{scene}_{suffix}.h5"
        local = hf_hub_download(
            repo_id="earthflow/GAMUS", repo_type="dataset", filename=filename,
            local_dir=str(OUT), local_dir_use_symlinks=False,
        )
        print(f"{scene}: {Path(local).stat().st_size:,} bytes", flush=True)
