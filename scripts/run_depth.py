#!/usr/bin/env python3
"""Run Depth Anything V2 and save lossless float32 depth plus a visual PNG."""
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.depth_inference import run_inference
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("image",nargs="?",type=Path,default=ROOT/"gamus_rgb.png")
parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs")
args=parser.parse_args()
try:
    image=args.image if args.image.is_absolute() else (Path.cwd()/args.image).resolve()
    out=args.output_dir if args.output_dir.is_absolute() else (ROOT/args.output_dir).resolve()
    run_inference(image,out)
except Exception as exc: parser.exit(1,f"Error: {exc}\n")
