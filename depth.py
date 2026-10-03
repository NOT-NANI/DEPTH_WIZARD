"""Backward-compatible entry point for scripts/run_depth.py."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.depth_inference import run_inference

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Depth Anything V2 inference")
    parser.add_argument("image", nargs="?", type=Path, default=ROOT / "gamus_rgb.png")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    image = args.image if args.image.is_absolute() else (Path.cwd() / args.image).resolve()
    output = args.output_dir if args.output_dir.is_absolute() else (ROOT / args.output_dir).resolve()
    run_inference(image, output)
