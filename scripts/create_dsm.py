#!/usr/bin/env python3
"""Create a float32 surface; optional calibration and georeferenced GeoTIFF export."""
import argparse, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.dsm import save_surface
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("prediction",nargs="?",type=Path,default=ROOT/"outputs/gamus_rgb_depth_raw.npy")
parser.add_argument("--output",type=Path,default=ROOT/"outputs/gamus_surface.npy")
parser.add_argument("--scale",type=float,default=1.0); parser.add_argument("--shift",type=float,default=0.0)
parser.add_argument("--source-raster",type=Path,help="Georeferenced raster to copy CRS/transform from for GeoTIFF output")
args=parser.parse_args()
def abs_path(p): return p if p.is_absolute() else (Path.cwd()/p).resolve()
try:
    src=abs_path(args.prediction); output=abs_path(args.output)
    surface=np.load(src,allow_pickle=False).astype(np.float32)*args.scale+args.shift
    dest=save_surface(surface,output,abs_path(args.source_raster) if args.source_raster else None)
    print(f"Surface: {dest}\nShape: {surface.shape}; scale={args.scale}; shift={args.shift}")
    if args.source_raster is None: print("Georeferencing unavailable; saved as an unreferenced pixel grid.")
except Exception as exc: parser.exit(1,f"Error: {exc}\n")
