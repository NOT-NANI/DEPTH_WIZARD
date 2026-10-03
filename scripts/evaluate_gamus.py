#!/usr/bin/env python3
"""Infer on RGB and evaluate against local GAMUS AGL with sample alignment."""
import argparse, csv, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.depth_inference import run_inference
from src.evaluation import evaluate
from src.gamus import load_gamus, print_statistics
from src.visualization import save_diagnostics
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("rgb",nargs="?",type=Path,default=ROOT/"gamus_rgb.png")
parser.add_argument("height",nargs="?",type=Path,default=ROOT/"gamus_height.h5")
parser.add_argument("--rgb-h5",type=Path,default=ROOT/"gamus_sample.h5")
parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs")
args=parser.parse_args()
def absolute(p): return p if p.is_absolute() else (Path.cwd()/p).resolve()
try:
    rgb,gt,valid=load_gamus(absolute(args.rgb_h5),absolute(args.height)); print_statistics(rgb,gt,valid)
    out=absolute(args.output_dir); raw_path,_,_=run_inference(absolute(args.rgb),out)
    import numpy as np
    raw=np.load(raw_path,allow_pickle=False)
    if raw.shape!=gt.shape: raise ValueError(f"Prediction {raw.shape} and GAMUS height {gt.shape} dimensions differ")
    calibration,aligned,metrics=evaluate(raw,gt,valid)
    jp=out/"gamus_baseline_metrics.json"; cp=out/"gamus_baseline_metrics.csv"
    np.save(out/"gamus_aligned_height.npy",aligned,allow_pickle=False)
    jp.write_text(json.dumps(metrics,indent=2)+"\n")
    with cp.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=metrics.keys()); writer.writeheader(); writer.writerow(metrics)
    fig,err=save_diagnostics(rgb,raw,gt,aligned,valid,out)
    print("\nScale/shift aligned baseline (not independently calibrated metric elevation):")
    for key,value in metrics.items(): print(f"{key}: {value}")
    print(f"Metrics JSON: {jp.resolve()}\nMetrics CSV: {cp.resolve()}\nDiagnostics: {fig.resolve()}\nError map: {err.resolve()}")
    report=f"""# DepthWizard baseline report

## What was tested

Depth Anything V2 Small on the local 1024×1024 GAMUS optical sample. The raw float32 output was resized to source dimensions and fit by least-squares `GT = a × prediction + b` over finite pixels excluding exactly the documented -5.0 NoData marker. This is a sample-aligned baseline, not deployment calibration or an independently metric DSM.

## Dataset and preprocessing

RGB and AGL were loaded from the HDF5 `image` datasets. The RGB PNG was passed as RGB; output PNG uses percentile stretch for display only. Evaluation used `{raw_path.name}`.

## Metrics

- Valid paired pixels: {metrics['valid_pixels']:,}
- Scale: {metrics['scale_coefficient']:.8g}
- Shift: {metrics['shift_coefficient_m']:.8g} m
- MAE: {metrics['mae_m']:.6g} m
- RMSE: {metrics['rmse_m']:.6g} m
- Pearson r: {metrics['pearson_r']:.6g}
- R²: {metrics['r_squared']:.6g}

## Observed limitations

Aggregate metrics alone do not establish which land-cover types or structures cause errors. Inspect `{fig.name}` and `{err.name}` before making claims about buildings, roads, vegetation, shadows, edges, or height ranges. The fit uses this sample's ground truth and is optimistic relative to inference without local references. Monocular predictions are scale ambiguous.

## Improvement directions

Inspect spatial error patterns and stratify against available labels or carefully defined regions. Then test a targeted GAMUS fine-tuning or edge-preservation experiment on held-out scenes; one scene cannot establish generalization.
"""
    (ROOT/"baseline_report.md").write_text(report)
except Exception as exc: parser.exit(1,f"Error: {exc}\n")
