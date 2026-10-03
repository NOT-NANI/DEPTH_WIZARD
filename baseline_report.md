# DepthWizard baseline report

## What was tested

Depth Anything V2 Small on the local 1024×1024 GAMUS optical sample. The raw float32 output was resized to source dimensions and fit by least-squares `GT = a × prediction + b` over finite pixels excluding exactly the documented -5.0 NoData marker. This is a sample-aligned baseline, not deployment calibration or an independently metric DSM.

## Dataset and preprocessing

RGB and AGL were loaded from the HDF5 `image` datasets. The RGB PNG was passed as RGB; output PNG uses percentile stretch for display only. Evaluation used `gamus_rgb_depth_raw.npy`.

## Metrics

- Valid paired pixels: 1,043,613
- Scale: -4.0139886
- Shift: 15.722362 m
- MAE: 9.31404 m
- RMSE: 10.8249 m
- Pearson r: 0.0785399
- R²: 0.00616854

## Observed limitations

Aggregate metrics alone do not establish which land-cover types or structures cause errors. Inspect `gamus_baseline_diagnostics.png` and `gamus_error_map.png` before making claims about buildings, roads, vegetation, shadows, edges, or height ranges. The fit uses this sample's ground truth and is optimistic relative to inference without local references. Monocular predictions are scale ambiguous.

## Improvement directions

Inspect spatial error patterns and stratify against available labels or carefully defined regions. Then test a targeted GAMUS fine-tuning or edge-preservation experiment on held-out scenes; one scene cannot establish generalization.
