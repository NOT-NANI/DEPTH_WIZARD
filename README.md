---
title: DepthWizard
emoji: 🏔️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# DepthWizard — single-view height estimation and 3D flythrough

Research prototype for estimating relative depth from optical remote-sensing imagery, measuring alignment to reference AGL where available, and developing toward DSM and 3D inspection. The pretrained monocular model does not produce metric elevation on its own.

## Current architecture

- `src/depth_inference.py`: Depth Anything V2 Small inference and lossless float32 `.npy` output.
- `src/gamus.py`: HDF5 `image` loading, shape checks, and exact `-5.0` NoData masking.
- `src/calibration.py`: least-squares scale/shift fitting abstraction.
- `src/evaluation.py`: MAE, RMSE, Pearson correlation, and R².
- `src/visualization.py`: RGB, depth, AGL, aligned estimate, error map, and error histogram.
- `src/dsm.py`: numeric surface export, with optional GeoTIFF only when a georeferenced source is supplied and rasterio is available.
- `src/stage14_pipeline.py`: frozen Stage 12 model inference, calibrated height/DSM products, optional GeoTIFF/DEM handling, slope preview, textured mesh export.
- `scripts/run_demo.py`: local upload UI and interactive terrain viewer server.
- `scripts/`: inference and GAMUS evaluation commands.

The current project directory has no Git repository. Local sample files are used; the full GAMUS dataset is not downloaded.

## Environment setup

Use Python 3.12 and a virtual environment. Install dependencies with `python -m pip install -r requirements.txt`. This project was inspected with the existing `venv` containing PyTorch 2.14.1, torchvision 0.29.1, Transformers, Pillow, h5py, NumPy, and Matplotlib. In this session `torch.backends.mps.is_available()` returned false, so inference selects CPU. A compatible Apple Silicon PyTorch build should select MPS automatically.

## Dataset preparation

Place the sample files in the project root: `gamus_sample.h5` (RGB), `gamus_height.h5` (AGL), and optionally `gamus_rgb.png` for inference. Each HDF5 file stores its array at `image`. Height values equal to exactly `-5.0` are treated as NoData; other negative AGL values are retained.

## Run baseline inference

```bash
./venv/bin/python scripts/run_depth.py gamus_rgb.png
```

The default input is `gamus_rgb.png`. Outputs are `outputs/gamus_rgb_depth_raw.npy` (float32 model values, not metres) and `outputs/gamus_rgb_depth.png` (visualization only, percentile stretched). Input paths are resolved from the current working directory; the default output directory is project-relative.

## Run GAMUS evaluation

```bash
./venv/bin/python scripts/evaluate_gamus.py gamus_rgb.png gamus_height.h5
```

This estimates `GT ≈ a × prediction + b` using all valid pixels in the supplied sample and saves metrics JSON/CSV and diagnostics in `outputs/`, plus `baseline_report.md`. These are **scale/shift aligned baseline** metrics; fitting and scoring on the same sample makes them optimistic. They do not demonstrate independently calibrated metric height or generalization. For meaningful research validation, hold out scenes from calibration and evaluation.

The evaluation also saves `outputs/gamus_aligned_height.npy`, solely to inspect the sample-fitted surface. Its pixel values are not deployment-calibrated heights.

## Multi-scene GAMUS baseline

Fetch at most ten filename-matched pairs from the test split, then evaluate scene by scene:

```bash
./venv/bin/python scripts/download_gamus_test_subset.py --count 10
./venv/bin/python scripts/evaluate_multiscene.py
```

This stores only the selected RGB/AGL files under `data/multiscene_test/` and writes per-scene and aggregate outputs to `outputs/multiscene/`. The reported aggregate is an unweighted summary of per-scene metrics; affine scale/shift is fit separately on each complete scene and must not be read as deployment calibration.

## Calibration and metric-height distinction

`src/calibration.py` currently provides a scale/offset abstraction demonstrated against GAMUS ground truth. Later reference adapters can supply GCPs, known elevations, SRTM/NASADEM, or georeferenced raster metadata. GAMUS sample alignment is not equivalent to operational calibration.

## DSM generation and georeferencing

Create a pixel-grid surface from the raw prediction (relative units):

```bash
./venv/bin/python scripts/create_dsm.py
```

For a supplied scale/shift, pass `--scale` and `--shift`. A GeoTIFF output requires `--source-raster` with matching dimensions and installed rasterio. For ordinary PNG/JPG, CRS and affine transform are unavailable, so the output is an unreferenced pixel grid. The current sample has no supplied georeferencing; no GeoTIFF is created.

## 3D visualization and flythrough

After evaluation, generate a viewer from the explicitly sample-fitted surface:

```bash
./venv/bin/python scripts/create_viewer.py
```

This creates a 256×256 preview mesh with RGB texture, height exaggeration, orbit/zoom, WASD camera movement, and click-to-inspect. Three.js and OrbitControls load from a CDN, so opening the HTML requires browser network access. This is a prototype preview, not a georeferenced or deployment-calibrated terrain product.

## Stage 12–13 model validation

Stage 12 adapted only the Depth Anything V2 Small neck/head on six scenes, with two validation scenes and two sealed test scenes. Stage 13 evaluated the frozen checkpoint on ten additional scenes; none overlap Stage 12. The new-scene validation-calibrated mean MAE/RMSE were 5.2171/6.3370 m for Experiment B and 5.5230/6.4579 m for baseline. Results cover those ten GAMUS scenes only. See `outputs/stage12/` and `outputs/stage13/`.

## Final local demo

```bash
./venv/bin/python scripts/run_demo.py --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765`. PNG/JPG input is unreferenced. GeoTIFF and DEM operations require optional Rasterio (`./venv/bin/python -m pip install rasterio`); existing CRS/transform are copied when available. The 3D viewer uses Three.js from a CDN, so its interactive panel needs browser internet. Demo and calibration limitations are documented in `outputs/final/`.

## Metrics and limitations

MAE/RMSE use metres only after the sample-fitted affine mapping; Pearson r measures linear association, and R² is computed against the evaluated ground truth. One image cannot substantiate land-cover-specific failure claims or field-ready performance. The model is scale ambiguous, optical single-view input has occlusions and appearance confounds, and sample-fitted metrics are not deployment metrics.

## Tested

The project test suite is `./venv/bin/python -m unittest discover -s tests -v`; Stage 1–14 pipeline checks are included. `./venv/bin/python -m compileall -q src scripts tests` checks Python syntax. MPS was unavailable in the recorded environment, so training/inference used CPU.
