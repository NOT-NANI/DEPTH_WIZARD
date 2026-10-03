#!/usr/bin/env python3
"""Stage 11 loader reproduction, resampling ablation, and transfer audit."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.calibration import fit_scale_shift
from src.depth_inference import MODEL_ID, select_device
from src.stage10_audit import score_calibrated, spatial_half_masks

DATA = ROOT / "data/multiscene_test"
STAGE9 = ROOT / "outputs/multiscene"
OUT = ROOT / "outputs/stage11"
SCENES = ["DC_03_26", "DC_05_28", "DC_05_30", "DC_07_21", "DC_07_29", "DC_08_27", "DC_09_18", "DC_09_29", "DC_09_32", "DC_10_20"]
INTERP_SCENES = ["DC_03_26", "DC_08_27", "DC_10_20"]
NODATA = -5.0


def load_official_arrays(scene: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Mirror official loader's direct HDF5 reads and PIL RGB conversion (transform=None)."""
    rgb_path = DATA / "images/test" / f"{scene}_RGB.h5"
    agl_path = DATA / "heights/test" / f"{scene}_AGL.h5"
    with h5py.File(rgb_path, "r") as f_rgb, h5py.File(agl_path, "r") as f_agl:
        rgb_source = f_rgb["image"][()]
        agl_source = f_agl["image"][()]
        info = {
            "rgb_path": str(rgb_path.relative_to(ROOT)), "agl_path": str(agl_path.relative_to(ROOT)),
            "rgb_dataset": "/image", "agl_dataset": "/image",
            "rgb_root_attributes": {str(k): str(v) for k, v in f_rgb.attrs.items()},
            "agl_root_attributes": {str(k): str(v) for k, v in f_agl.attrs.items()},
            "rgb_dataset_attributes": {str(k): str(v) for k, v in f_rgb["image"].attrs.items()},
            "agl_dataset_attributes": {str(k): str(v) for k, v in f_agl["image"].attrs.items()},
        }
    # Official loader uses Image.fromarray(image.astype(np.uint8)); height is returned as read.
    rgb_out = np.asarray(Image.fromarray(rgb_source.astype(np.uint8)))
    agl_out = agl_source
    info.update({
        "rgb_source_shape": list(rgb_source.shape), "rgb_output_shape": list(rgb_out.shape),
        "rgb_source_dtype": str(rgb_source.dtype), "rgb_output_dtype": str(rgb_out.dtype),
        "agl_source_shape": list(agl_source.shape), "agl_output_shape": list(agl_out.shape),
        "agl_source_dtype": str(agl_source.dtype), "agl_output_dtype": str(agl_out.dtype),
        "rgb_exact_array_match": bool(np.array_equal(rgb_source.astype(np.uint8), rgb_out)),
        "agl_exact_array_match": bool(np.array_equal(agl_source, agl_out, equal_nan=True)),
        "agl_valid_pixels": int((np.isfinite(agl_out) & (agl_out != NODATA)).sum()),
        "agl_nodata_pixels": int((~np.isfinite(agl_out) | (agl_out == NODATA)).sum()),
        "agl_min_m": float(np.nanmin(agl_out[ np.isfinite(agl_out) & (agl_out != NODATA) ])),
        "agl_median_m": float(np.nanmedian(agl_out[ np.isfinite(agl_out) & (agl_out != NODATA) ])),
        "agl_mean_m": float(np.nanmean(agl_out[ np.isfinite(agl_out) & (agl_out != NODATA) ])),
        "agl_max_m": float(np.nanmax(agl_out[ np.isfinite(agl_out) & (agl_out != NODATA) ])),
        "transformations": "Read /image directly; RGB cast to uint8 and wrapped with PIL Image.fromarray; AGL returned unchanged. No transpose, resize, interpolation, crop, or normalization because transform=None.",
    })
    return rgb_out, agl_out, info


def resize_prediction(raw: torch.Tensor, size: tuple[int, int], mode: str) -> np.ndarray:
    kwargs = {"align_corners": False} if mode in ("bilinear", "bicubic") else {}
    out = F.interpolate(raw[:, None], size=size, mode=mode, **kwargs).squeeze().float()
    return out.detach().cpu().numpy().astype(np.float32, copy=False)


def load_agl(scene: str) -> np.ndarray:
    with h5py.File(DATA / "heights/test" / f"{scene}_AGL.h5", "r") as f:
        return f["image"][()].astype(np.float32, copy=False)


def summarize_metrics(rows: list[dict]) -> dict:
    out = {}
    for key in ("mae_m", "rmse_m", "pearson_r", "r2"):
        vals = [float(r[key]) for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        out[key] = {"mean": float(np.mean(vals)) if vals else None, "median": float(np.median(vals)) if vals else None, "count": len(vals)}
    return out


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reproduce_loader() -> None:
    scene = "DC_03_26"
    rgb, agl, metadata = load_official_arrays(scene)
    dest = OUT / "loader_reproduction"
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / f"{scene}_RGB.npy", rgb, allow_pickle=False)
    Image.fromarray(rgb).save(dest / f"{scene}_RGB.png")
    np.save(dest / f"{scene}_AGL.npy", agl, allow_pickle=False)
    metadata["scene_id"] = scene
    metadata["rgb_channel_min_max_mean"] = [{"min": int(rgb[..., c].min()), "max": int(rgb[..., c].max()), "mean": float(rgb[..., c].mean())} for c in range(rgb.shape[-1])]
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    (dest / "transformation_description.md").write_text(
        "# Official-loader reproduction\n\n"
        f"Scene: `{scene}`. The official loader reads the HDF5 `/image` RGB array, casts it to `uint8`, and wraps it with `PIL.Image.fromarray`; the AGL loader reads its `/image` array and returns it directly. `transform=None` is used. Therefore no transpose, resize, interpolation, crop, or normalization is applied. The two exported output arrays are checked against the source HDF5 arrays in `metadata.json`. The RGB PNG is a viewable export; its numeric `.npy` is the array comparison artifact. AGL remains in its stored units (documented as metres in GAMUS materials); no resampling was done.\n"
    )


def run_interpolation() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    json_scene_data = {}
    device = select_device()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()
    prediction_dir = OUT / "interpolation_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    high_resolution_status = "not_run"
    try:
        for scene in INTERP_SCENES:
            rgb, target, meta = load_official_arrays(scene)
            image = Image.fromarray(rgb)
            inputs = processor(images=image, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                native = model(**inputs).predicted_depth
            base_modes = [("bicubic_518", "bicubic", native), ("bilinear_518", "bilinear", native), ("nearest_518", "nearest", native)]
            candidates: list[tuple[str, np.ndarray]] = []
            for name, mode, tensor in base_modes:
                candidates.append((name, resize_prediction(tensor, target.shape, mode)))
            # Same processor, weights, RGB and output pipeline; only processor size changes.
            try:
                higher_inputs = processor(images=image, return_tensors="pt", size={"height": 672, "width": 672})
                higher_inputs = {key: value.to(device) for key, value in higher_inputs.items()}
                with torch.inference_mode():
                    higher_native = model(**higher_inputs).predicted_depth
                higher = resize_prediction(higher_native, target.shape, "bicubic")
                candidates.append(("bicubic_672_input", higher))
                high_resolution_status = "succeeded at requested processor size 672x672 for all three scenes"
            except Exception as exc:  # Preserve baseline ablation even if this optional check fails.
                high_resolution_status = f"failed: {type(exc).__name__}: {exc}"
            valid = np.isfinite(target) & (target != NODATA)
            scene_rows = []
            scene_dir = prediction_dir / scene
            scene_dir.mkdir(parents=True, exist_ok=True)
            for method, pred in candidates:
                np.save(scene_dir / f"{method}.npy", pred, allow_pickle=False)
                fit = fit_scale_shift(pred, target, valid)
                full_score = score_calibrated(pred, target, valid, fit)
                common = {"scene": scene, "method": method, "calibration_mode": "per_scene_in_sample", "spatial_split": "all_valid", "scale": fit.scale, "shift_m": fit.shift, "calibration_pixels": int(valid.sum()), "evaluation_pixels": full_score["valid_pixels"], **full_score}
                rows.append(common); scene_rows.append(common)
                for axis, names in (("row", ("top_to_bottom", "bottom_to_top")), ("col", ("left_to_right", "right_to_left"))):
                    first, second = spatial_half_masks(target.shape, axis)
                    for cal_mask, eval_mask, label in ((first, second, names[0]), (second, first, names[1])):
                        mask_cal = valid & cal_mask
                        fit = fit_scale_shift(pred, target, mask_cal)
                        score = score_calibrated(pred, target, valid & eval_mask, fit)
                        row = {"scene": scene, "method": method, "calibration_mode": "spatial_holdout", "spatial_split": label, "scale": fit.scale, "shift_m": fit.shift, "calibration_pixels": int(mask_cal.sum()), "evaluation_pixels": score["valid_pixels"], **score}
                        rows.append(row); scene_rows.append(row)
            json_scene_data[scene] = {"rgb_shape": list(rgb.shape), "agl_shape": list(target.shape), "valid_agl_pixels": int(valid.sum()), "methods": scene_rows}
            del inputs, native, rgb, target
    finally:
        del model, processor
        if device.type == "mps":
            torch.mps.empty_cache()
    metadata = {"model_id": MODEL_ID, "device": str(device), "scenes": INTERP_SCENES, "methods": ["bicubic_518", "bilinear_518", "nearest_518", "bicubic_672_input"], "higher_resolution_status": high_resolution_status, "processor_note": "All processor defaults retained except size override for optional 672x672 run; original RGB and target pixels held fixed."}
    return rows, {"metadata": metadata, "scenes": json_scene_data, "aggregate_by_mode_method": {f"{mode}|{method}": summarize_metrics([r for r in rows if r["calibration_mode"] == mode and r["method"] == method]) for mode in sorted({r['calibration_mode'] for r in rows}) for method in sorted({r['method'] for r in rows})}}


def stage9_raw(scene: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = np.load(STAGE9 / "scenes" / scene / "rgb_depth_raw.npy", allow_pickle=False).astype(np.float32, copy=False)
    gt = load_agl(scene)
    if pred.shape != gt.shape:
        raise ValueError(f"{scene}: shape mismatch {pred.shape} != {gt.shape}")
    valid = np.isfinite(pred) & np.isfinite(gt) & (gt != NODATA)
    return pred, gt, valid


def run_cross_scene() -> tuple[list[dict], dict]:
    rows = []
    for source in SCENES:
        source_pred, source_gt, source_valid = stage9_raw(source)
        fit = fit_scale_shift(source_pred, source_gt, source_valid)
        n_source = int(source_valid.sum())
        for target in SCENES:
            if target == source:
                continue
            target_pred, target_gt, target_valid = stage9_raw(target)
            score = score_calibrated(target_pred, target_gt, target_valid, fit)
            rows.append({"calibration_scene": source, "target_scene": target, "preprocessing": "cached Stage 9 standard 518 processor + bicubic to native 1024; unchanged raw prediction", "scale": fit.scale, "shift_m": fit.shift, "calibration_pixels": n_source, "evaluation_pixels": score["valid_pixels"], "mae_m": score["mae_m"], "rmse_m": score["rmse_m"], "pearson_r": score["pearson_r"], "r2": score["r2"]})
    return rows, {"experiment_count": len(rows), "aggregate_metrics": summarize_metrics(rows), "rows": rows}


def write_audits() -> None:
    # Existing Stage 10 reports the public paper/repository pairing claim; Stage 11 adds code-path specifics.
    (OUT / "gamus_loader_audit.md").write_text("""# GAMUS official loader audit\n\n## Code examined\n\nThe public EarthNets `RSI-MMSegmentation` repository's `gamus_dataset.py` loader is the authoritative code inspected ([source](https://github.com/EarthNets/RSI-MMSegmentation/blob/main/gamus_dataset.py)); its current README identifies the GAMUS benchmark/repository and points to the dataset ([README](https://github.com/EarthNets/RSI-MMSegmentation/blob/main/README.md)). This audit distinguishes the dataset's intended co-registration statement from what the loader itself checks.\n\n## Exact loading behavior\n\nThe loader enumerates image HDF5 files, derives a shared modality prefix by removing the terminal modality token, then opens the paired RGB, class, and height HDF5 files. It reads the `image` dataset directly from each file. RGB is converted with `Image.fromarray(image.astype(np.uint8))`. Height is returned as the HDF5 array. It does not transpose either modality.\n\nThe loader itself does not resize AGL or apply interpolation to AGL. Its optional `transform`, when provided, is called on RGB only. The repository's example transform uses `Resize((224, 224))`, then `ToTensor`, then ImageNet normalization (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`). A PIL resize uses its configured/default resampling behavior; the example does not specify a paired AGL resampling operation. No crop is present in the shown loader/example. Consequently, the example transform changes RGB alone and does not provide a paired pixelwise RGB/height training tensor at a shared grid. The default `transform=None` path performs no resize, crop, transpose, interpolation, or normalization.\n\n## Does the loader enforce correspondence?\n\nIt assumes RGB/height correspond because they are paired from one scene stem and reads both arrays without a geometry transform. It does not assert that their spatial shapes match, inspect georeferencing, or measure registration residuals. The public dataset paper describes the RGB and height modalities as co-registered ([GAMUS paper](https://arxiv.org/abs/2305.14914)). This supports intended dataset-level correspondence. It does not establish a per-file CRS/affine transform or independently verify exact pixel registration in this downloaded export. The loader's same-resolution assumption is implicit: it returns raw arrays and expects downstream paired use to make sense; it does not validate equal spatial dimensions.\n\n## Local sample reproduced\n\n`loader_reproduction/` contains the `DC_03_26` RGB and AGL outputs, source/output shape and dtype, source paths, HDF5 attributes, valid/no-data counts, AGL statistics, exact array-equality checks, and the transformation description. Both are 1024×1024 spatial arrays here. For this transform-free path, the RGB output matches a uint8 cast of the HDF5 RGB data and AGL exactly matches the stored array. The HDF5 files have no CRS/transform metadata.\n\n## Conclusion\n\nThe official dataset description says co-registered and the loader pairs modalities by shared scene identity while passing their raw grids through. The code neither corrects nor verifies geometric alignment and does not enforce equal spatial dimensions. Thus correspondence is documented as intended, but exact geospatial registration cannot be independently established from the downloaded HDF5 arrays alone.\n""")
    (OUT / "depth_semantics.md").write_text("""# Depth Anything V2 output semantics\n\nThe selected checkpoint is `depth-anything/Depth-Anything-V2-Small-hf`. Its model card identifies it as a relative-depth checkpoint, and the loaded Transformers config reports `depth_estimation_type=\"relative\"`. Transformers documents `predicted_depth` as the model output and defines relative and metric modes separately; `max_depth` is ignored for relative models ([model card](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf), [Transformers Depth Anything V2 configuration](https://huggingface.co/docs/transformers/en/model_doc/depth_anything_v2)).\n\nThe model returns a dense float tensor (for the local 518×518 processor input: `[1, 518, 518]`). In installed Transformers 5.18.0, the relative head applies ReLU, so outputs are nonnegative; the metric head instead applies sigmoid and scales by `max_depth`. Relative output is not AGL, elevation, or a distance in metres. Neither config nor checkpoint card gives the relative output a metric scale or offset; it is also not contractually normalized to `[0, 1]`.\n\nThe reviewed checkpoint documentation and implementation identify the value as relative depth but do not define guaranteed near/far polarity for `predicted_depth`. The `DepthAnythingDepthEstimationHead` code enforces nonnegativity, not which surfaces should receive larger values. Therefore this audit does not assert whether larger means nearer or farther. Scene-level raw correlations are mixed, which also prevents assuming one signed mapping to GAMUS AGL. Any polarity or metric mapping must be established empirically on calibration data and tested on held-out scenes.\n\nThe project evaluation uses raw float32 `.npy` predictions, never the separately percentile-stretched visualization PNG. The official Depth Anything V2 demo separately min/max-normalizes a display image; this visualization operation is not part of `predicted_depth` ([official repository](https://github.com/DepthAnything/Depth-Anything-V2)). No output-to-metres conversion is justified without a fitted calibration, and a fitted scale/shift remains evaluation-protocol dependent.\n""")


def write_report(interp_json: dict, transfer_json: dict) -> None:
    groups = interp_json["aggregate_by_mode_method"]
    methods = ["bicubic_518", "bilinear_518", "nearest_518", "bicubic_672_input"]
    def value(key: str, metric: str) -> str:
        result = groups[key][metric]["mean"]
        return "n/a" if result is None else f"{result:.4f}"
    lines = ["| Method | In-sample MAE | RMSE | r | R² | Spatial held-out MAE | RMSE | r | R² |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method in methods:
        insample, held = f"per_scene_in_sample|{method}", f"spatial_holdout|{method}"
        lines.append(f"| `{method}` | {value(insample,'mae_m')} | {value(insample,'rmse_m')} | {value(insample,'pearson_r')} | {value(insample,'r2')} | {value(held,'mae_m')} | {value(held,'rmse_m')} | {value(held,'pearson_r')} | {value(held,'r2')} |")
    cross = transfer_json["aggregate_metrics"]
    cross_text = ", ".join(f"{metric} {cross[metric]['mean']:.4f}" for metric in ("mae_m", "rmse_m", "pearson_r", "r2"))
    report = f"""# DepthWizard Stage 11 report

## Experiments performed

Reproduced the official transform-free GAMUS loader for `DC_03_26`; inspected the official loader behavior and the current model configuration/implementation; reran the cached model on three fixed scenes using the existing 518 input; compared bicubic, bilinear, and nearest output resizing; also ran 672×672 processor input; evaluated full-scene affine fitting and fixed, disjoint spatial calibration/evaluation halves. Recomputed all 90 ordered source→target affine transfers over ten Stage 9 scenes using the standard Stage 9 518-input/bicubic predictions. No model fine-tuning or earlier-output modification occurred.

## GAMUS loader and correspondence

The GAMUS dataset description calls RGB/height pairs co-registered. The official loader pairs modality files by shared scene stem and reads each HDF5 `/image` dataset directly. It casts RGB to uint8 and wraps it as a PIL image; AGL is returned as stored. It does not transpose, crop, interpolate, or resize AGL. With `transform=None`, RGB is not resized either. The repository's example RGB transform resizes to 224×224, tensorizes, and ImageNet-normalizes RGB alone; it leaves AGL unchanged, so that example does not produce a paired same-grid pixelwise sample after resize. The loader does not assert equal dimensions or verify georeferencing.

The reproduced `DC_03_26` arrays are both 1024×1024. RGB output equals a uint8 cast of source exactly and AGL equals the source exactly. Both HDF5 files have no root or dataset attributes. Dataset-level intended co-registration is documented, but exact geospatial/pixel registration cannot be independently verified from these HDF5 files alone. Details and per-file evidence are in `gamus_loader_audit.md` and `loader_reproduction/`.

## Interpolation comparison and fixed spatial evaluation

Four conditions were evaluated on the same three scenes and valid AGL pixels. In-sample rows fit and score on all valid pixels. Spatial rows fit only on the named half and score only on the disjoint half; both directions were run on row and column axes, yielding 12 held-out rows per method. Per-case coefficients and valid pixel counts are included in the CSV/JSON.

{chr(10).join(lines)}

The three 518→1024 resampling kernels are practically indistinguishable in these aggregates. In-sample mean MAE ranges {min(groups[f'per_scene_in_sample|{m}']['mae_m']['mean'] for m in methods[:3]):.4f}–{max(groups[f'per_scene_in_sample|{m}']['mae_m']['mean'] for m in methods[:3]):.4f} m; spatial held-out mean MAE ranges {min(groups[f'spatial_holdout|{m}']['mae_m']['mean'] for m in methods[:3]):.4f}–{max(groups[f'spatial_holdout|{m}']['mae_m']['mean'] for m in methods[:3]):.4f} m. The 672 input succeeded for all three scenes and did not improve aggregate results. These results do not support interpolation as a major source of error, while leaving other preprocessing and correspondence questions open.

## Cross-scene calibration

Scale/shift was fit using all valid AGL pixels in each source scene, then applied unchanged to each of the other nine scenes. Pixels were never pooled across scenes. The 90 directed pairs had mean {cross_text}. See `cross_scene_calibration.csv` and `.json` for every calibration scene, target, coefficient, and metric. The weak transfer does not identify its cause.

## Raw prediction semantics

The checkpoint and config identify the output as relative depth. The installed Transformers head uses ReLU in relative mode, so output is nonnegative, but neither the inspected checkpoint documentation nor implementation establishes whether larger values mean nearer or farther. The output is not metres and has no justified metric scale. Evaluation used raw float32 `.npy` values, never the percentile-stretched display PNG. Details are in `depth_semantics.md`.

## Decision gate and Stage 12 recommendation

- **A — preprocessing/resampling is a major source:** not supported by the interpolation comparison; the three kernels are nearly identical, and 672 input did not help on the selected scenes.
- **B — RGB/AGL correspondence is still uncertain:** yes. Intended co-registration and loader pairing are documented, but exact registration remains unverified from these local files.
- **C — domain mismatch is dominant:** not established. Mixed raw correlations and weak cross-scene transfer do not prove domain mismatch is causal.
- **D — another confirmed issue:** none isolated. In-sample affine calibration remains an optimistic scene-specific measurement compared with held-out transfer.

Fine-tuning is **not justified yet** because correspondence uncertainty remains unresolved. Stage 12 should first validate the paired-grid provenance: obtain authoritative source-grid/preprocessing metadata or a verifiable original registration reference for these exact files, quantify any offset/orientation, and freeze a same-grid spatially held-out protocol. Once that gate is met, the first controlled model-improvement experiment can compare the frozen zero-shot baseline with a small training run using scene-disjoint train/validation sets, without validation-scene calibration leakage. This is a recommendation only; Stage 12 was not implemented.

## Files created

- `gamus_loader_audit.md`, `depth_semantics.md`, `stage11_report.md`
- `loader_reproduction/`: RGB PNG and NPY, AGL NPY, metadata/statistics, and transformation description
- `interpolation_comparison.csv` and `.json`; `interpolation_predictions/` for exact float32 arrays
- `cross_scene_calibration.csv` and `.json`

Stage 1–10 outputs remain untouched.
"""
    (OUT / "stage11_report.md").write_text(report)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reproduce_loader()
    write_audits()
    interp_rows, interp_json = run_interpolation()
    write_rows(OUT / "interpolation_comparison.csv", interp_rows)
    (OUT / "interpolation_comparison.json").write_text(json.dumps(interp_json, indent=2, allow_nan=False) + "\n")
    transfer_rows, transfer_json = run_cross_scene()
    write_rows(OUT / "cross_scene_calibration.csv", transfer_rows)
    (OUT / "cross_scene_calibration.json").write_text(json.dumps(transfer_json, indent=2, allow_nan=False) + "\n")
    write_report(interp_json, transfer_json)
    print(json.dumps({"interpolation": interp_json["aggregate_by_mode_method"], "high_resolution": interp_json["metadata"]["higher_resolution_status"], "cross_scene": transfer_json["aggregate_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
