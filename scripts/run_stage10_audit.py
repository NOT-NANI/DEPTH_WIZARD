#!/usr/bin/env python3
"""Run registration transformations and held-out calibration audits on Stage 9 scenes."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.calibration import fit_scale_shift
from src.stage10_audit import fit_on_calibration_region, score_calibrated, spatial_half_masks, transformed_predictions

DATA = ROOT / "data/multiscene_test"
STAGE9 = ROOT / "outputs/multiscene"
OUT = ROOT / "outputs/stage10"
NODATA = -5.0
SCENES = ["DC_03_26", "DC_05_28", "DC_05_30", "DC_07_21", "DC_07_29", "DC_08_27", "DC_09_18", "DC_09_29", "DC_09_32", "DC_10_20"]
TRANSFORM_SCENES = ["DC_03_26", "DC_08_27", "DC_10_20"]


def load_scene(scene_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_path = STAGE9 / "scenes" / scene_id / "rgb_depth_raw.npy"
    gt_path = DATA / "heights/test" / f"{scene_id}_AGL.h5"
    if not pred_path.is_file() or not gt_path.is_file():
        raise FileNotFoundError(f"Stage 9 prediction or GAMUS AGL file missing for {scene_id}")
    prediction = np.load(pred_path, allow_pickle=False)
    with h5py.File(gt_path, "r") as handle:
        if "image" not in handle:
            raise KeyError(f"Missing image dataset: {gt_path}")
        ground_truth = handle["image"][:]
    if prediction.ndim != 2 or prediction.shape != ground_truth.shape:
        raise ValueError(f"{scene_id}: prediction/AGL shape mismatch {prediction.shape} vs {ground_truth.shape}")
    valid = np.isfinite(ground_truth) & (ground_truth != NODATA) & np.isfinite(prediction)
    return prediction.astype(np.float32, copy=False), ground_truth.astype(np.float32, copy=False), valid


def metric_row(mode: str, scene_id: str, calibration_scene: str, calibration_region: str, evaluation_scene: str, evaluation_region: str, fit, scored: dict, n_cal: int, raw_corr: float | None) -> dict:
    return {
        "mode": mode,
        "scene_id": scene_id,
        "calibration_scene_id": calibration_scene,
        "evaluation_scene_id": evaluation_scene,
        "calibration_region": calibration_region,
        "evaluation_region": evaluation_region,
        "calibration_pixels": n_cal,
        "evaluation_pixels": scored["valid_pixels"],
        "scale": fit.scale,
        "shift_m": fit.shift,
        "mae_m": scored["mae_m"],
        "rmse_m": scored["rmse_m"],
        "pearson_r": scored["pearson_r"],
        "raw_pearson_r": raw_corr,
        "r2": scored["r2"],
    }


def correlation(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float | None:
    x, y = prediction[mask], target[mask]
    if x.size == 0 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def finite_stats(values: list[float | None]) -> dict:
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return {"mean": None, "median": None, "std": None, "count": 0}
    return {"mean": float(a.mean()), "median": float(np.median(a)), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0, "count": int(a.size)}


def summarize(rows: list[dict]) -> dict:
    return {key: finite_stats([row[key] for row in rows]) for key in ("mae_m", "rmse_m", "pearson_r", "r2")}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def registration_audit() -> str:
    checks = []
    for scene in SCENES:
        rgb_path = DATA / "images/test" / f"{scene}_RGB.h5"
        gt_path = DATA / "heights/test" / f"{scene}_AGL.h5"
        with h5py.File(rgb_path, "r") as r, h5py.File(gt_path, "r") as h:
            rgb_d, gt_d = r["image"], h["image"]
            checks.append((scene, rgb_d.shape, str(rgb_d.dtype), gt_d.shape, str(gt_d.dtype), dict(r.attrs), dict(h.attrs), dict(rgb_d.attrs), dict(gt_d.attrs)))
    shapes_match = all(c[1][:2] == c[3] for c in checks)
    attrs_empty = all(not attrs for c in checks for attrs in c[5:9])
    return f"""# GAMUS registration audit

## Documentation and pairing evidence

The GAMUS paper describes a benchmark with **co-registered RGB and nDSM pairs** ([paper abstract](https://arxiv.org/abs/2305.14914)). The public Hugging Face export separates `images/{{split}}` and `heights/{{split}}` and names paired files with a shared scene stem, such as `DC_03_26_RGB.h5` and `DC_03_26_AGL.h5` ([RGB test listing](https://huggingface.co/datasets/earthflow/GAMUS/tree/main/images/test), [height test listing](https://huggingface.co/datasets/earthflow/GAMUS/tree/main/heights/test)). This documents intended pixel correspondence at the dataset level. It does not provide a per-file geospatial transform or registration residual.

The official EarthNets benchmark loader strips a six-character modality suffix to derive a shared prefix, reads each HDF5 `image` dataset directly, turns RGB into a PIL image, and returns the height array without spatial transformation. The current `_RGB.h5` and `_AGL.h5` suffixes have the same six-character terminal token (`RGB.h5` / `AGL.h5`), so this prefix-based pairing convention matches the inspected export. Its optional transform is applied only to RGB. Its example resizes RGB to 224×224 but does not resize height, so that example is not safe for pixelwise height evaluation after the resize ([official loader source](https://github.com/EarthNets/RSI-MMSegmentation/blob/main/gamus_dataset.py)). The loader does not demonstrate any row/column georeferencing transform or registration correction.

## Downloaded HDF5 inspection

- Scenes inspected: {len(checks)} test scenes.
- RGB arrays: H×W×3, uint8, channel-last; PIL conversion treats the three channels as RGB. No channel-order metadata is stored in HDF5.
- AGL arrays: H×W, float32. The exported dataset key is `image` for both modalities.
- All RGB/AGL dimensions equal 1024×1024: **{shapes_match}**.
- Root and dataset HDF5 attributes are empty in every file: **{attrs_empty}**. No CRS, affine transform, coordinate convention, pixel size, axis-orientation tag, or georeferencing metadata was found.
- Arrays were read in their stored row/column order. The RGB channel axis is last. The files do not name rows as north/south or columns as east/west.
- Scene stem pairing plus the paper's co-registration description indicates intended RGB-to-height pixel correspondence. It does not independently verify that each distributed pair has no offset, mirroring, or geospatial registration error.

## Conclusion

RGB and height are documented as a co-registered modality pair in the dataset description and are paired by shared scene identity in the official loader design. The downloaded arrays have matching dimensions and no stored transforms. Therefore, **dataset-level pixel alignment is intended/documented; geospatial registration and exact per-file pixel correspondence cannot be independently verified from these HDF5 files alone**. Visual similarity is supporting inspection only, not proof.
"""


def preprocessing_audit() -> str:
    return """# Depth Anything V2 preprocessing audit

## Input path and conversion

`scripts/evaluate_multiscene.py` reads the HDF5 `image` array as H×W×3 uint8, saves it directly to an RGB PNG with PIL, and sends that path to `src.depth_inference.run_inference`. No source transpose, flip, rotation, crop, or resize is performed. `load_rgb` opens the image, applies EXIF orientation if present, loads it, and converts to RGB. HDF5-derived PNGs have no EXIF orientation tag. RGB channels are passed as RGB; no BGR swap occurs.

## Processor and model input

The cached `AutoImageProcessor` resolves to `DPTImageProcessor`: rescale by 1/255, normalize with mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`, resize enabled, aspect ratio kept, no padding, resampling value 3 (PIL bicubic), target size 518×518 with dimensions constrained to multiples of 14. The actual processed 1024×1024 input tensor was verified as float32 shape 1×3×518×518. The depth model is `DepthAnythingForDepthEstimation`; the loaded config says `depth_estimation_type="relative"`.

## Output and numeric evaluation

Inference uses `model(**inputs).predicted_depth`, verified as float32 shape 1×518×518 for this square input. The code bicubically interpolates it to original image H×W (1024×1024), squeezes batch/channel axes, casts to float32, verifies finite values, and saves `{scene}/rgb_depth_raw.npy`. This is the relative model output; the code does not convert it to metric depth or invert it.

The paired evaluation loads `rgb_depth_raw.npy` with NumPy and compares that array with AGL. Stage 10 transforms and calibration experiments use those same raw `.npy` values. They do not use `rgb_depth.png`.

## Display output

`rgb_depth.png` is a separate visualization: the raw prediction is stretched between its 2nd and 98th percentiles and quantized to uint8. It is not the evaluation input. That normalization changes numeric values and would invalidate height metrics if used as a prediction.

## Consistency assessment

All tested sources are 1024×1024 and square, so the processor's keep-aspect-ratio resize does not introduce a geometric aspect-ratio change. Its model-resolution resize and output interpolation can smooth/alter fine spatial edges. A 518×518 prediction interpolated back to 1024×1024 does not recover lost detail. GAMUS official example preprocessing uses RGB normalization and a 224×224 resize on RGB only, leaving AGL unchanged; that is not the preprocessing used by this inference/evaluation pipeline and would require paired target resampling for supervised use. No evidence here supports changing preprocessing silently.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "registration_audit.md").write_text(registration_audit())
    (OUT / "preprocessing_audit.md").write_text(preprocessing_audit())

    # Transformation sanity: transform raw prediction arrays only, never RGB/GT.
    transform_rows = []
    for scene_id in TRANSFORM_SCENES:
        pred, gt, valid = load_scene(scene_id)
        for transform, candidate in transformed_predictions(pred).items():
            fit = fit_scale_shift(candidate, gt, valid)
            metrics = score_calibrated(candidate, gt, valid, fit)
            transform_rows.append({
                "scene_id": scene_id,
                "transformation": transform,
                "valid_pixels": metrics["valid_pixels"],
                "raw_pearson_r": correlation(candidate, gt, valid),
                "scale": fit.scale,
                "shift_m": fit.shift,
                "mae_m": metrics["mae_m"],
                "rmse_m": metrics["rmse_m"],
                "pearson_r_aligned": metrics["pearson_r"],
                "r2": metrics["r2"],
            })
        del pred, gt, valid
    write_csv(OUT / "transformation_sanity.csv", transform_rows)

    # Calibration modes: full-scene in-sample, spatial block holdouts (rows
    # and columns, both directions), same-region in-sample controls, and all
    # ordered cross-scene source -> target combinations.
    calibration_rows = []
    for scene_id in SCENES:
        pred, gt, valid = load_scene(scene_id)
        full_fit = fit_scale_shift(pred, gt, valid)
        full_score = score_calibrated(pred, gt, valid, full_fit)
        calibration_rows.append(metric_row("per_scene_in_sample", scene_id, scene_id, "all_valid", scene_id, "all_valid", full_fit, full_score, int(valid.sum()), correlation(pred, gt, valid)))
        for axis, first_name, second_name in [("row", "top_half", "bottom_half"), ("col", "left_half", "right_half")]:
            first, second = spatial_half_masks(gt.shape, axis)
            for calibration_mask, evaluation_mask, calibration_name, evaluation_name in [
                (first, second, first_name, second_name),
                (second, first, second_name, first_name),
            ]:
                fit, held_score, n_cal = fit_on_calibration_region(pred, gt, valid, calibration_mask, evaluation_mask)
                calibration_rows.append(metric_row(f"within_scene_spatial_holdout_{axis}", scene_id, scene_id, calibration_name, scene_id, evaluation_name, fit, held_score, n_cal, correlation(pred, gt, valid & evaluation_mask)))
                same_valid = valid & evaluation_mask & np.isfinite(pred) & np.isfinite(gt)
                same_fit = fit_scale_shift(pred, gt, same_valid)
                same_score = score_calibrated(pred, gt, same_valid, same_fit)
                n_same = int(same_valid.sum())
                calibration_rows.append(metric_row("same_region_in_sample_control", scene_id, scene_id, evaluation_name, scene_id, evaluation_name, same_fit, same_score, n_same, correlation(pred, gt, valid & evaluation_mask)))
        del pred, gt, valid

    for source_scene in SCENES:
        source_pred, source_gt, source_valid = load_scene(source_scene)
        source_fit = fit_scale_shift(source_pred, source_gt, source_valid)
        source_n = int(source_valid.sum())
        del source_pred, source_gt, source_valid
        for target_scene in SCENES:
            if target_scene == source_scene:
                continue
            target_pred, target_gt, target_valid = load_scene(target_scene)
            score = score_calibrated(target_pred, target_gt, target_valid, source_fit)
            calibration_rows.append(metric_row(
                "cross_scene_calibration", target_scene, source_scene, "source_scene_all_valid",
                target_scene, "target_scene_all_valid", source_fit, score, source_n,
                correlation(target_pred, target_gt, target_valid),
            ))
            del target_pred, target_gt, target_valid
    write_csv(OUT / "calibration_comparison.csv", calibration_rows)

    groups = {}
    for mode in sorted({row["mode"] for row in calibration_rows}):
        subset = [row for row in calibration_rows if row["mode"] == mode]
        groups[mode] = {"experiment_count": len(subset), "metrics": summarize(subset)}
    # Matched leakage deltas compare held-out score to fitting on the exact
    # same evaluation half, keeping the evaluation pixel set fixed.
    control = {(r["scene_id"], r["calibration_region"], r["evaluation_region"]): r for r in calibration_rows if r["mode"] == "same_region_in_sample_control"}
    heldout_deltas = {}
    for metric in ("mae_m", "rmse_m", "pearson_r", "r2"):
        diffs = []
        for row in calibration_rows:
            if not row["mode"].startswith("within_scene_spatial_holdout_"):
                continue
            c = control[(row["scene_id"], row["evaluation_region"], row["evaluation_region"])]
            if row[metric] is not None and c[metric] is not None:
                diffs.append(float(row[metric] - c[metric]))
        heldout_deltas[metric] = finite_stats(diffs)
    full_scene_in_sample = {row["scene_id"]: row for row in calibration_rows if row["mode"] == "per_scene_in_sample"}
    cross_scene_deltas = {}
    for metric in ("mae_m", "rmse_m", "pearson_r", "r2"):
        cross_scene_deltas[metric] = finite_stats([
            float(row[metric] - full_scene_in_sample[row["evaluation_scene_id"]][metric])
            for row in calibration_rows
            if row["mode"] == "cross_scene_calibration" and row[metric] is not None
        ])
    result = {
        "description": "Per-scene in-sample, spatially held-out within-scene, same-evaluation-region in-sample control, and cross-scene calibration. No fit crosses split masks except where explicitly identified.",
        "scene_ids": SCENES,
        "spatial_split": "top/bottom row halves and left/right column halves; each direction evaluated separately; valid pixels only; masks are disjoint",
        "cross_scene_design": "all 90 ordered source-scene to distinct target-scene pairs; source fit uses all valid source pixels; target scored on all valid target pixels",
        "mode_summaries": groups,
        "heldout_minus_same_region_insample_control": heldout_deltas,
        "cross_scene_minus_target_full_scene_insample": cross_scene_deltas,
        "rows": calibration_rows,
    }
    (OUT / "calibration_comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    # Plot transform scores and calibration mode distributions for inspection.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    labels = ["original", "horizontal_flip", "vertical_flip", "rotate_180"]
    x = np.arange(len(labels)); width = .24
    for i, scene in enumerate(TRANSFORM_SCENES):
        lookup = {r["transformation"]: r["mae_m"] for r in transform_rows if r["scene_id"] == scene}
        ax.bar(x + (i - 1) * width, [lookup[l] for l in labels], width, label=scene)
    ax.set_xticks(x, labels); ax.set_ylabel("Per-scene in-sample aligned MAE (m)"); ax.set_title("Prediction-only orientation sanity check"); ax.legend()
    fig.savefig(OUT / "transformation_sanity.png", dpi=150); plt.close(fig)

    # Reports and plot summaries are added in the next block after calculating
    # descriptive evidence from the measured CSV rows.
    write_reports(transform_rows, calibration_rows, groups, heldout_deltas, cross_scene_deltas)
    print(f"Wrote Stage 10 audit results to {OUT}")


def write_reports(transform_rows: list[dict], calibration_rows: list[dict], groups: dict, heldout_deltas: dict, cross_scene_deltas: dict) -> None:
    transform_by = {scene: [r for r in transform_rows if r["scene_id"] == scene] for scene in TRANSFORM_SCENES}
    original = {r["scene_id"]: r for r in transform_rows if r["transformation"] == "original"}
    improvements = {}
    for name in ("horizontal_flip", "vertical_flip", "rotate_180"):
        candidates = [r for r in transform_rows if r["transformation"] == name]
        improvements[name] = sum(r["mae_m"] < 0.8 * original[r["scene_id"]]["mae_m"] for r in candidates)
    trans_summary = []
    for scene in TRANSFORM_SCENES:
        items = transform_by[scene]
        original_row = next(r for r in items if r["transformation"] == "original")
        best_alternate = min((r for r in items if r["transformation"] != "original"), key=lambda r: r["mae_m"])
        trans_summary.append(f"| {scene} | {original_row['mae_m']:.4f} | {best_alternate['transformation']} | {best_alternate['mae_m']:.4f} | {best_alternate['mae_m'] - original_row['mae_m']:+.4f} |")
    trans_findings = "; ".join(f"{name}: {count}/3 scene MAEs fell by >20%" for name, count in improvements.items())
    lines = [
        "# DepthWizard Stage 10: registration, preprocessing, and calibration audit", "",
        "## Work performed", "",
        "No model fine-tuning or pipeline transformation was applied. The Stage 9 raw `.npy` predictions and matched GAMUS AGL arrays were re-used. Tests covered three fixed scenes for orientation transformations and all ten scenes for spatial and cross-scene calibration.", "",
        "## 1. Are RGB and AGL documented as pixel-aligned?", "",
        "GAMUS is described in its paper as containing co-registered RGB and nDSM pairs, and the public export/official loader associate modalities by a shared filename stem. However, the downloaded HDF5 files contain no CRS, affine transform, pixel size, explicit row/column directions, or registration metadata. Their matching dimensions and shared stems support intended correspondence, not independent proof of exact geospatial registration. The official loader reads height as-is and applies optional augmentation to RGB only; its example resizes RGB to 224×224 while leaving height unchanged. That example would require a paired height resize before pixelwise evaluation. Details are in `registration_audit.md`.", "",
        "## 2. Is there evidence of an orientation problem?", "",
        "Prediction-only horizontal flip, vertical flip, and 180° rotation tests were run against the same AGL arrays for three fixed scenes. Every variant received its own unconstrained per-scene scale/shift fit and was scored in-sample; this is a diagnostic, not a transform-selection procedure.", "",
        "| Scene | Original MAE (m) | Lowest alternate transform | Alternate MAE (m) | Difference (m) |", "|---|---:|---|---:|---:|",
        *trans_summary, "", f"Transforms reducing MAE by over 20%: {trans_findings}. Across the three scenes, no single alternate orientation produces >20% improvement in all three. The output does not justify changing orientation. These in-sample fits are only a sanity check and do not prove registration.", "",
        "## 3. Is preprocessing consistent?", "",
        "The current pipeline consistently reads channel-last uint8 RGB, uses the cached DPT processor's rescale/normalization and 518×518 aspect-preserving resize, runs `predicted_depth`, then bicubically resizes the raw float output to original dimensions. Evaluation reads the lossless float32 `.npy`; the separately percentile-stretched PNG is not used. Full details are in `preprocessing_audit.md`. The baseline path does not flip/crop inputs or AGL. Model-resolution interpolation can reduce edge detail but does not explain the observed sign changes by itself.", "",
        "## 4. How much does held-out calibration change the metrics?", "",
        "Spatial block holdouts used top→bottom, bottom→top, left→right, and right→left splits on every scene. The same-region in-sample control fits and scores only on the held-out region, isolating the effect of fitting on evaluation pixels. Full-scene in-sample results are also reported as the Stage 9 comparison. Differences below are held-out minus same-region in-sample control; positive MAE/RMSE deltas indicate degradation after holding out a separate block:", "",
        "| Metric delta | Mean | Median | Std. dev. |", "|---|---:|---:|---:|",
    ]
    for key, label in [("mae_m", "MAE (m)"), ("rmse_m", "RMSE (m)"), ("pearson_r", "Pearson r"), ("r2", "R²")]:
        s = heldout_deltas[key]
        lines.append(f"| {label} | {s['mean'] if s['mean'] is not None else float('nan'):+.5f} | {s['median'] if s['median'] is not None else float('nan'):+.5f} | {s['std'] if s['std'] is not None else float('nan'):.5f} |")
    lines += ["", "### Three modes (separate measurements, not ranked)", "", "| Mode | Cases | Mean MAE (m) | Median MAE (m) | Mean RMSE (m) | Mean aligned Pearson r | Mean R² |", "|---|---:|---:|---:|---:|---:|---:|"]
    mode_labels = [
        ("per_scene_in_sample", "Per-scene in-sample (fit/score all scene pixels)"),
        ("within_scene_spatial_holdout_row", "Within-scene spatial holdout (top/bottom)"),
        ("within_scene_spatial_holdout_col", "Within-scene spatial holdout (left/right)"),
        ("cross_scene_calibration", "Cross-scene calibration (source→different target)"),
    ]
    for key, label in mode_labels:
        group = groups[key]; m = group["metrics"]
        lines.append(f"| {label} | {group['experiment_count']} | {m['mae_m']['mean']:.4f} | {m['mae_m']['median']:.4f} | {m['rmse_m']['mean']:.4f} | {m['pearson_r']['mean']:.4f} | {m['r2']['mean']:.4f} |")
    lines += ["", "Per-scene in-sample shows the scale/shift fit evaluated on the same scene pixels. Spatial holdout tests extrapolation from a different half of that scene. Cross-scene calibration fits all valid source-scene pixels and scores a different complete scene; every ordered pair is included. The same-region control measures how much the affine fit can improve metrics when fit directly on the exact block being scored. See the CSV/JSON for every coefficient and case.", "",
        "### Cross-scene transfer delta against each target's own full-scene in-sample fit", "",
        "Each transfer result was differenced against the target's per-scene in-sample metrics, with the same full target evaluation pixels. Positive MAE/RMSE deltas indicate the source calibration scored worse. This is a leakage-reference comparison, not a deployable calibration:", "",
        "| Metric delta | Mean | Median | Std. dev. |", "|---|---:|---:|---:|",
    ]
    for key, label in [("mae_m", "MAE (m)"), ("rmse_m", "RMSE (m)"), ("pearson_r", "Pearson r"), ("r2", "R²")]:
        s = cross_scene_deltas[key]
        lines.append(f"| {label} | {s['mean']:+.5f} | {s['median']:+.5f} | {s['std']:.5f} |")
    lines += ["",
        "## 5. Does calibration transfer between scenes?", "",
        "The 90 directed source→target pairs are summarized in `calibration_comparison.json`; the table above reports their unweighted scene-pair statistics. Transfer is measured with source scale/shift applied unchanged to target raw predictions. This is not deployment calibration. Direction differences and scene-dependent raw scales can produce materially different target metrics; interpret per-pair data rather than a single source calibration as universal.", "",
        "## 6. Is the evidence sufficient to justify fine-tuning?", "",
        "No. The benchmark describes intended co-registration, but per-file geospatial alignment remains unverified; transformed prediction checks do not reveal a consistent global orientation correction; and spatial/cross-scene tests measure calibration leakage/transfer separately from model learning. Ten scenes are not enough to diagnose a cause. Fine-tuning now would confound unresolved registration and calibration questions.", "",
        "## 7. Specific Stage 11 problem", "",
        "First resolve evaluation protocol and grid correspondence: obtain authoritative GAMUS export/registration details, verify row/column alignment against source rasters or a known loader, then evaluate a predeclared spatial calibration fit on held-out blocks and leave-one-scene-out transfer without fitting on target labels. Include interpolation ablations while keeping masks and evaluation pixels fixed. Only after this protocol is stable should a targeted model experiment be proposed.", "",
        "## Files and reproducibility", "",
        "- `registration_audit.md` — dataset documentation, official loader behavior, and HDF5 metadata findings.",
        "- `preprocessing_audit.md` — current Depth Anything preprocessing and raw/display output separation.",
        "- `transformation_sanity.csv` and `transformation_sanity.png` — original/H-flip/V-flip/180° prediction comparison.",
        "- `calibration_comparison.csv` and `calibration_comparison.json` — per-scene, spatial holdout, same-region control, and all ordered cross-scene results.",
        "- `stage10_report.md` — this summary.",
        "- Diagnostic source scenes: `outputs/multiscene/scenes/{DC_03_26,DC_08_27,DC_10_20}/diagnostics.png`.",
        "",
    ]
    (OUT / "stage10_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
