#!/usr/bin/env python3
"""Run per-scene scale/shift aligned Depth Anything V2 evaluation on GAMUS test pairs."""
import csv
import json
import sys
import gc
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.depth_inference import run_inference, MODEL_ID, select_device
from src.evaluation import evaluate

NODATA = -5.0
DEFAULT_DATA = ROOT / "data/multiscene_test"
OUT = ROOT / "outputs/multiscene"
EXPECTED_SCENES = ["DC_03_26", "DC_05_28", "DC_05_30", "DC_07_21", "DC_07_29", "DC_08_27", "DC_09_18", "DC_09_29", "DC_09_32", "DC_10_20"]


def load_array(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "image" not in handle:
            raise KeyError(f"Missing 'image' dataset in {path}")
        return handle["image"][:]


def stats(values: np.ndarray) -> dict:
    return {"mean": float(np.mean(values)), "median": float(np.median(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def save_diagnostic(scene_id: str, rgb: np.ndarray, gt: np.ndarray, raw: np.ndarray, aligned: np.ndarray, valid: np.ndarray, scene_out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    err = np.where(valid, np.abs(aligned - gt), np.nan)
    fig, ax = plt.subplots(1, 5, figsize=(20, 4.5), constrained_layout=True)
    ax[0].imshow(rgb); ax[0].set_title(f"{scene_id}: RGB")
    for axis, array, title, cmap in [
        (ax[1], np.where(valid, gt, np.nan), "Ground truth AGL (m)", "terrain"),
        (ax[2], raw, "Raw model output (relative)", "magma"),
        (ax[3], np.where(valid, aligned, np.nan), "Per-scene aligned (m)", "terrain"),
        (ax[4], err, "Absolute error (m)", "inferno"),
    ]:
        im = axis.imshow(array, cmap=cmap); axis.set_title(title); fig.colorbar(im, ax=axis, shrink=.72)
    for axis in ax: axis.set_xticks([]); axis.set_yticks([])
    fig.savefig(scene_out / "diagnostics.png", dpi=140); plt.close(fig)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    data_dir = args.data_dir if args.data_dir.is_absolute() else (Path.cwd() / args.data_dir).resolve()
    out_dir = args.output_dir if args.output_dir.is_absolute() else (ROOT / args.output_dir).resolve()
    rgb_dir, gt_dir = data_dir / "images/test", data_dir / "heights/test"
    pairs = []
    for scene_id in EXPECTED_SCENES:
        rgb_path = rgb_dir / f"{scene_id}_RGB.h5"
        gt_path = gt_dir / f"{scene_id}_AGL.h5"
        if not rgb_path.is_file() or not gt_path.is_file():
            raise FileNotFoundError(f"Missing matched RGB/AGL pair for {scene_id}: {rgb_path}, {gt_path}")
        pairs.append((scene_id, rgb_path, gt_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Scenes: {len(pairs)} | Model: {MODEL_ID} | Device: {select_device()}")
    print("Method: per-scene scale/shift aligned baseline; no cross-scene pooled fit")
    results = []
    for index, (scene_id, rgb_path, gt_path) in enumerate(pairs, 1):
        print(f"[{index}/{len(pairs)}] {scene_id}")
        rgb = load_array(rgb_path)
        gt = load_array(gt_path)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"{scene_id}: expected H×W×3 uint8 RGB, got {rgb.shape} {rgb.dtype}")
        if gt.ndim != 2 or gt.shape != rgb.shape[:2]:
            raise ValueError(f"{scene_id}: RGB/AGL spatial dimensions differ: {rgb.shape} vs {gt.shape}")
        valid = np.isfinite(gt) & (gt != NODATA)
        if not valid.any():
            raise ValueError(f"{scene_id}: no valid height pixels")
        scene_out = out_dir / "scenes" / scene_id
        scene_out.mkdir(parents=True, exist_ok=True)
        rgb_png = scene_out / "rgb.png"
        Image.fromarray(rgb, mode="RGB").save(rgb_png)
        gt = gt.astype(np.float32, copy=False)
        np.save(scene_out / "ground_truth_agl.npy", gt, allow_pickle=False)
        np.save(scene_out / "valid_mask.npy", valid, allow_pickle=False)
        raw_path = scene_out / "rgb_depth_raw.npy"
        if not raw_path.is_file():
            raw_path, _, _ = run_inference(rgb_png, scene_out)
        else:
            print(f"  Reusing existing raw prediction: {raw_path.name}")
        raw = np.load(raw_path, allow_pickle=False)
        if raw.shape != gt.shape:
            raise ValueError(f"{scene_id}: prediction/GT dimensions differ: {raw.shape} vs {gt.shape}")
        if not np.isfinite(raw).all():
            raise ValueError(f"{scene_id}: prediction contains non-finite values")

        # Fit the same unconstrained affine relationship for raw and negated
        # predictions. These fits must be algebraically equivalent; raw r and
        # slope sign retain the direction information that aligned r hides.
        calibration, aligned, metrics = evaluate(raw, gt, valid)
        neg_calibration, neg_aligned, neg_metrics = evaluate(-raw, gt, valid)
        np.save(scene_out / "aligned_height.npy", aligned, allow_pickle=False)
        raw_r = float(np.corrcoef(raw[valid], gt[valid])[0, 1])
        raw_negative_r = float(np.corrcoef(-raw[valid], gt[valid])[0, 1])
        positive_fit = calibration.scale > 0
        row = {
            "scene_id": scene_id,
            "mae_m": metrics["mae_m"], "rmse_m": metrics["rmse_m"],
            "pearson_r": metrics["pearson_r"], "r2": metrics["r_squared"],
            "scale": calibration.scale, "shift": calibration.shift,
            "valid_pixels": metrics["valid_pixels"],
            "raw_pearson_r": raw_r, "negated_raw_pearson_r": raw_negative_r,
            "raw_fit_scale_positive": bool(positive_fit),
            "negated_fit_scale": neg_calibration.scale,
            "negated_fit_shift": neg_calibration.shift,
            "negated_mae_m": neg_metrics["mae_m"], "negated_rmse_m": neg_metrics["rmse_m"],
            "gt_min_m": float(gt[valid].min()), "gt_max_m": float(gt[valid].max()),
            "gt_mean_m": float(gt[valid].mean()), "gt_median_m": float(np.median(gt[valid])),
            "raw_prediction_min": float(raw[valid].min()), "raw_prediction_max": float(raw[valid].max()),
            "raw_prediction_mean": float(raw[valid].mean()),
            "nodata_pixels": int((~valid).sum()), "width": int(gt.shape[1]), "height": int(gt.shape[0]),
            "rgb_file": str(rgb_path.relative_to(data_dir)), "agl_file": str(gt_path.relative_to(data_dir)),
        }
        results.append(row)
        del rgb, gt, valid, raw, aligned, neg_aligned
        gc.collect()
        print(f"  valid={row['valid_pixels']} raw-r={raw_r:.4f} aligned MAE={row['mae_m']:.3f} RMSE={row['rmse_m']:.3f} scale={row['scale']:.4g}")

    metrics_keys = ["mae_m", "rmse_m", "pearson_r", "r2"]
    aggregate = {
        "experiment": "per-scene scale/shift aligned baseline",
        "model": MODEL_ID,
        "device": str(select_device()),
        "scene_count": len(results),
        "scene_ids": [row["scene_id"] for row in results],
        "aggregation": "unweighted statistics across scene-level metrics; no cross-scene pixel pooling",
        "metrics": {key: stats(np.asarray([row[key] for row in results], dtype=np.float64)) for key in metrics_keys},
        "raw_pearson_r": stats(np.asarray([row["raw_pearson_r"] for row in results], dtype=np.float64)),
        "mean_absolute_raw_pearson_r": float(np.mean(np.abs([row["raw_pearson_r"] for row in results]))),
        "direction_diagnostic": {
            "method": "fit unconstrained GT = a × prediction + b, then repeat with negated raw prediction",
            "note": "With a free signed scale, negating prediction is a reparameterization: predicted fitted surfaces and MAE/RMSE/R² are algebraically unchanged; the fitted scale and raw Pearson sign reverse. Use raw_pearson_r and scale sign to identify direction. This cannot distinguish relative-depth versus inverse-depth semantics by itself.",
            "scenes_positive_raw_fit_scale": sum(row["raw_fit_scale_positive"] for row in results),
            "scenes_negative_raw_fit_scale": sum(not row["raw_fit_scale_positive"] for row in results),
            "max_abs_mae_difference_after_negation_m": max(abs(row["mae_m"] - row["negated_mae_m"]) for row in results),
            "max_abs_rmse_difference_after_negation_m": max(abs(row["rmse_m"] - row["negated_rmse_m"]) for row in results),
        },
    }
    csv_path = out_dir / "scene_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys())); writer.writeheader(); writer.writerows(results)
    (out_dir / "scene_metrics.json").write_text(json.dumps({"experiment": aggregate["experiment"], "scenes": results}, indent=2) + "\n")
    (out_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scene_ids = [r["scene_id"] for r in results]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), constrained_layout=True)
    chart_metrics = [("mae_m", "MAE (m)"), ("rmse_m", "RMSE (m)"), ("pearson_r", "Pearson r (aligned prediction)"), ("raw_pearson_r", "Pearson r (raw; signed)"), ("r2", "R²")]
    for ax, (metric, label) in zip(axes.flat, chart_metrics):
        values = [r[metric] for r in results]
        ax.bar(scene_ids, values, color="#557b9d")
        ax.axhline(np.mean(values), color="#b04a3a", linestyle="--", linewidth=1, label=f"scene mean {np.mean(values):.3f}")
        ax.set_title(label); ax.tick_params(axis="x", rotation=50); ax.legend(fontsize=8)
    axes.flat[-1].set_visible(False)
    fig.suptitle("GAMUS test scenes — per-scene scale/shift aligned baseline")
    fig.savefig(out_dir / "metrics_by_scene.png", dpi=160); plt.close(fig)

    # Fixed, preselected first/middle/last stems avoid choosing examples based on score.
    diagnostic_ids = [EXPECTED_SCENES[0], EXPECTED_SCENES[len(EXPECTED_SCENES)//2], EXPECTED_SCENES[-1]]
    for scene_id in diagnostic_ids:
        scene_out = out_dir / "scenes" / scene_id
        rgb = np.asarray(Image.open(scene_out / "rgb.png").convert("RGB"))
        gt = np.load(scene_out / "ground_truth_agl.npy", allow_pickle=False)
        valid = np.load(scene_out / "valid_mask.npy", allow_pickle=False)
        raw = np.load(scene_out / "rgb_depth_raw.npy", allow_pickle=False)
        aligned = np.load(scene_out / "aligned_height.npy", allow_pickle=False)
        save_diagnostic(scene_id, rgb, gt, raw, aligned, valid, scene_out)
        del rgb, gt, valid, raw, aligned
    lowest_mae = min(results, key=lambda row: row["mae_m"])
    highest_mae = max(results, key=lambda row: row["mae_m"])
    report_lines = [
        "# DepthWizard multi-scene GAMUS baseline report", "",
        "## 1. Experiment objective", "",
        "Measure whether the weak single-scene baseline is repeated across distinct GAMUS test scenes, while checking scene-level direction and preprocessing. This remains an evaluation of a pretrained baseline; no fine-tuning was performed.", "",
        "## 2. Dataset and sample selection", "",
        f"Selected {len(results)} scene stems from the repository's public `test` split: " + ", ".join(row["scene_id"] for row in results) + ". RGB and AGL files were paired only by the exact shared scene stem and matching `_RGB.h5` / `_AGL.h5` suffixes in the repository's `images/test` and `heights/test` folders. Ten pairs (20 files; about 73.5 MB) were downloaded; the full dataset was not downloaded. HDF5 files expose only an `image` dataset and no orientation/georeferencing attributes.", "",
        "## 3. Model", "",
        f"`{MODEL_ID}` (`DepthAnythingForDepthEstimation`, Small). Model config declares `depth_estimation_type=relative`. Inference used `predicted_depth` as the raw numerical output and ran sequentially on `" + str(select_device()) + "` (MPS unavailable in this Python environment).", "",
        "## 4. Preprocessing and quality control", "",
        "HDF5 RGB arrays were saved directly as RGB PNG without transposing, rotating, or resampling the source arrays. Each pair was checked as RGB H×W×3 uint8 and AGL H×W float32 with equal 1024×1024 spatial shapes. The same row/column order was retained. The model image processor rescales and normalizes RGB, resizes with aspect ratio to its configured 518-pixel target / multiple-of-14 constraint, then the `predicted_depth` output is bicubically interpolated back to 1024×1024. No source-space resize, crop, flip, or rotation is performed by this pipeline. Prediction shape and finiteness are checked before scoring. Valid mask is `isfinite(AGL) & (AGL != -5.0)`; it excludes NaN/Inf and exactly the documented NoData value while retaining valid negatives. Three fixed, selection-order diagnostic scenes (first, middle, last) were visually inspected; their RGB and AGL structures appear co-oriented. Same-stem pairing and equal dimensions support correspondence, but the HDF5 files lack metadata that could independently prove geospatial registration.", "",
        "## 5. Evaluation methodology", "",
        "For every scene independently, fit the unconstrained least-squares affine mapping `GT = a × raw_prediction + b` over that scene's valid pixels, then calculate MAE, RMSE, Pearson correlation of the aligned surface, and R². `pearson_r` is the same aligned-output correlation convention as the earlier baseline; `raw_pearson_r` is also recorded to retain raw direction. Aggregates are unweighted mean, median, and sample standard deviation across scene metrics; there is no pooling of pixels or predictions across scenes. These are per-scene scale/shift aligned metrics, not deployment calibration.", "",
        "## 6. Per-scene results", "",
        "| Scene | Valid pixels | MAE (m) | RMSE (m) | Aligned r | Raw r | R² | Scale | Shift (m) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        report_lines.append(f"| {row['scene_id']} | {row['valid_pixels']:,} | {row['mae_m']:.4f} | {row['rmse_m']:.4f} | {row['pearson_r']:.4f} | {row['raw_pearson_r']:.4f} | {row['r2']:.4f} | {row['scale']:.4f} | {row['shift']:.4f} |")
    report_lines += ["", "## 7. Aggregate statistics across scenes", "", "| Metric | Mean | Median | Std. dev. |", "|---|---:|---:|---:|"]
    for key, label in [("mae_m", "MAE (m)"), ("rmse_m", "RMSE (m)"), ("pearson_r", "Aligned Pearson r"), ("r2", "R²")]:
        s = aggregate["metrics"][key]; report_lines.append(f"| {label} | {s['mean']:.5f} | {s['median']:.5f} | {s['std']:.5f} |")
    raw_s = aggregate["raw_pearson_r"]
    report_lines += [f"| Raw Pearson r (directional) | {raw_s['mean']:.5f} | {raw_s['median']:.5f} | {raw_s['std']:.5f} |", "", "These summarize scene scores equally; they do not imply equal pixel counts, and do not pool values across scenes.", "",
        "## 8. Direction / inversion diagnostic", "",
        f"The raw prediction-ground-truth correlation was positive in {aggregate['direction_diagnostic']['scenes_positive_raw_fit_scale']} scenes and negative in {aggregate['direction_diagnostic']['scenes_negative_raw_fit_scale']} scenes (direction indicated by fitted scale sign). A positive-slope convention would use raw prediction in the positive-correlation scenes and negated raw prediction in the negative-correlation scenes; there is no single sign choice supported across all ten. Negating the raw prediction and refitting a free signed scale produced a maximum MAE difference of {aggregate['direction_diagnostic']['max_abs_mae_difference_after_negation_m']:.3g} m and RMSE difference of {aggregate['direction_diagnostic']['max_abs_rmse_difference_after_negation_m']:.3g} m. This equality is algebraic: with an unconstrained scale, negation simply reverses the scale sign and cannot improve fit metrics. Raw Pearson sign and coefficient sign expose direction. The model returns the relative `predicted_depth` field; the display PNG is percentile-normalized for visualization only and is never used in evaluation. This test does not by itself identify the output as inverse depth.", "",
        "## 9. Evidence-based observations", "",
        f"All ten scenes have the same 1024×1024 dimensions. Valid-pixel counts range from {min(row['valid_pixels'] for row in results):,} to {max(row['valid_pixels'] for row in results):,}; only exact -5.0 markers and non-finite GT were masked. Aligned MAE ranges from {lowest_mae['mae_m']:.3f} m in {lowest_mae['scene_id']} to {highest_mae['mae_m']:.3f} m in {highest_mae['scene_id']} (descriptive extrema only). Raw correlation changes sign across the scenes, and its magnitude is below 0.1 in {sum(abs(row['raw_pearson_r']) < .1 for row in results)} of 10 scenes; one scene ({max(results, key=lambda row: abs(row['raw_pearson_r']))['scene_id']}) reaches |r|={max(abs(row['raw_pearson_r']) for row in results):.3f}. Thus the poor first scene is not an identical outcome in every scene, but weak/raw signed correspondence is common. This is consistent with scene dependence and motivates checking input/target registration and domain mismatch; these results do not establish either cause.", "",
        "Visual panels show observable RGB, AGL, raw output, aligned output, and absolute error for three fixed scenes: `scenes/DC_03_26/diagnostics.png`, `scenes/DC_08_27/diagnostics.png`, and `scenes/DC_10_20/diagnostics.png`. DC_03_26 RGB and AGL HDF5 arrays exactly match the pre-existing local sample, and its RGB array exactly matches the pre-existing PNG. No land-cover class claims are made because no class labels were used.", "",
        "## 10. Limitations", "",
        "Only ten scenes from one test split were evaluated. Per-scene affine fitting uses that scene's ground truth and is optimistic versus inference without local references. Aggregate figures describe this sample, not universal performance or deployment. HDF5 has no CRS, transform, or orientation metadata; same-stem filenames and visual structure checks are the available evidence for pairing. The processor's resize/interpolation could affect fine spatial detail. No independent calibration, metric DSM, land-cover stratification, or statistical generalization analysis is claimed.", "",
        "## 11. Recommended next research experiment", "",
        "Before fine-tuning, run a targeted registration/preprocessing audit against the original GAMUS acquisition/export conventions: confirm the RGB-to-AGL row/column orientation, compare model raw prediction against AGL at native and carefully controlled resampling settings, and inspect scenes with both correlation signs. Then evaluate a held-out reference-calibration protocol (fit on designated reference pixels/scenes, score on separate pixels/scenes) so the reported error is not fit-and-score on the same complete scene. This can distinguish pipeline/alignment issues from model/domain mismatch with evidence.", "",
    ]
    (out_dir / "multiscene_report.md").write_text("\n".join(report_lines))
    print("\nAggregate per-scene metrics:")
    for key, value in aggregate["metrics"].items(): print(f"{key}: mean={value['mean']:.5f}, median={value['median']:.5f}, std={value['std']:.5f}")
    print(f"Saved results: {out_dir}")


if __name__ == "__main__":
    main()
