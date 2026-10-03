#!/usr/bin/env python3
"""Evaluate frozen Stage 12 checkpoints on predeclared, new GAMUS scenes."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_stage12_experiment import MODEL_ID, make_model, infer_normalized, raw_scores
from src.depth_inference import select_device

DATA = ROOT / "data/stage13_gamus"
OUT = ROOT / "outputs/stage13"
NODATA = -5.0
SCENES = ["DC_10_29", "DC_11_15", "DC_11_19", "DC_11_31", "DC_11_32", "DC_12_15", "DC_12_20", "DC_12_34", "DC_12_35", "DC_13_19"]
SEALED_TEST = ["DC_08_27", "DC_05_28"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pair(scene: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_path = DATA / "images/test" / f"{scene}_RGB.h5"
    agl_path = DATA / "heights/test" / f"{scene}_AGL.h5"
    if not rgb_path.is_file() or not agl_path.is_file():
        raise FileNotFoundError(f"Missing Stage 13 pair for {scene}")
    with h5py.File(rgb_path, "r") as f:
        rgb_raw = f["image"][()]
    with h5py.File(agl_path, "r") as f:
        agl_raw = f["image"][()]
    if rgb_raw.ndim != 3 or rgb_raw.shape[-1] != 3:
        raise ValueError(f"{scene}: invalid RGB shape {rgb_raw.shape}")
    rgb = np.asarray(__import__("PIL.Image", fromlist=["Image"]).fromarray(rgb_raw.astype(np.uint8)))
    if agl_raw.ndim != 2 or agl_raw.shape != rgb.shape[:2]:
        raise ValueError(f"{scene}: RGB/AGL shape mismatch {rgb.shape} vs {agl_raw.shape}")
    agl = agl_raw.astype(np.float32, copy=False)
    valid = np.isfinite(agl) & (agl != NODATA)
    if not valid.any():
        raise ValueError(f"{scene}: no valid AGL pixels")
    return rgb, agl, valid


def score_pair(pred: np.ndarray, target: np.ndarray, valid: np.ndarray, model: str, condition: str, calibration: dict[str, Any]) -> dict[str, Any]:
    if condition == "stage12_validation_calibrated":
        pred = calibration["scale"] * pred + calibration["shift_m"]
    metrics = raw_scores(pred, target, valid)
    return {
        "scene": "", "model": model, "condition": condition,
        "calibration_source": "Stage 12 validation scenes only" if condition != "raw" else "none",
        "calibration_scale": calibration.get("scale") if condition != "raw" else None,
        "calibration_shift_m": calibration.get("shift_m") if condition != "raw" else None,
        **metrics,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for model in ("Depth Anything V2 Small", "Stage 12 Experiment B"):
        output[model] = {}
        for condition in ("raw", "stage12_validation_calibrated"):
            subset = [r for r in rows if r["model"] == model and r["condition"] == condition]
            metrics = {}
            for key in ("mae_m", "rmse_m", "pearson_r", "r2"):
                values = np.asarray([row[key] for row in subset if row[key] is not None], dtype=np.float64)
                metrics[key] = {
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "n_scenes": int(values.size),
                }
            output[model][condition] = metrics
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_plots(per_scene: list[dict[str, Any]]) -> None:
    plot_dir = OUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    calibrated = [r for r in per_scene if r["condition"] == "stage12_validation_calibrated"]
    scenes = SCENES
    x = np.arange(len(scenes))
    width = 0.38
    base = [next(r["mae_m"] for r in calibrated if r["scene"] == s and r["model"] == "Depth Anything V2 Small") for s in scenes]
    adapted = [next(r["mae_m"] for r in calibrated if r["scene"] == s and r["model"] == "Stage 12 Experiment B") for s in scenes]
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    ax.bar(x - width / 2, base, width, label="Depth Anything V2 Small")
    ax.bar(x + width / 2, adapted, width, label="Frozen Stage 12 Experiment B")
    ax.set_ylabel("Validation-calibrated MAE (m)")
    ax.set_xticks(x, scenes, rotation=45, ha="right")
    ax.legend()
    ax.set_title("Stage 13: new-scene per-scene errors")
    fig.savefig(plot_dir / "mae_by_scene.png", dpi=160)
    plt.close(fig)

    base_r = [next(r["pearson_r"] for r in calibrated if r["scene"] == s and r["model"] == "Depth Anything V2 Small") for s in scenes]
    adapt_r = [next(r["pearson_r"] for r in calibrated if r["scene"] == s and r["model"] == "Stage 12 Experiment B") for s in scenes]
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    ax.bar(x - width / 2, base_r, width, label="Depth Anything V2 Small")
    ax.bar(x + width / 2, adapt_r, width, label="Frozen Stage 12 Experiment B")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Pearson r")
    ax.set_xticks(x, scenes, rotation=45, ha="right")
    ax.legend()
    ax.set_title("Stage 13: per-scene height correlation")
    fig.savefig(plot_dir / "pearson_by_scene.png", dpi=160)
    plt.close(fig)


def main() -> None:
    start_time = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads((ROOT / "outputs/stage12/split.json").read_text())
    stage12_scenes = set(split["train"] + split["validation"] + split["test"])
    if len(SCENES) != len(set(SCENES)) or stage12_scenes.intersection(SCENES):
        raise RuntimeError("Stage 13 IDs overlap Stage 12 scenes or include duplicates; refusing evaluation")
    if set(SEALED_TEST) != set(split["test"]):
        raise RuntimeError("Sealed Stage 12 test scene list does not match frozen split.json")
    if any(not (DATA / "images/test" / f"{scene}_RGB.h5").is_file() or not (DATA / "heights/test" / f"{scene}_AGL.h5").is_file() for scene in SCENES):
        raise FileNotFoundError("All predeclared Stage 13 scene pairs must be present")

    stage12_cal = json.loads((ROOT / "outputs/stage12/calibration.json").read_text())
    calibrations = {
        "Depth Anything V2 Small": {"scale": stage12_cal["Depth Anything V2 Small"]["scale"], "shift_m": stage12_cal["Depth Anything V2 Small"]["shift_m"]},
        "Stage 12 Experiment B": {"scale": stage12_cal["Adapted B SmoothL1+edge"]["scale"], "shift_m": stage12_cal["Adapted B SmoothL1+edge"]["shift_m"]},
    }
    checkpoint = ROOT / "outputs/stage12/checkpoints/experiment_B_best.pt"
    checkpoint_hash_before = sha256(checkpoint)
    stage12_config = json.loads((ROOT / "outputs/stage12/training_config.json").read_text())
    target_min = float(stage12_config["train_target_min_m"])
    target_max = float(stage12_config["train_target_max_m"])
    device = select_device()

    baseline, processor = make_model(device)
    adapted, adapted_processor, loaded_min, loaded_max = __import__("scripts.run_stage12_experiment", fromlist=["load_model_state"]).load_model_state(checkpoint, device)
    if not np.isclose(loaded_min, target_min) or not np.isclose(loaded_max, target_max):
        raise RuntimeError("Checkpoint target normalization does not match frozen Stage 12 config")
    rows: list[dict[str, Any]] = []
    pred_dir = OUT / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for index, scene in enumerate(SCENES, 1):
        print(f"Stage 13 [{index}/{len(SCENES)}] {scene}", flush=True)
        rgb, gt, valid = load_pair(scene)
        base_pred = infer_normalized(baseline, processor, rgb, device)
        b_normalized = infer_normalized(adapted, adapted_processor, rgb, device)
        b_pred = loaded_min + b_normalized * (loaded_max - loaded_min)
        if base_pred.shape != gt.shape or b_pred.shape != gt.shape:
            raise ValueError(f"{scene}: prediction does not match AGL grid")
        np.save(pred_dir / f"{scene}_baseline_relative.npy", base_pred.astype(np.float32), allow_pickle=False)
        np.save(pred_dir / f"{scene}_experimentB_height_m.npy", b_pred.astype(np.float32), allow_pickle=False)
        for name, prediction in (("Depth Anything V2 Small", base_pred), ("Stage 12 Experiment B", b_pred)):
            for condition in ("raw", "stage12_validation_calibrated"):
                row = score_pair(prediction, gt, valid, name, condition, calibrations[name])
                row["scene"] = scene
                rows.append(row)

    write_csv(OUT / "stage13_metrics.csv", rows)
    summary = summarize(rows)
    # Aggregate comparisons are unweighted changes in per-scene summary statistics.
    comparison = []
    for condition in ("raw", "stage12_validation_calibrated"):
        for metric in ("mae_m", "rmse_m", "pearson_r", "r2"):
            base = summary["Depth Anything V2 Small"][condition][metric]["mean"]
            exp_b = summary["Stage 12 Experiment B"][condition][metric]["mean"]
            delta = exp_b - base
            comparison.append({
                "condition": condition, "metric": metric,
                "baseline_mean": base, "experiment_b_mean": exp_b,
                "absolute_change_b_minus_baseline": delta,
                "relative_change_percent_using_abs_baseline": (100.0 * delta / abs(base)) if base != 0 else None,
            })
    # Per-scene error changes use validation-only calibration coefficients from Stage 12.
    consistent_counts = {}
    for metric in ("mae_m", "rmse_m"):
        better = 0
        for scene in SCENES:
            base = next(r[metric] for r in rows if r["scene"] == scene and r["model"] == "Depth Anything V2 Small" and r["condition"] == "stage12_validation_calibrated")
            exp_b = next(r[metric] for r in rows if r["scene"] == scene and r["model"] == "Stage 12 Experiment B" and r["condition"] == "stage12_validation_calibrated")
            if exp_b < base:
                better += 1
        consistent_counts[metric] = {"experiment_b_better_scenes": better, "total_scenes": len(SCENES), "improved_on_every_scene": better == len(SCENES)}
    write_csv(OUT / "comparison.csv", comparison)

    per_scene = []
    for scene in SCENES:
        item: dict[str, Any] = {"scene": scene}
        for name, prefix in (("Depth Anything V2 Small", "baseline"), ("Stage 12 Experiment B", "experiment_b")):
            for condition, cprefix in (("raw", "raw"), ("stage12_validation_calibrated", "validation_calibrated")):
                row = next(r for r in rows if r["scene"] == scene and r["model"] == name and r["condition"] == condition)
                for metric in ("mae_m", "rmse_m", "pearson_r", "r2"):
                    item[f"{prefix}_{cprefix}_{metric}"] = row[metric]
        per_scene.append(item)
    write_csv(OUT / "per_scene_metrics.csv", per_scene)
    build_plots(rows)

    elapsed = time.perf_counter() - start_time
    checkpoint_hash_after = sha256(checkpoint)
    if checkpoint_hash_before != checkpoint_hash_after:
        raise RuntimeError("Frozen Stage 12 Experiment B checkpoint changed during Stage 13")
    summary_json = {
        "stage": 13,
        "scene_count": len(SCENES),
        "scenes": SCENES,
        "sealed_stage12_test_scenes_excluded": SEALED_TEST,
        "all_stage12_scenes_excluded": sorted(stage12_scenes),
        "data_source": "public earthflow/GAMUS test split; only ten predeclared new RGB/AGL HDF5 pairs downloaded",
        "downloaded_pair_count": len(SCENES),
        "protocol": {
            "baseline": "cached original Depth Anything V2 Small; same processor, 518 input, bicubic align_corners=False to 1024",
            "experiment_b": "outputs/stage12/checkpoints/experiment_B_best.pt, unchanged; same processor/output resize; inverse Stage 12 training-only target min/max mapping",
            "calibration": "reuse exact Stage 12 validation-only scale/shift coefficients; no Stage 13 or sealed-test pixels used for calibration",
            "mask": "finite AGL and AGL != -5.0; all other values including valid negatives retained",
            "aggregation": "unweighted scene-level mean, median, sample standard deviation (ddof=1)",
        },
        "checkpoint_sha256": checkpoint_hash_after,
        "elapsed_seconds": elapsed,
        "aggregate_metrics": summary,
        "comparison": comparison,
        "per_scene_consistency": consistent_counts,
        "limitations": ["Ten additional scenes only; results do not establish performance beyond these evaluated scenes.", "New scenes are from the same public GAMUS test split, not an independent acquisition/dataset.", "Exact geospatial RGB/AGL registration is not verifiable from HDF5 metadata.", "Relative baseline raw MAE/RMSE against metres are units-mismatched diagnostics; validation-calibrated results provide the like-for-like height comparison."],
    }
    (OUT / "stage13_summary.json").write_text(json.dumps(summary_json, indent=2, allow_nan=False) + "\n")
    report = build_report(summary_json)
    (OUT / "stage13_report.md").write_text(report)
    del baseline, processor, adapted, adapted_processor
    print(json.dumps({"summary": summary, "consistency": consistent_counts, "elapsed_seconds": elapsed}, indent=2), flush=True)


def build_report(data: dict[str, Any]) -> str:
    metrics = data["aggregate_metrics"]
    lines = ["| Model | Condition | Metric | Mean | Median | Std. dev. |", "|---|---|---|---:|---:|---:|"]
    for model, modes in metrics.items():
        for condition, entries in modes.items():
            for metric, values in entries.items():
                lines.append(f"| {model} | {condition} | {metric} | {values['mean']:.4f} | {values['median']:.4f} | {values['std']:.4f} |")
    consistency = data["per_scene_consistency"]
    mae_every = consistency["mae_m"]["improved_on_every_scene"]
    rmse_every = consistency["rmse_m"]["improved_on_every_scene"]
    consistency_text = "Experiment B improved MAE and RMSE on every evaluated scene." if mae_every and rmse_every else (
        f"Consistency check (validation-calibrated): Experiment B had lower MAE on {consistency['mae_m']['experiment_b_better_scenes']}/{data['scene_count']} scenes and lower RMSE on {consistency['rmse_m']['experiment_b_better_scenes']}/{data['scene_count']} scenes. Improvement was not uniform across all scenes."
    )
    comparisons = ["| Condition | Metric | Baseline mean | Experiment B mean | Absolute change (B−baseline) | Relative change |", "|---|---|---:|---:|---:|---:|"]
    for item in data["comparison"]:
        pct = "n/a" if item["relative_change_percent_using_abs_baseline"] is None else f"{item['relative_change_percent_using_abs_baseline']:+.2f}%"
        comparisons.append(f"| {item['condition']} | {item['metric']} | {item['baseline_mean']:.4f} | {item['experiment_b_mean']:.4f} | {item['absolute_change_b_minus_baseline']:+.4f} | {pct} |")
    scenes = ", ".join(data["scenes"])
    limitations = "\n".join(f"- {item}" for item in data["limitations"])
    return f"""# DepthWizard Stage 13 — frozen checkpoint validation

## Purpose and frozen protocol

Stage 13 asks whether the Stage 12 Experiment B result is reproduced on new, scene-disjoint GAMUS test scenes. The baseline and Experiment B checkpoints, inference preprocessing, target inverse mapping, validation-only calibration coefficients, valid mask, and metrics were unchanged. No training, checkpoint selection, calibration fitting, hyperparameter tuning, or model modification occurred.

The Stage 12 sealed test scenes **{', '.join(data['sealed_stage12_test_scenes_excluded'])}** were explicitly excluded. All Stage 12 training and validation scenes were also excluded: {', '.join(data['all_stage12_scenes_excluded'])}.

## Evaluated scenes

Scene count: **{data['scene_count']} new scenes**. IDs: {scenes}.

These were the first ten additional scene IDs shown in the public GAMUS test-folder listing after the previously used IDs, selected before viewing their AGL values or model results. Only ten RGB/AGL pairs were downloaded (roughly 73 MB); the full dataset was not downloaded. [GAMUS public test listing](https://huggingface.co/datasets/earthflow/GAMUS/tree/main/images/test).

## Exact evaluation protocol

- **Baseline:** original cached `{MODEL_ID}` weights.
- **Experiment B:** frozen `outputs/stage12/checkpoints/experiment_B_best.pt`; SHA-256 `{data['checkpoint_sha256']}` (verified unchanged after evaluation).
- Both use official RGB HDF5 conversion, the Stage 12 image processor at 518×518, raw numerical prediction, and bicubic interpolation (`align_corners=False`) to each original 1024×1024 target grid.
- Experiment B predictions are mapped to metres using only the Stage 12 training-scene target min/max. Its scale/shift calibration reuses the exact coefficients fitted from Stage 12 validation scenes. Baseline calibration likewise reuses its Stage 12 validation-only coefficients. No new-scene AGL pixels were used for calibration.
- Raw and Stage-12-validation-calibrated metrics are reported separately. The raw baseline is relative depth compared numerically with metre AGL, so its raw MAE/RMSE are units-mismatched diagnostics. The calibrated comparison is the intended height comparison.
- Valid pixels are finite AGL values unequal to exactly `-5.0`; all other negative AGL values are retained. Per-scene metrics are unweighted in aggregate summaries. Standard deviation is sample standard deviation (`ddof=1`).

## Per-scene results

See `per_scene_metrics.csv` for all raw and validation-calibrated per-scene MAE, RMSE, Pearson r, and R². The long-form source metrics are in `stage13_metrics.csv`.

## Aggregate results

{chr(10).join(lines)}

Absolute and relative changes are Experiment B minus baseline. Relative change uses `100 × change / abs(baseline)`; percentage changes for Pearson r and R² are arithmetic descriptions only, not interpretable performance ratios, especially when baseline values are near zero or negative.

{chr(10).join(comparisons)}

## Improvement consistency

{consistency_text} The aggregate changes and per-scene values are reported without claiming performance beyond these {data['scene_count']} scenes.

## Limitations

{limitations}

## Decision

""" + ("The validation-calibrated Experiment B aggregate has lower MAE and RMSE than baseline on this new subset. As instructed, Stage 14 integration begins using the unchanged frozen checkpoint; this result is limited to the ten scenes reported here.\n" if metrics["Stage 12 Experiment B"]["stage12_validation_calibrated"]["mae_m"]["mean"] < metrics["Depth Anything V2 Small"]["stage12_validation_calibrated"]["mae_m"]["mean"] and metrics["Stage 12 Experiment B"]["stage12_validation_calibrated"]["rmse_m"]["mean"] < metrics["Depth Anything V2 Small"]["stage12_validation_calibrated"]["rmse_m"]["mean"] else "The validation-calibrated Experiment B aggregate does not improve both MAE and RMSE over baseline on this new subset. Stage 14's fine-tuned model is therefore not activated; Stage 14 may still implement the demo pipeline using the original model and clearly labeled calibration options.\n") + f"\nStage 13 runtime: {data['elapsed_seconds']:.2f} seconds. Figures: `plots/mae_by_scene.png` and `plots/pearson_by_scene.png`.\n"


if __name__ == "__main__":
    main()
