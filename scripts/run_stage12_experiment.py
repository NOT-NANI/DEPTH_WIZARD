#!/usr/bin/env python3
"""Run frozen-split, scene-disjoint GAMUS adaptation experiments."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.calibration import fit_scale_shift
from src.depth_inference import MODEL_ID, select_device

DATA = ROOT / "data/multiscene_test"
OUT = ROOT / "outputs/stage12"
SCENES = ["DC_03_26", "DC_05_28", "DC_05_30", "DC_07_21", "DC_07_29", "DC_08_27", "DC_09_18", "DC_09_29", "DC_09_32", "DC_10_20"]
NODATA = -5.0
SEED = 2026
PATCH_SIZE = 224
TRAIN_PATCHES_PER_SCENE_EPOCH = 4
EPOCHS = 10
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 1e-4
WEIGHT_DECAY = 1e-4
EDGE_WEIGHT = 0.1


def load_pair(scene: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_path = DATA / "images/test" / f"{scene}_RGB.h5"
    agl_path = DATA / "heights/test" / f"{scene}_AGL.h5"
    with h5py.File(rgb_path, "r") as rf, h5py.File(agl_path, "r") as af:
        if "image" not in rf or "image" not in af:
            raise KeyError(f"{scene}: HDF5 pair missing /image dataset")
        raw_rgb = rf["image"][()]
        agl = af["image"][()]
    if raw_rgb.ndim != 3 or raw_rgb.shape[-1] != 3:
        raise ValueError(f"{scene}: expected channel-last HxWx3 RGB, got {raw_rgb.shape}")
    rgb = np.asarray(Image.fromarray(raw_rgb.astype(np.uint8)))
    if rgb.dtype != np.uint8:
        raise ValueError(f"{scene}: official-loader RGB conversion did not produce uint8")
    if agl.ndim != 2 or agl.shape != rgb.shape[:2]:
        raise ValueError(f"{scene}: RGB/AGL dimensions differ: {rgb.shape} vs {agl.shape}")
    agl = agl.astype(np.float32, copy=False)
    # Exact sentinel handling: mask only non-finite values and the documented -5 marker.
    valid = np.isfinite(agl) & (agl != NODATA)
    if not valid.any():
        raise ValueError(f"{scene}: no valid AGL targets")
    return rgb, agl, valid


def make_split() -> dict[str, Any]:
    randomized = SCENES.copy()
    random.Random(SEED).shuffle(randomized)
    return {
        "seed": SEED,
        "method": "Python random.Random(seed).shuffle over the fixed Stage 9 ten-scene list; first 6 train, next 2 validation, final 2 test",
        "train": randomized[:6], "validation": randomized[6:8], "test": randomized[8:],
        "all_scenes": SCENES,
        "test_scene_use": "held out from training, validation/model selection, and calibration fitting; used only for final baseline/adapted scoring",
    }


def frozen_config(split: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "seed": SEED,
        "device": str(select_device()),
        "architecture": "DepthAnythingForDepthEstimation Small; freeze backbone; train pretrained neck and depth head; retain relative head ReLU and max_depth=1",
        "input_training": {"method": "random aligned paired crops from source arrays", "crop_size": PATCH_SIZE, "rgb": "uint8 RGB / 255 then ImageNet mean/std normalization", "agl": "same crop coordinates as RGB; valid mask finite and != -5.0"},
        "input_validation_test": {"method": "cached AutoImageProcessor defaults", "model_input": "518x518 for square 1024x1024 RGB", "output": "bicubic interpolation, align_corners=False, to original 1024x1024 grid"},
        "target_normalization": "training-scene valid-target global min/max only: target_norm=(AGL-train_min)/(train_max-train_min); no clipping; valid negative AGL retained; invert with train_min + target_norm*(train_max-train_min)",
        "train_target_min_m": None,
        "train_target_max_m": None,
        "epochs": EPOCHS,
        "train_patches_per_scene_per_epoch": TRAIN_PATCHES_PER_SCENE_EPOCH,
        "total_training_patches_per_experiment": len(split["train"]) * TRAIN_PATCHES_PER_SCENE_EPOCH * EPOCHS,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "optimizer": "AdamW",
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "loss_a": "masked Smooth L1 (beta=0.05) on normalized target",
        "loss_b": "same as A plus 0.1 * masked L1 error between predicted and target horizontal/vertical normalized gradients; adjacent pair valid only when both endpoints valid",
        "validation_selection": "lowest uncalibrated per-scene-mean validation MAE in metres; no test metrics inspected for selection",
        "calibration": "separate unconstrained scale/shift per model fitted only to pooled valid pixels from the two validation scenes; applied unchanged to all pixels in each held-out test scene",
        "split": split,
    }


class PairedPatchDataset(Dataset):
    def __init__(self, scenes: list[str], target_min: float, target_max: float, mean: tuple[float, ...], std: tuple[float, ...]):
        self.scenes = scenes
        self.target_min = target_min
        self.target_range = target_max - target_min
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]
        self.arrays = {scene: load_pair(scene) for scene in scenes}
        self.epoch = 0
        self.coords: list[tuple[str, int, int]] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        rng = np.random.default_rng(SEED + 1009 * epoch)
        self.coords = []
        for scene in self.scenes:
            rgb, _, _ = self.arrays[scene]
            h, w = rgb.shape[:2]
            for _ in range(TRAIN_PATCHES_PER_SCENE_EPOCH):
                y = int(rng.integers(0, h - PATCH_SIZE + 1))
                x = int(rng.integers(0, w - PATCH_SIZE + 1))
                self.coords.append((scene, y, x))

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        scene, y, x = self.coords[index]
        rgb, agl, valid = self.arrays[scene]
        sl = np.s_[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
        rgb_crop = torch.from_numpy(rgb[sl].copy()).permute(2, 0, 1).float() / 255.0
        rgb_crop = (rgb_crop - self.mean) / self.std
        target = (agl[sl].copy() - self.target_min) / self.target_range
        target = torch.from_numpy(target.astype(np.float32, copy=False))
        valid_crop = torch.from_numpy(valid[sl].copy())
        return {"pixel_values": rgb_crop, "target": target, "valid": valid_crop}


def raw_scores(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict[str, float | int | None]:
    mask = valid & np.isfinite(pred) & np.isfinite(gt)
    p, y = pred[mask].astype(np.float64), gt[mask].astype(np.float64)
    if p.size == 0:
        raise ValueError("No valid pixels to score")
    corr = None if p.std() == 0 or y.std() == 0 else float(np.corrcoef(p, y)[0, 1])
    err = p - y
    tss = float(np.sum((y - y.mean()) ** 2))
    return {"valid_pixels": int(mask.sum()), "mae_m": float(np.mean(np.abs(err))), "rmse_m": float(np.sqrt(np.mean(err**2))), "pearson_r": corr, "r2": float(1 - np.sum(err**2) / tss) if tss else None}


def aligned_scores(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, scale: float, shift: float) -> dict[str, float | int | None]:
    return raw_scores(scale * pred + shift, gt, valid)


def set_trainability(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.neck, model.head):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.eval()
    model.neck.train()
    model.head.train()


def make_model(device: torch.device) -> tuple[torch.nn.Module, Any]:
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, local_files_only=True).to(device)
    if model.config.depth_estimation_type != "relative":
        raise ValueError("Expected relative Depth Anything V2 checkpoint")
    model.config.max_depth = 1
    model.head.max_depth = 1
    set_trainability(model)
    return model, processor


def target_train_range(scenes: list[str]) -> tuple[float, float]:
    low, high = float("inf"), float("-inf")
    for scene in scenes:
        _, agl, valid = load_pair(scene)
        values = agl[valid]
        low, high = min(low, float(values.min())), max(high, float(values.max()))
    if not np.isfinite(low + high) or high <= low:
        raise ValueError("Invalid training target range")
    return low, high


def infer_normalized(model: torch.nn.Module, processor: Any, rgb: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    image = Image.fromarray(rgb)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        out = model(**inputs).predicted_depth
        out = F.interpolate(out[:, None], size=rgb.shape[:2], mode="bicubic", align_corners=False).squeeze().float()
    return out.cpu().numpy().astype(np.float32, copy=False)


def validate_model(model: torch.nn.Module, processor: Any, scenes: list[str], device: torch.device, target_min: float, target_range: float) -> dict[str, float]:
    model.eval()
    scores = []
    for scene in scenes:
        rgb, gt, valid = load_pair(scene)
        normalized = infer_normalized(model, processor, rgb, device)
        pred_m = target_min + normalized * target_range
        scores.append(raw_scores(pred_m, gt, valid))
    model.train(); model.neck.train(); model.head.train()
    return {f"{metric}_mean": float(np.mean([s[metric] for s in scores])) for metric in ("mae_m", "rmse_m", "pearson_r", "r2")}


def smooth_l1_masked(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    values = F.smooth_l1_loss(pred, target, beta=0.05, reduction="none")
    mask = valid.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    terms = []
    for dim in (-1, -2):
        p1, p2 = pred.diff(dim=dim), target.diff(dim=dim)
        v1 = valid.narrow(dim, 1, valid.shape[dim] - 1)
        v0 = valid.narrow(dim, 0, valid.shape[dim] - 1)
        pair_valid = v0 & v1
        difference = (p1 - p2).abs()
        mask = pair_valid.to(difference.dtype)
        terms.append((difference * mask).sum() / mask.sum().clamp_min(1))
    return torch.stack(terms).mean()


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_score: float, target_min: float, target_max: float, exp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"experiment": exp, "epoch": epoch, "best_validation_mae_m": best_score, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "target_min_m": target_min, "target_max_m": target_max, "model_id": MODEL_ID, "seed": SEED}, path)


def smoke_test(exp: str, split: dict[str, Any], target_min: float, target_max: float, device: torch.device) -> None:
    torch.manual_seed(SEED)
    model, processor = make_model(device)
    dataset = PairedPatchDataset(split["train"], target_min, target_max, processor.image_mean, processor.image_std)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        x = batch["pixel_values"].to(device)
        y = batch["target"].to(device)
        valid = batch["valid"].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(pixel_values=x).predicted_depth
        loss_a = smooth_l1_masked(pred, y, valid)
        loss = loss_a + (EDGE_WEIGHT * gradient_loss(pred, y, valid) if exp == "B" else 0)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{exp} smoke test: loss is not finite")
        loss.backward()
        optimizer.step()
        save_checkpoint(OUT / "checkpoints" / f"smoke_test_{exp}.pt", model, optimizer, 0, float("inf"), target_min, target_max, exp)
        break
    del model, processor, dataset, optimizer
    if device.type == "mps":
        torch.mps.empty_cache()


def append_history(row: dict[str, Any]) -> None:
    path = OUT / "training_history.csv"
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_experiment(exp: str, split: dict[str, Any], target_min: float, target_max: float, device: torch.device, resume: Path | None = None) -> Path:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model, processor = make_model(device)
    dataset = PairedPatchDataset(split["train"], target_min, target_max, processor.image_mean, processor.image_std)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    best_score = float("inf")
    start_epoch = 1
    checkpoints = OUT / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=False)
        if state["experiment"] != exp:
            raise ValueError("Resume checkpoint experiment type mismatch")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        best_score = float(state["best_validation_mae_m"])
        start_epoch = int(state["epoch"]) + 1
    best_path = checkpoints / f"experiment_{exp}_best.pt"
    start_time = time.perf_counter()
    for epoch in range(start_epoch, EPOCHS + 1):
        dataset.set_epoch(epoch)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        model.train(); model.backbone.eval(); model.neck.train(); model.head.train()
        optimizer.zero_grad(set_to_none=True)
        losses_a, losses_edge = [], []
        num_batches = len(loader)
        for step, batch in enumerate(loader, 1):
            x = batch["pixel_values"].to(device)
            y = batch["target"].to(device)
            valid = batch["valid"].to(device)
            pred = model(pixel_values=x).predicted_depth
            loss_a = smooth_l1_masked(pred, y, valid)
            loss_edge = gradient_loss(pred, y, valid) if exp == "B" else pred.new_zeros(())
            total = loss_a + (EDGE_WEIGHT * loss_edge if exp == "B" else 0)
            if not torch.isfinite(total):
                raise RuntimeError(f"{exp} epoch {epoch}: non-finite loss")
            (total / GRAD_ACCUM).backward()
            losses_a.append(float(loss_a.detach().cpu()))
            losses_edge.append(float(loss_edge.detach().cpu()))
            if step % GRAD_ACCUM == 0 or step == num_batches:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        val = validate_model(model, processor, split["validation"], device, target_min, target_max)
        score = val["mae_m_mean"]
        row = {"experiment": exp, "epoch": epoch, "train_smooth_l1": float(np.mean(losses_a)), "train_gradient_loss": float(np.mean(losses_edge)), "train_total_loss": float(np.mean(losses_a) + (EDGE_WEIGHT * np.mean(losses_edge) if exp == "B" else 0)), **{f"validation_{k}": v for k, v in val.items()}, "epoch_elapsed_seconds": time.perf_counter() - start_time, "selected_best": score < best_score}
        append_history(row)
        current_path = checkpoints / f"experiment_{exp}_last.pt"
        save_checkpoint(current_path, model, optimizer, epoch, min(best_score, score), target_min, target_max, exp)
        if score < best_score:
            best_score = score
            save_checkpoint(best_path, model, optimizer, epoch, best_score, target_min, target_max, exp)
        print(f"Experiment {exp} epoch {epoch}/{EPOCHS}: train={row['train_total_loss']:.6f}, val_mae={score:.4f}m, best={best_score:.4f}m", flush=True)
    elapsed = time.perf_counter() - start_time
    (OUT / "checkpoints" / f"experiment_{exp}_training_time_seconds.txt").write_text(f"{elapsed:.6f}\n")
    del model, processor, dataset, optimizer
    if device.type == "mps":
        torch.mps.empty_cache()
    return best_path


def load_model_state(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, Any, float, float]:
    model, processor = make_model(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, processor, float(state["target_min_m"]), float(state["target_max_m"])


def compute_predictions(model: torch.nn.Module, processor: Any, scenes: list[str], device: torch.device, decode: tuple[float, float] | None = None) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    result = {}
    for scene in scenes:
        rgb, gt, valid = load_pair(scene)
        pred = infer_normalized(model, processor, rgb, device)
        if decode is not None:
            target_min, target_max = decode
            pred = target_min + pred * (target_max - target_min)
        result[scene] = (pred, gt, valid)
    return result


def validation_calibration(predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[float, float, int]:
    pred_values, gt_values = [], []
    for pred, gt, valid in predictions.values():
        mask = valid & np.isfinite(pred) & np.isfinite(gt)
        pred_values.append(pred[mask].astype(np.float32, copy=False))
        gt_values.append(gt[mask].astype(np.float32, copy=False))
    p, g = np.concatenate(pred_values), np.concatenate(gt_values)
    fit = fit_scale_shift(p, g, np.ones(p.shape, dtype=bool))
    return fit.scale, fit.shift, int(p.size)


def metric_rows(model_name: str, predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], calibration: tuple[float, float], condition: str) -> list[dict[str, Any]]:
    scale, shift = calibration
    rows = []
    for scene, (pred, gt, valid) in predictions.items():
        metrics = raw_scores(pred if condition == "raw" else scale * pred + shift, gt, valid)
        rows.append({"model": model_name, "condition": condition, "scene": scene, "calibration_source": "none" if condition == "raw" else "validation scenes only", "calibration_scale": None if condition == "raw" else scale, "calibration_shift_m": None if condition == "raw" else shift, **metrics})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for model in sorted({r["model"] for r in rows}):
        for condition in ("raw", "validation_calibrated"):
            subset = [r for r in rows if r["model"] == model and r["condition"] == condition]
            if not subset:
                continue
            output.append({"model": model, "condition": condition, "scenes": len(subset), **{key: float(np.mean([r[key] for r in subset])) for key in ("mae_m", "rmse_m", "pearson_r", "r2")}})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-experiment", choices=("A", "B"))
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    split = make_split()
    split_path = OUT / "split.json"
    if split_path.exists():
        existing = json.loads(split_path.read_text())
        if existing != split:
            raise RuntimeError("Frozen split differs from the existing split.json; refusing to change test protocol")
    else:
        split_path.write_text(json.dumps(split, indent=2) + "\n")
    target_min, target_max = target_train_range(split["train"])
    config = frozen_config(split)
    config["train_target_min_m"] = target_min
    config["train_target_max_m"] = target_max
    config_path = OUT / "training_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous != config:
            raise RuntimeError("Training configuration differs from existing frozen config; refusing silent protocol change")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    device = select_device()
    print(f"Frozen scene split: train={split['train']}; validation={split['validation']}; test={split['test']}", flush=True)
    print(f"Training target range: {target_min:.6f} to {target_max:.6f} m; device={device}", flush=True)
    for exp in ("A", "B"):
        smoke_test(exp, split, target_min, target_max, device)
    resume = args.resume_checkpoint
    best_paths = {}
    for exp in ("A", "B"):
        best_paths[exp] = train_experiment(exp, split, target_min, target_max, device, resume if args.resume_experiment == exp else None)
    # Test scenes are first loaded only after both training and validation-only checkpoint selection finish.
    pretrained, processor = make_model(device)
    base_val = compute_predictions(pretrained, processor, split["validation"], device)
    base_test = compute_predictions(pretrained, processor, split["test"], device)
    baseline_fit = validation_calibration(base_val)
    baseline_rows = metric_rows("Depth Anything V2 Small", base_test, baseline_fit[:2], "raw") + metric_rows("Depth Anything V2 Small", base_test, baseline_fit[:2], "validation_calibrated")
    write_csv(OUT / "baseline_test_metrics.csv", baseline_rows)
    np.savez_compressed(OUT / "checkpoints" / "baseline_validation_calibration.npz", scale=baseline_fit[0], shift=baseline_fit[1], pixels=baseline_fit[2])
    all_rows = list(baseline_rows)
    selected_val = {"Depth Anything V2 Small": {"scale": baseline_fit[0], "shift_m": baseline_fit[1], "calibration_pixels": baseline_fit[2], "scene_metrics": {s: raw_scores(*base_val[s]) for s in split["validation"]}}}
    for exp in ("A", "B"):
        model, proc, norm_min, norm_max = load_model_state(best_paths[exp], device)
        val = compute_predictions(model, proc, split["validation"], device, decode=(norm_min, norm_max))
        test = compute_predictions(model, proc, split["test"], device, decode=(norm_min, norm_max))
        fit = validation_calibration(val)
        name = "Adapted A SmoothL1" if exp == "A" else "Adapted B SmoothL1+edge"
        rows = metric_rows(name, test, fit[:2], "raw") + metric_rows(name, test, fit[:2], "validation_calibrated")
        write_csv(OUT / f"experiment_{exp}_test_metrics.csv", rows)
        all_rows.extend(rows)
        selected_val[name] = {"best_checkpoint": str(best_paths[exp].relative_to(ROOT)), "target_min_m": norm_min, "target_max_m": norm_max, "scale": fit[0], "shift_m": fit[1], "calibration_pixels": fit[2], "scene_metrics": {s: raw_scores(*val[s]) for s in split["validation"]}}
        pred_dir = OUT / "test_predictions" / f"experiment_{exp}"
        pred_dir.mkdir(parents=True, exist_ok=True)
        for scene, (pred, _, _) in test.items():
            np.save(pred_dir / f"{scene}_prediction_m.npy", pred.astype(np.float32), allow_pickle=False)
        del model, proc
    write_csv(OUT / "experiment_comparison.csv", summarize(all_rows))
    write_csv(OUT / "adapted_test_metrics.csv", [row for row in all_rows if row["model"].startswith("Adapted")])
    (OUT / "calibration.json").write_text(json.dumps(selected_val, indent=2) + "\n")
    report = build_report(split, config, best_paths, selected_val, all_rows)
    (OUT / "stage12_report.md").write_text(report)
    del pretrained, processor
    print("Stage 12 complete", flush=True)


def build_report(split: dict[str, Any], config: dict[str, Any], best_paths: dict[str, Path], calibration_data: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    summaries = summarize(rows)
    summary_lookup = {(row["model"], row["condition"]): row for row in summaries}
    models = ["Depth Anything V2 Small", "Adapted A SmoothL1", "Adapted B SmoothL1+edge"]
    table = ["| Model | Condition | Test MAE (m) | RMSE (m) | Pearson r | R² |", "|---|---|---:|---:|---:|---:|"]
    per_scene = ["| Model | Condition | Scene | MAE (m) | RMSE (m) | Pearson r | R² |", "|---|---|---|---:|---:|---:|---:|"]
    for model in models:
        for condition in ("raw", "validation_calibrated"):
            item = summary_lookup.get((model, condition))
            if item:
                table.append(f"| {model} | {condition} | {item['mae_m']:.4f} | {item['rmse_m']:.4f} | {item['pearson_r']:.4f} | {item['r2']:.4f} |")
    for row in rows:
        per_scene.append(f"| {row['model']} | {row['condition']} | {row['scene']} | {row['mae_m']:.4f} | {row['rmse_m']:.4f} | {row['pearson_r']:.4f} | {row['r2']:.4f} |")
    elapsed = {}
    for exp in ("A", "B"):
        path = OUT / "checkpoints" / f"experiment_{exp}_training_time_seconds.txt"
        elapsed[exp] = float(path.read_text()) if path.exists() else None
    comparisons = []
    for condition in ("raw", "validation_calibrated"):
        base = summary_lookup.get((models[0], condition))
        for model in models[1:]:
            adapted = summary_lookup.get((model, condition))
            if base and adapted:
                comparisons.append(f"- {model} minus baseline ({condition}): MAE {adapted['mae_m']-base['mae_m']:+.4f} m; RMSE {adapted['rmse_m']-base['rmse_m']:+.4f} m; r {adapted['pearson_r']-base['pearson_r']:+.4f}; R² {adapted['r2']-base['r2']:+.4f}.")
    return f"""# DepthWizard Stage 12 report

## 1. Research question

Does adapting Depth Anything V2 Small's pretrained depth head and neck to normalized GAMUS AGL improve results on scenes excluded from training, validation/model selection, and calibration?

## 2. Dataset split

The split was frozen before training with Python `random.Random(2026).shuffle` over the fixed ten-scene list. Counts: {len(split['train'])} train, {len(split['validation'])} validation, {len(split['test'])} test. Exact IDs and policy are in `split.json`.

- Train: {', '.join(split['train'])}
- Validation: {', '.join(split['validation'])}
- Test: {', '.join(split['test'])}

Test scenes were not loaded until both experiments and checkpoint selection were complete. They were not used for training, validation, calibration fitting, or hyperparameter selection.

## 3. Model architecture and training configuration

Cached checkpoint: `{MODEL_ID}`. The backbone was frozen; pretrained neck and depth head were trainable. The relative head's ReLU was retained and its max-depth multiplier set to 1 so the head predicts a nonnegative normalized height representation. The pretrained checkpoint was not overwritten.

CPU training used paired 224×224 random crops, batch size {config['batch_size']}, gradient accumulation {config['gradient_accumulation_steps']}, {config['epochs']} epochs, AdamW at {config['learning_rate']}, weight decay {config['weight_decay']}, and {config['train_patches_per_scene_per_epoch']} crops per training scene per epoch. RGB received [0,1] conversion and the checkpoint's ImageNet mean/std. Validation/test RGB used the standard processor at 518×518; predictions were bicubically resized to the original 1024×1024 grid.

## 4. Target representation and losses

Valid AGL values were normalized with the min/max computed from training scenes only: `[min,max]=[{config['train_target_min_m']:.6f}, {config['train_target_max_m']:.6f}] m`; `normalized=(AGL-min)/(max-min)`. NoData was exactly `-5.0` plus nonfinite values; other negative AGL values remained valid. Predictions were inverted with the same training-only values for direct metre-scale scoring.

- Experiment A: masked Smooth L1, beta 0.05.
- Experiment B: identical Smooth L1 plus 0.1×masked L1 difference between predicted and target horizontal/vertical gradients. A gradient pair contributed only when both target pixels were valid.

Each run first completed a one-batch forward/backward/optimizer/checkpoint smoke test. Smoke checkpoints are saved under `checkpoints/`.

## 5. Training and validation behavior

Training/validation losses and metrics for every epoch are in `training_history.csv`. Best checkpoints were selected solely by lowest mean per-scene uncalibrated validation MAE. Training elapsed time: A {elapsed['A']:.2f} s; B {elapsed['B']:.2f} s. Best checkpoint files: `{best_paths['A'].relative_to(ROOT)}` and `{best_paths['B'].relative_to(ROOT)}`.

Validation-only calibration fits used {calibration_data['Depth Anything V2 Small']['calibration_pixels']:,} pooled valid pixels from the two validation scenes for the baseline; adapted model fits used their corresponding validation predictions and the same scene pixels. Model-specific scale/shift coefficients are saved in `calibration.json`. No test pixels contributed to calibration.

## 6. Held-out test results

Raw means direct, uncalibrated output against AGL. For the baseline, this is relative output compared numerically to metres and is a units-mismatched diagnostic. Adapted output is inverted from the training-only normalized height representation into metres. Validation-calibrated applies one model-specific affine mapping fitted only on validation scenes, unchanged to test scenes.

{chr(10).join(table)}

## 7. Per-scene results

{chr(10).join(per_scene)}

## 8. Baseline versus adapted

Positive deltas are worse; negative MAE/RMSE deltas are lower errors. These are descriptive differences on the two held-out scenes, not a significance claim.

{chr(10).join(comparisons)}

## 9. Limitations

Only ten scenes were available, leaving six training, two validation, and two test scenes. Test-scene count is too small for broad generalization claims. HDF5 pairs have matching dimensions and documented intended co-registration, but exact geospatial registration cannot be independently verified from file metadata. Training used random 224-pixel crops while full-image evaluation used processor input 518; this context/resolution difference may affect results. Validation calibration is a reference-scene protocol, not a deployable calibration unless such reference scenes are available.

## 10. What to try next

On these two held-out scenes, Experiment B had lower mean validation-calibrated MAE/RMSE than the baseline (see measured deltas above), while Experiment A was nearly unchanged. Treat this as a preliminary result from a two-scene test, not evidence of robust generalization. Preserve the current test split and do not tune against it. The next experiment should replicate the frozen B configuration on a larger, newly held-out scene set; keep the current test scenes sealed from all future selection and calibration. If obtaining more scenes is not practical, report the sample-size limitation instead of adding another loss or drawing a broad improvement claim.
"""


if __name__ == "__main__":
    main()
