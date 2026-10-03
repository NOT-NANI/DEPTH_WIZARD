"""Depth Anything V2 inference with a lossless float32 prediction output."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def load_rgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {path}")
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im.load()
            return im.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Unsupported or corrupted image {path}: {exc}") from exc


def select_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_inference(input_path: Path, output_dir: Path, device: torch.device | None = None) -> tuple[Path, Path, float]:
    image = load_rgb(input_path)
    device = device or select_device()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    raw_path = output_dir / f"{stem}_depth_raw.npy"
    visual_path = output_dir / f"{stem}_depth.png"
    print(f"Device: {device}")
    print(f"Input: {input_path.resolve()}")
    print(f"Image dimensions: {image.width} x {image.height}")
    print(f"Model: {MODEL_ID}")
    started = time.perf_counter()
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()
    except Exception as exc:
        raise RuntimeError(f"Could not load model {MODEL_ID}; check Hugging Face/network access: {exc}") from exc
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        predicted = model(**inputs).predicted_depth
        predicted = F.interpolate(predicted.unsqueeze(1), size=(image.height, image.width), mode="bicubic", align_corners=False).squeeze().float()
    prediction = predicted.detach().cpu().numpy().astype(np.float32, copy=False)
    if prediction.shape != (image.height, image.width) or not np.isfinite(prediction).all():
        raise RuntimeError(f"Model returned invalid prediction array: shape={prediction.shape}")
    elapsed = time.perf_counter() - started
    np.save(raw_path, prediction, allow_pickle=False)
    # Visualization only: robust percentile stretch; numeric evaluation uses the .npy.
    lo, hi = np.percentile(prediction, [2, 98])
    shown = np.zeros_like(prediction, dtype=np.uint8) if hi <= lo else np.clip((prediction - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(shown, mode="L").save(visual_path)
    del model, processor, inputs, predicted
    if device.type == "mps":
        torch.mps.empty_cache()
    print("Inference complete")
    print(f"Raw prediction: {raw_path.resolve()}")
    print(f"Depth visualization: {visual_path.resolve()}")
    print(f"Inference time: {elapsed:.2f} seconds")
    return raw_path, visual_path, elapsed
