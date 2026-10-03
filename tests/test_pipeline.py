import tempfile
import unittest
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.calibration import fit_scale_shift
from src.evaluation import evaluate
from src.gamus import load_gamus, load_image_h5
from src.depth_inference import load_rgb
from src.stage10_audit import fit_on_calibration_region, spatial_half_masks, transformed_predictions
from scripts.run_stage11_audit import load_official_arrays, resize_prediction
from scripts.run_stage12_experiment import load_pair, make_split
from src.stage14_pipeline import export_obj, load_rgb as load_stage14_rgb, slope_degrees
from src import stage14_pipeline
from scripts import run_demo
import zipfile
import torch

ROOT = Path(__file__).resolve().parents[1]

class PipelineTests(unittest.TestCase):
    def test_load_rgb_and_bad_path(self):
        image = load_rgb(ROOT / "gamus_rgb.png")
        self.assertEqual(image.size, (1024, 1024))
        with self.assertRaises(FileNotFoundError): load_rgb(ROOT / "does-not-exist.png")

    def test_gamus_shapes_and_nodata(self):
        rgb, gt, valid = load_gamus(ROOT / "gamus_sample.h5", ROOT / "gamus_height.h5")
        self.assertEqual(rgb.shape, (1024, 1024, 3))
        self.assertEqual(gt.shape, (1024, 1024))
        self.assertEqual(int(valid.sum()), 1_043_613)
        self.assertTrue(np.any(gt[valid] < 0))
        with self.assertRaises(ValueError): load_image_h5(ROOT / "gamus_height.h5", 3)

    def test_alignment_and_metrics(self):
        raw = np.arange(12, dtype=np.float32).reshape(3, 4)
        gt = 2.5 * raw - 3
        valid = np.ones_like(raw, dtype=bool)
        valid[0, 0] = False
        fit = fit_scale_shift(raw, gt, valid)
        self.assertAlmostEqual(fit.scale, 2.5)
        self.assertAlmostEqual(fit.shift, -3)
        _, _, metrics = evaluate(raw, gt, valid)
        self.assertAlmostEqual(metrics["rmse_m"], 0, places=5)

    def test_stage10_spatial_holdout_and_transformations(self):
        shape = (6, 8)
        top, bottom = spatial_half_masks(shape, "row")
        left, right = spatial_half_masks(shape, "col")
        self.assertFalse(np.any(top & bottom))
        self.assertTrue(np.all(top | bottom))
        self.assertFalse(np.any(left & right))
        self.assertTrue(np.all(left | right))
        raw = np.arange(48, dtype=np.float32).reshape(shape)
        gt = 3 * raw + 2
        valid = np.ones(shape, dtype=bool)
        fit, metrics, count = fit_on_calibration_region(raw, gt, valid, top, bottom)
        self.assertAlmostEqual(fit.scale, 3)
        self.assertAlmostEqual(fit.shift, 2)
        self.assertEqual(count, 24)
        self.assertAlmostEqual(metrics["rmse_m"], 0, places=5)
        variants = transformed_predictions(raw)
        self.assertEqual(set(variants), {"original", "horizontal_flip", "vertical_flip", "rotate_180"})
        for array in variants.values(): self.assertEqual(array.shape, shape)

    def test_stage10_rejects_overlapping_calibration_masks(self):
        raw = np.arange(16, dtype=np.float32).reshape(4, 4)
        gt = raw * 2
        valid = np.ones((4, 4), dtype=bool)
        with self.assertRaises(ValueError):
            fit_on_calibration_region(raw, gt, valid, valid, valid)

    def test_stage11_resize_modes_preserve_target_dimensions(self):
        raw = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        for mode in ("bicubic", "bilinear", "nearest"):
            output = resize_prediction(raw, (7, 9), mode)
            self.assertEqual(output.shape, (7, 9))
            self.assertTrue(np.isfinite(output).all())

    def test_stage11_loader_reproduction_is_untransformed(self):
        rgb, agl, metadata = load_official_arrays("DC_03_26")
        self.assertEqual(rgb.shape, (1024, 1024, 3))
        self.assertEqual(agl.shape, (1024, 1024))
        self.assertTrue(metadata["rgb_exact_array_match"])
        self.assertTrue(metadata["agl_exact_array_match"])

    def test_stage12_split_is_reproducible_and_scene_disjoint(self):
        split = make_split()
        self.assertEqual(split["test"], ["DC_08_27", "DC_05_28"])
        self.assertEqual(len(split["train"]), 6)
        self.assertEqual(len(split["validation"]), 2)
        self.assertEqual(len(split["test"]), 2)
        self.assertFalse(set(split["train"]) & set(split["validation"]))
        self.assertFalse(set(split["train"]) & set(split["test"]))
        self.assertFalse(set(split["validation"]) & set(split["test"]))

    def test_stage12_pair_keeps_valid_negative_heights(self):
        rgb, agl, valid = load_pair("DC_03_26")
        self.assertEqual(rgb.shape, (1024, 1024, 3))
        self.assertEqual(agl.shape, valid.shape)
        self.assertTrue(np.any(agl[valid] < 0))
        self.assertFalse(np.any(valid & (agl == -5.0)))

    def test_stage13_scenes_exclude_all_stage12_splits(self):
        stage12 = json.loads((ROOT / "outputs/stage12/split.json").read_text())
        stage13 = json.loads((ROOT / "outputs/stage13/stage13_summary.json").read_text())
        prior = set(stage12["train"] + stage12["validation"] + stage12["test"])
        new_scenes = set(stage13["scenes"])
        self.assertEqual(stage13["scene_count"], 10)
        self.assertFalse(prior & new_scenes)
        self.assertEqual(set(stage13["sealed_stage12_test_scenes_excluded"]), set(stage12["test"]))

    def test_stage13_metrics_cover_all_scenes_and_models(self):
        path = ROOT / "outputs/stage13/stage13_metrics.csv"
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 40)
        self.assertEqual(len({row["scene"] for row in rows}), 10)
        self.assertEqual({row["condition"] for row in rows}, {"raw", "stage12_validation_calibrated"})
        self.assertTrue(all(row["mae_m"] and row["rmse_m"] for row in rows))

    def test_stage14_plain_image_has_no_georeference(self):
        rgb, grid = load_stage14_rgb(ROOT / "test.jpg")
        self.assertEqual(rgb.ndim, 3)
        self.assertEqual(rgb.shape[-1], 3)
        self.assertFalse(grid.georeferenced)

    def test_stage14_slope_and_textured_mesh_exports(self):
        height = np.tile(np.arange(8, dtype=np.float32), (8, 1))
        slope = slope_degrees(height, 1.0)
        self.assertTrue(np.all(np.isfinite(slope)))
        self.assertTrue(np.allclose(slope, 45.0, atol=1e-4))
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb[..., 1] = 180
        with tempfile.TemporaryDirectory() as tmp:
            mesh = Path(tmp) / "mesh.zip"
            export_obj(rgb, height, mesh, "estimated metres", max_size=8)
            with zipfile.ZipFile(mesh) as archive:
                self.assertTrue({"terrain.obj", "terrain.mtl", "rgb_texture.png"}.issubset(set(archive.namelist())))

    def test_stage14_process_exports_relative_height_mesh_and_point_sample(self):
        rgb = np.zeros((8, 10, 3), dtype=np.uint8)
        rgb[..., 0] = 120
        relative = np.linspace(0, 1, 80, dtype=np.float32).reshape(8, 10)
        height = (relative * 20 - 3).astype(np.float32)
        info = {"mode": "stage12_validation_calibrated", "units": "m (estimated)", "label": "estimated metres"}
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            source = temp / "source.png"
            Image.fromarray(rgb).save(source)
            run_dir = temp / "runs" / "012345abcdef"
            with unittest.mock.patch.object(stage14_pipeline, "run_frozen_model", return_value=(height, info, relative)):
                metadata = stage14_pipeline.process_image(source, run_dir)
            self.assertEqual(metadata["relative_depth_output"]["units"], "unitless")
            for name in ("relative_depth.npy", "relative_depth.png", "height_map.npy", "height_map.png", "terrain_viewer.html", "terrain_mesh.zip", "result_bundle.zip"):
                self.assertTrue((run_dir / name).is_file(), name)
            self.assertTrue(np.allclose(np.load(run_dir / "relative_depth.npy"), relative))
            (run_dir / "metadata.json").write_text(json.dumps({"height_output": {"units": "m (estimated)"}}))
            with unittest.mock.patch.object(run_demo, "RUNS", temp / "runs"):
                sample = run_demo.run_sample("012345abcdef", 4, 3)
                self.assertAlmostEqual(sample["relative_depth"], float(relative[3, 4]))
                self.assertAlmostEqual(sample["height"], float(height[3, 4]))
                surface_bytes, surface_info = run_demo.run_surface("012345abcdef", "height", max_size=5)
                surface = np.frombuffer(surface_bytes, dtype="<f4").reshape(surface_info["height"], surface_info["width"])
                self.assertLessEqual(max(surface.shape), 5)
                self.assertEqual(surface_info["format"], "float32-le-row-major")
                self.assertEqual(surface_info["units"], "m (estimated)")
                self.assertTrue(np.isfinite(surface).all())
                with self.assertRaises(ValueError):
                    run_demo.run_sample("012345abcdef", 10, 3)
                with self.assertRaises(ValueError):
                    run_demo.run_surface("012345abcdef", "arbitrary")

if __name__ == "__main__": unittest.main()
