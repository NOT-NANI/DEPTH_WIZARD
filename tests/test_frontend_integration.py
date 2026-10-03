"""Verification of frontend static assets, syntax, and backend API integration."""
import json
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"


class FrontendIntegrationTest(unittest.TestCase):
    def test_frontend_structure(self):
        self.assertTrue((FRONTEND / "index.html").is_file())
        self.assertTrue((FRONTEND / "config.js").is_file())
        self.assertTrue((FRONTEND / "css/styles.css").is_file())
        self.assertTrue((FRONTEND / "js/api.js").is_file())
        self.assertTrue((FRONTEND / "js/app.js").is_file())
        self.assertTrue((FRONTEND / "js/colormaps.js").is_file())
        self.assertTrue((FRONTEND / "js/labels.js").is_file())
        self.assertTrue((FRONTEND / "js/mapview.js").is_file())
        self.assertTrue((FRONTEND / "js/terrain3d.js").is_file())
        self.assertTrue((FRONTEND / "vendor/three/three.module.js").is_file())
        self.assertTrue((FRONTEND / "vendor/three/OrbitControls.js").is_file())
        self.assertTrue((FRONTEND / "README.md").is_file())

    def test_frontend_html_and_config(self):
        html = (FRONTEND / "index.html").read_text()
        self.assertIn("DepthWizard", html)
        self.assertIn('type="importmap"', html)
        self.assertIn('src="js/app.js"', html)
        self.assertIn('id="step-image"', html)
        self.assertIn('id="step-calibration"', html)
        self.assertIn('id="step-process"', html)
        self.assertIn('id="results-stage"', html)
        self.assertIn('id="step-export"', html)

        config = (FRONTEND / "config.js").read_text()
        self.assertIn("API_BASE_URL", config)
        self.assertIn("http://127.0.0.1:8765", config)

    def test_backend_live_interaction(self):
        # Health check
        with urllib.request.urlopen("http://127.0.0.1:8765/api/health") as resp:
            self.assertEqual(resp.status, 200)
            health = json.loads(resp.read().decode())
            self.assertEqual(health["status"], "ok")
            self.assertTrue(health["model_checkpoint_available"])

        # Run query
        run_id = "3cb59ee67137"
        with urllib.request.urlopen(f"http://127.0.0.1:8765/api/runs/{run_id}") as resp:
            self.assertEqual(resp.status, 200)
            meta = json.loads(resp.read().decode())
            self.assertEqual(meta["run_id"], run_id)
            self.assertIn("exports", meta)

        # Sample query
        with urllib.request.urlopen(f"http://127.0.0.1:8765/api/runs/{run_id}/sample?x=100&y=100") as resp:
            self.assertEqual(resp.status, 200)
            sample = json.loads(resp.read().decode())
            self.assertIn("relative_depth", sample)
            self.assertEqual(sample["relative_depth_units"], "unitless")
            self.assertIn("height", sample)

        # Surface query
        req = urllib.request.Request(f"http://127.0.0.1:8765/api/runs/{run_id}/surface?kind=height&max_size=64")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = resp.read()
            w = int(resp.headers.get("X-Surface-Width"))
            h = int(resp.headers.get("X-Surface-Height"))
            self.assertEqual(len(data), w * h * 4)


if __name__ == "__main__":
    unittest.main()
