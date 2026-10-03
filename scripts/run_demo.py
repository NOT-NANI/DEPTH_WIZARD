#!/usr/bin/env python3
"""Start the local DepthWizard Stage 14 upload-and-preview demo."""
from __future__ import annotations

import json
import mimetypes
import re
import shutil
import sys
import uuid
import zipfile
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "outputs/final/runs"
PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DepthWizard</title><style>
:root{font-family:Inter,system-ui,-apple-system,sans-serif;color:#15222f;background:#f3f6f8}body{margin:0}header{background:#142c40;color:#fff;padding:24px max(24px,calc((100vw - 1100px)/2))}header h1{margin:0 0 6px}header p{margin:0;color:#c9d8e4}.wrap{max-width:1100px;margin:24px auto;padding:0 20px}.card{background:white;border:1px solid #dce4ea;border-radius:12px;padding:22px;margin-bottom:18px;box-shadow:0 2px 10px #142c400a}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}label{display:block;font-weight:600;margin:12px 0 6px}input,select,button{font:inherit}input[type=file],input[type=number],select{width:100%;box-sizing:border-box;padding:10px;border:1px solid #bbc8d1;border-radius:7px;background:white}input[type=checkbox]{margin-right:6px}small,.muted{color:#596c7a;line-height:1.5}button{background:#137c72;color:white;border:0;border-radius:7px;padding:12px 18px;font-weight:700;cursor:pointer}button:disabled{opacity:.5}#status{margin-left:12px;font-weight:600}#result{display:none}#links a{display:inline-block;margin:6px 10px 6px 0;color:#096b91}.preview{max-width:100%;max-height:500px;object-fit:contain;border:1px solid #ddd;border-radius:8px}.viewer{width:100%;height:650px;border:1px solid #ccd6dc;border-radius:8px}.notice{border-left:4px solid #d69d25;padding:10px 14px;background:#fff8e6}.hidden{display:none}
</style></head><body><header><h1>DepthWizard</h1><p>Single-image relative depth and calibrated height visualization</p></header><main class="wrap"><section class="card"><h2>Build a terrain preview</h2><p class="notice"><b>Research prototype:</b> the default Stage 12 validation calibration produces an estimated height map. It is not surveyed absolute elevation. PNG/JPG inputs have no CRS or map transform, and none will be invented.</p><form id="form"><div class="grid"><div><label for="image">RGB image (PNG/JPG, or GeoTIFF with rasterio installed)</label><input id="image" name="image" type="file" accept=".png,.jpg,.jpeg,.tif,.tiff" required></div><div><label for="dem">Optional ground DEM (GeoTIFF)</label><input id="dem" name="dem" type="file" accept=".tif,.tiff"><small>When georeferenced, it is reprojected to a georeferenced RGB grid. With ordinary images, dimensions must match and you must confirm pixel alignment.</small></div></div><label for="calibration">Height interpretation</label><select id="calibration" name="calibration"><option value="stage12_validation">Stage 12 validation-calibrated estimate (m)</option><option value="custom">Custom scale/shift</option><option value="relative">Unmapped model output (unitless)</option></select><div id="custom" class="grid hidden"><div><label>Scale</label><input name="scale" type="number" step="any" value="1"></div><div><label>Shift</label><input name="shift" type="number" step="any" value="0"><small>Units follow your calibration reference.</small></div></div><label id="custom-units-label" class="hidden"><input name="custom_units_m" type="checkbox" value="true">Custom calibration output is in metres</label><label><input name="alignment_confirmed" type="checkbox" value="true">For non-georeferenced imagery, I confirm the optional DEM is pixel-aligned and same-sized.</label><label for="pixel_size_m">Pixel size in metres (optional, for slope in degrees)</label><input id="pixel_size_m" name="pixel_size_m" type="number" min="0" step="any" placeholder="Use projected GeoTIFF spacing when available"><p><button id="submit" type="submit">Generate height map and 3D preview</button><span id="status"></span></p><small>Inference uses the frozen Stage 12 Experiment B checkpoint. The original checkpoint and all Stage 1–13 results remain unchanged.</small></form></section><section id="result" class="card"><h2>Result</h2><p id="summary"></p><div id="links"></div><div class="grid"><div><h3>RGB input</h3><img id="rgbPreview" class="preview"></div><div><h3>Height map</h3><img id="heightPreview" class="preview"></div></div><h3>Interactive terrain</h3><p class="muted">Drag to orbit, scroll to zoom, use W/A/S/D to fly, click a point to inspect its height and slope. Three.js is loaded from a CDN, so the 3D panel needs browser internet access.</p><iframe id="viewer" class="viewer"></iframe><pre id="metadata"></pre></section></main><script>
const form=document.querySelector('#form'),mode=document.querySelector('#calibration'),custom=document.querySelector('#custom');mode.onchange=()=>{custom.classList.toggle('hidden',mode.value!=='custom');document.querySelector('#custom-units-label').classList.toggle('hidden',mode.value!=='custom')};form.onsubmit=async e=>{e.preventDefault();const button=document.querySelector('#submit'),status=document.querySelector('#status');button.disabled=true;status.textContent='Running frozen model…';try{const data=new FormData(form);const response=await fetch('/api/process',{method:'POST',body:data});const result=await response.json();if(!response.ok)throw new Error(result.error||'Processing failed');document.querySelector('#result').style.display='block';document.querySelector('#summary').textContent=result.height_product_label+' · '+result.input_shape.join(' × ')+' · georeferenced: '+result.input_georeferenced;const root='/files/'+result.run_id+'/';document.querySelector('#heightPreview').src=root+'height_map.png';document.querySelector('#rgbPreview').src=root+'input_preview.png';document.querySelector('#viewer').src=root+'terrain_viewer.html';document.querySelector('#metadata').textContent=JSON.stringify(result,null,2);const links=document.querySelector('#links');links.innerHTML='';for(const [label,file] of Object.entries(result.exports)){if(file){const a=document.createElement('a');a.href=root+file;a.download=file;a.textContent='Download '+label.replaceAll('_',' ');links.append(a)}}status.textContent='Complete';document.querySelector('#result').scrollIntoView({behavior:'smooth'});}catch(err){status.textContent='Error: '+err.message;}finally{button.disabled=false;}};
</script></body></html>'''


def resolve_run_dir(run_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{12}", run_id):
        raise ValueError("Invalid run ID")
    run_dir = (RUNS / run_id).resolve()
    try:
        run_dir.relative_to(RUNS.resolve())
    except ValueError as exc:
        raise ValueError("Invalid run ID") from exc
    return run_dir


def run_sample(run_id: str, x: int, y: int) -> dict:
    """Return exact source-grid values at zero-based column x, row y."""
    run_dir = resolve_run_dir(run_id)
    metadata_path = run_dir / "metadata.json"
    height_path = run_dir / "height_map.npy"
    relative_path = run_dir / "relative_depth.npy"
    if not metadata_path.is_file() or not height_path.is_file() or not relative_path.is_file():
        raise FileNotFoundError("Run or required output was not found")
    height = np.load(height_path, allow_pickle=False)
    relative = np.load(relative_path, allow_pickle=False)
    if height.ndim != 2 or relative.shape != height.shape:
        raise ValueError("Run contains inconsistent output grids")
    if x < 0 or y < 0 or x >= height.shape[1] or y >= height.shape[0]:
        raise ValueError(f"Pixel coordinate out of bounds; expected x in [0,{height.shape[1]-1}], y in [0,{height.shape[0]-1}]")
    metadata = json.loads(metadata_path.read_text())
    result = {
        "run_id": run_id,
        "x": x,
        "y": y,
        "coordinate_convention": "zero-based pixel column x and row y in the original input grid",
        "relative_depth": float(relative[y, x]),
        "relative_depth_units": "unitless",
        "height": float(height[y, x]),
        "height_units": metadata.get("height_output", {}).get("units", "unknown"),
    }
    surface_path = run_dir / "dsm.npy"
    if surface_path.is_file():
        surface = __import__("numpy").load(surface_path, allow_pickle=False)
        result["dsm"] = None if not np.isfinite(surface[y, x]) else float(surface[y, x])
        result["dsm_units"] = "m, combining provided DEM and estimated height"
    slope_path = run_dir / "slope_degrees.npy"
    if slope_path.is_file():
        slope = __import__("numpy").load(slope_path, allow_pickle=False)
        result["slope_degrees"] = float(slope[y, x])
    return result


def run_surface(run_id: str, kind: str, max_size: int = 512) -> tuple[bytes, dict]:
    """Return a row-major little-endian float32 grid for browser 3D rendering."""
    if kind not in {"relative", "height", "dsm", "slope"}:
        raise ValueError("kind must be one of: relative, height, dsm, slope")
    if max_size < 2 or max_size > 1024:
        raise ValueError("max_size must be between 2 and 1024")
    run_dir = resolve_run_dir(run_id)
    filename = {"relative": "relative_depth.npy", "height": "height_map.npy", "dsm": "dsm.npy", "slope": "slope_degrees.npy"}[kind]
    path = run_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Surface '{kind}' is not available for this run")
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2:
        raise ValueError("Surface must be a 2D grid")
    original_height, original_width = array.shape
    ratio = min(1.0, max_size / max(original_height, original_width))
    if ratio < 1:
        size = (max(2, int(round(original_width * ratio))), max(2, int(round(original_height * ratio))))
        array = np.asarray(Image.fromarray(array.astype(np.float32), mode="F").resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
    else:
        array = array.astype(np.float32, copy=False)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    units = {
        "relative": "unitless",
        "height": metadata.get("height_output", {}).get("units", "unknown"),
        "dsm": "m (estimated combined with supplied DEM)" if (run_dir / "dsm.npy").is_file() else "unavailable",
        "slope": "degrees",
    }[kind]
    info = {
        "run_id": run_id,
        "kind": kind,
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
        "original_width": int(original_width),
        "original_height": int(original_height),
        "units": units,
        "format": "float32-le-row-major",
        "resampling": "bilinear" if ratio < 1 else "none",
    }
    return np.ascontiguousarray(array, dtype="<f4").tobytes(order="C"), info


class Handler(BaseHTTPRequestHandler):
    server_version = "DepthWizardLocal/1.0"

    def _send(self, status: int, payload: bytes, content_type: str, extra_headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Content-Type-Options", "nosniff")
        origin = self.headers.get("Origin")
        if origin:
            parsed_origin = urlsplit(origin)
            if parsed_origin.scheme == "http" and parsed_origin.hostname in {"localhost", "127.0.0.1", "::1"}:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Expose-Headers", "X-Surface-Width, X-Surface-Height, X-Original-Width, X-Original-Height, X-Surface-Units, X-Surface-Format, X-Surface-Resampling")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        if self.path == "/api/process" or self.path.startswith("/api/"):
            self._send(204, b"", "text/plain")
            return
        self._send(404, b"Not found", "text/plain")

    def do_GET(self) -> None:
        request = urlsplit(self.path)
        if request.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            return
        if request.path == "/api/health":
            checkpoint = ROOT / "outputs/stage12/checkpoints/experiment_B_best.pt"
            try:
                import rasterio  # noqa: F401
                rasterio_available = True
            except ImportError:
                rasterio_available = False
            self._send(200, json.dumps({"status": "ok", "model_checkpoint_available": checkpoint.is_file(), "rasterio_available": rasterio_available, "run_storage": str(RUNS.relative_to(ROOT))}).encode(), "application/json")
            return
        if request.path.startswith("/api/runs/"):
            parts = request.path.strip("/").split("/")
            try:
                if len(parts) == 3 and parts[2] and re.fullmatch(r"[0-9a-f]{12}", parts[2]):
                    metadata_path = RUNS / parts[2] / "metadata.json"
                    if not metadata_path.is_file():
                        raise FileNotFoundError("Run not found")
                    self._send(200, metadata_path.read_bytes(), "application/json")
                    return
                if len(parts) == 4 and parts[3] == "sample":
                    query = parse_qs(request.query)
                    if "x" not in query or "y" not in query:
                        raise ValueError("Both x and y query parameters are required")
                    sample = run_sample(parts[2], int(query["x"][0]), int(query["y"][0]))
                    self._send(200, json.dumps(sample, allow_nan=False).encode(), "application/json")
                    return
                if len(parts) == 4 and parts[3] == "surface":
                    query = parse_qs(request.query)
                    kind = query.get("kind", ["height"])[0]
                    max_size = int(query.get("max_size", ["512"])[0])
                    payload, info = run_surface(parts[2], kind, max_size)
                    headers = {
                        "X-Surface-Width": str(info["width"]),
                        "X-Surface-Height": str(info["height"]),
                        "X-Original-Width": str(info["original_width"]),
                        "X-Original-Height": str(info["original_height"]),
                        "X-Surface-Units": info["units"],
                        "X-Surface-Format": info["format"],
                        "X-Surface-Resampling": info["resampling"],
                    }
                    self._send(200, payload, "application/vnd.depthwizard.float32", headers)
                    return
            except FileNotFoundError as exc:
                self._send(404, json.dumps({"error": str(exc)}).encode(), "application/json")
                return
            except (ValueError, IndexError) as exc:
                self._send(400, json.dumps({"error": str(exc)}).encode(), "application/json")
                return
        if request.path.startswith("/files/"):
            relative = unquote(request.path.removeprefix("/files/"))
            target = (RUNS / relative).resolve()
            try:
                target.relative_to(RUNS.resolve())
            except ValueError:
                self._send(403, b"Forbidden", "text/plain"); return
            if not target.is_file():
                self._send(404, b"Not found", "text/plain"); return
            self._send(200, target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            return
        self._send(404, b"Not found", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/api/process":
            self._send(404, b"Not found", "text/plain"); return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 150 * 1024 * 1024:
            self._send(413, b'{"error":"Upload size must be between 1 byte and 150 MB"}', "application/json"); return
        try:
            message = BytesParser(policy=policy.default).parsebytes(
                b"Content-Type: " + self.headers.get("Content-Type", "").encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + self.rfile.read(length)
            )
            fields: dict[str, str] = {}
            files: dict[str, tuple[str, bytes]] = {}
            for part in message.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if not name:
                    continue
                payload = part.get_payload(decode=True) or b""
                filename = part.get_filename()
                if filename:
                    if payload:
                        files[name] = (Path(filename).name, payload)
                else:
                    fields[name] = payload.decode("utf-8", errors="replace")
            if "image" not in files:
                raise ValueError("Select an RGB image")
            image_name, image_bytes = files["image"]
            suffix = Path(image_name).suffix.lower()
            if suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                raise ValueError("Supported RGB formats: PNG, JPG/JPEG, GeoTIFF")
            run_id = uuid.uuid4().hex[:12]
            run_dir = RUNS / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            input_path = run_dir / f"input{suffix}"
            input_path.write_bytes(image_bytes)
            (run_dir / "input_preview.png").write_bytes(image_bytes) if suffix == ".png" else None
            if suffix != ".png":
                from PIL import Image
                from src.stage14_pipeline import load_rgb
                rgb, _ = load_rgb(input_path)
                Image.fromarray(rgb).save(run_dir / "input_preview.png")
            dem_path = None
            if "dem" in files:
                dem_name, dem_bytes = files["dem"]
                dem_suffix = Path(dem_name).suffix.lower()
                if dem_suffix not in {".tif", ".tiff"}:
                    raise ValueError("DEM input must be GeoTIFF")
                dem_path = run_dir / f"dem_input{dem_suffix}"
                dem_path.write_bytes(dem_bytes)
            from src.stage14_pipeline import process_image
            pixel_size = float(fields["pixel_size_m"]) if fields.get("pixel_size_m", "").strip() else None
            metadata = process_image(
                input_path, run_dir,
                calibration_mode=fields.get("calibration", "stage12_validation"),
                custom_scale=float(fields.get("scale", "1") or 1),
                custom_shift=float(fields.get("shift", "0") or 0),
                dem_path=dem_path,
                alignment_confirmed=fields.get("alignment_confirmed") == "true",
                pixel_size_m=pixel_size,
                custom_units_m=fields.get("custom_units_m") == "true",
            )
            metadata["run_id"] = run_id
            exports = metadata["exports"]
            exports["metadata"] = "metadata.json"
            exports["result_bundle"] = "result_bundle.zip"
            (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            with zipfile.ZipFile(run_dir / "result_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for artifact in run_dir.iterdir():
                    if artifact.is_file() and artifact.name != "result_bundle.zip":
                        archive.write(artifact, arcname=artifact.name)
            self._send(200, json.dumps(metadata).encode(), "application/json")
        except Exception as exc:
            payload = json.dumps({"error": str(exc)}).encode()
            self._send(400, payload, "application/json")

    def log_message(self, format: str, *args) -> None:
        print("DepthWizard:", format % args)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"DepthWizard demo: http://{args.host}:{args.port}")
    print("Stop with Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
