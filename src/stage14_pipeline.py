"""Stage 14 inference, optional geospatial handling, surface export, and 3D assets."""
from __future__ import annotations

import base64
import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
STAGE12_CHECKPOINT = ROOT / "outputs/stage12/checkpoints/experiment_B_best.pt"
STAGE12_CALIBRATION = ROOT / "outputs/stage12/calibration.json"
STAGE12_CONFIG = ROOT / "outputs/stage12/training_config.json"


@dataclass
class GridInfo:
    source_path: Path
    is_raster: bool
    georeferenced: bool
    crs: Any = None
    transform: Any = None
    profile: dict[str, Any] | None = None
    rasterio_available: bool = False


def to_rgb8(bands: np.ndarray) -> np.ndarray:
    if bands.ndim != 3:
        raise ValueError("Expected H×W×C imagery")
    if bands.shape[-1] < 3:
        raise ValueError("RGB input must contain at least three bands")
    out = []
    for channel in range(3):
        band = bands[..., channel]
        if band.dtype == np.uint8:
            out.append(band)
        else:
            values = band[np.isfinite(band)]
            if values.size == 0:
                out.append(np.zeros(band.shape, dtype=np.uint8)); continue
            lo, hi = np.percentile(values, [2, 98])
            if hi <= lo:
                hi = lo + 1
            out.append(np.clip((band.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8))
    return np.stack(out, axis=-1)


def load_rgb(path: Path) -> tuple[np.ndarray, GridInfo]:
    path = path.resolve()
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rasterio
        except ImportError as exc:
            raise RuntimeError("GeoTIFF input requires optional rasterio. Install it with `python -m pip install rasterio`; no metadata will be guessed or stripped.") from exc
        with rasterio.open(path) as src:
            if src.count < 3:
                raise ValueError("GeoTIFF input needs at least three RGB bands")
            raw = np.moveaxis(src.read([1, 2, 3]), 0, -1)
            rgb = to_rgb8(raw)
            profile = src.profile.copy()
            grid = GridInfo(path, True, src.crs is not None, src.crs, src.transform, profile, True)
            if src.crs is None:
                grid.georeferenced = False
        return rgb, grid
    with Image.open(path) as im:
        image = ImageOps.exif_transpose(im)
        image.load()
        rgb = np.asarray(image.convert("RGB"))
    return rgb, GridInfo(path, False, False)


def run_frozen_model(rgb: np.ndarray, calibration_mode: str = "stage12_validation", custom_scale: float = 1.0, custom_shift: float = 0.0, custom_units_m: bool = False) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """Run frozen Stage 12 B and return labeled output plus its calibration metadata."""
    import torch
    from scripts.run_stage12_experiment import infer_normalized, load_model_state
    device = torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else torch.device("cpu")
    model, processor, low, high = load_model_state(STAGE12_CHECKPOINT, device)
    normalized = infer_normalized(model, processor, rgb, device)
    base_height = low + normalized * (high - low)
    calibration = json.loads(STAGE12_CALIBRATION.read_text())["Adapted B SmoothL1+edge"]
    if calibration_mode == "relative":
        values = normalized.astype(np.float32)
        info = {"mode": "relative_output", "units": "unitless", "scale": None, "shift_m": None, "label": "Model output in normalized relative units; not metres."}
    elif calibration_mode == "stage12_validation":
        scale, shift = float(calibration["scale"]), float(calibration["shift_m"])
        values = (scale * base_height + shift).astype(np.float32)
        info = {"mode": "stage12_validation_calibrated", "units": "m (estimated)", "scale": scale, "shift_m": shift, "label": "Height estimate using Stage 12 validation-scene calibration; not independently validated absolute elevation."}
    elif calibration_mode == "custom":
        values = (float(custom_scale) * base_height + float(custom_shift)).astype(np.float32)
        units = "m (user-specified)" if custom_units_m else "user-specified"
        info = {"mode": "custom_scale_shift", "units": units, "scale": float(custom_scale), "shift_m": float(custom_shift), "label": "Custom user scale/shift; units and accuracy depend on the supplied calibration."}
    else:
        raise ValueError(f"Unsupported calibration mode: {calibration_mode}")
    del model, processor
    return values, {**info, "model_id": MODEL_ID, "checkpoint": str(STAGE12_CHECKPOINT.relative_to(ROOT)), "target_inverse_min_m": low, "target_inverse_max_m": high}, normalized.astype(np.float32)


def align_dem(dem_path: Path, grid: GridInfo, shape: tuple[int, int], alignment_confirmed: bool) -> tuple[np.ndarray, dict[str, Any]]:
    """Load/reproject DEM to RGB grid only from explicit geospatial metadata or user confirmation."""
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("DEM integration requires optional rasterio. Install it with `python -m pip install rasterio`.") from exc
    from rasterio.warp import Resampling, reproject
    with rasterio.open(dem_path) as dem:
        data = dem.read(1).astype(np.float32)
        nodata = dem.nodata
        if grid.georeferenced:
            if dem.crs is None or dem.transform is None:
                raise ValueError("A georeferenced RGB requires a DEM with CRS and transform for safe alignment")
            aligned = np.full(shape, np.nan, dtype=np.float32)
            reproject(source=data, destination=aligned, src_transform=dem.transform, src_crs=dem.crs,
                      src_nodata=nodata, dst_transform=grid.transform, dst_crs=grid.crs,
                      dst_nodata=np.nan, resampling=Resampling.bilinear)
            description = "DEM reprojected to RGB GeoTIFF CRS/transform with bilinear resampling"
        else:
            if data.shape != shape or not alignment_confirmed:
                raise ValueError("For non-georeferenced imagery, DEM must match dimensions and the user must confirm pixel alignment")
            aligned = data
            description = "DEM used on same-sized pixel grid after user confirmed alignment; no CRS assigned"
    if nodata is not None:
        aligned = np.where(aligned == nodata, np.nan, aligned)
    return aligned.astype(np.float32), {"path": str(dem_path), "description": description, "valid_pixels": int(np.isfinite(aligned).sum())}


def slope_degrees(height: np.ndarray, pixel_size_m: float) -> np.ndarray:
    if pixel_size_m <= 0 or not np.isfinite(pixel_size_m):
        raise ValueError("pixel_size_m must be a finite positive value")
    dz_dy, dz_dx = np.gradient(height.astype(np.float64), pixel_size_m, pixel_size_m)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    return slope.astype(np.float32)


def write_display(array: np.ndarray, path: Path, cmap: str = "gray") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    valid = np.isfinite(array)
    lo, hi = np.percentile(array[valid], [2, 98]) if valid.any() else (0, 1)
    normalized = np.zeros(array.shape, dtype=np.float32) if hi <= lo else np.clip((array - lo) / (hi - lo), 0, 1)
    if cmap == "gray":
        shown = (normalized * 255).astype(np.uint8)
        Image.fromarray(shown).save(path)
    else:
        rgb = (plt.get_cmap(cmap)(normalized)[..., :3] * 255).astype(np.uint8)
        Image.fromarray(rgb).save(path)
    plt.close("all")


def save_geotiff(array: np.ndarray, path: Path, grid: GridInfo) -> bool:
    if not grid.is_raster or not grid.rasterio_available or grid.profile is None:
        return False
    import rasterio
    profile = grid.profile.copy()
    profile.update(driver="GTiff", count=1, dtype="float32", height=array.shape[0], width=array.shape[1], compress="deflate", photometric="MINISBLACK")
    # Keep the exact source CRS/transform if present; never synthesize either.
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)
    return grid.georeferenced


def process_image(
    input_path: Path,
    run_dir: Path,
    calibration_mode: str = "stage12_validation",
    custom_scale: float = 1.0,
    custom_shift: float = 0.0,
    dem_path: Path | None = None,
    alignment_confirmed: bool = False,
    pixel_size_m: float | None = None,
    custom_units_m: bool = False,
) -> dict[str, Any]:
    rgb, grid = load_rgb(input_path)
    height, calibration, relative_depth = run_frozen_model(rgb, calibration_mode, custom_scale, custom_shift, custom_units_m)
    if height.shape != rgb.shape[:2] or not np.isfinite(height).all():
        raise RuntimeError("Model output is not a finite surface on the input grid")
    if relative_depth.shape != rgb.shape[:2] or not np.isfinite(relative_depth).all():
        raise RuntimeError("Model relative-depth output is not a finite surface on the input grid")
    metric_units = calibration["units"].startswith("m")
    if dem_path is not None and not metric_units:
        raise ValueError("DEM integration requires a height output calibrated in metres")
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "relative_depth.npy", relative_depth.astype(np.float32), allow_pickle=False)
    write_display(relative_depth, run_dir / "relative_depth.png", "gray")
    np.save(run_dir / "height_map.npy", height.astype(np.float32), allow_pickle=False)
    write_display(height, run_dir / "height_map.png", "terrain")
    geotiff = save_geotiff(height, run_dir / "height_map.tif", grid)
    product = height
    product_label = calibration["label"]
    dem_info = None
    if dem_path is not None:
        dem, dem_info = align_dem(dem_path, grid, height.shape, alignment_confirmed)
        valid_dem = np.isfinite(dem)
        if not valid_dem.any():
            raise ValueError("The aligned DEM has no finite ground pixels")
        dsm = np.full(height.shape, np.nan, dtype=np.float32)
        dsm[valid_dem] = dem[valid_dem] + height[valid_dem]
        np.save(run_dir / "dsm.npy", dsm, allow_pickle=False)
        if grid.is_raster and grid.rasterio_available:
            import rasterio
            profile = grid.profile.copy()
            profile.update(driver="GTiff", count=1, dtype="float32", height=dsm.shape[0], width=dsm.shape[1], compress="deflate", photometric="MINISBLACK", nodata=np.nan)
            with rasterio.open(run_dir / "dsm.tif", "w", **profile) as dst:
                dst.write(dsm, 1)
            geotiff = bool(grid.georeferenced)
        write_display(np.nan_to_num(dsm, nan=float(np.nanmedian(dsm))), run_dir / "dsm.png", "terrain")
        product, product_label = dsm, "DSM = user-provided ground DEM + estimated AGL; calibration and DEM errors propagate"

    inferred_pixel_size = pixel_size_m
    pixel_size_source = "user input" if pixel_size_m else None
    if inferred_pixel_size is None and grid.georeferenced and grid.transform is not None and grid.crs is not None:
        try:
            if grid.crs.is_projected and str(grid.crs.linear_units).lower() in {"metre", "meter", "meters", "metres"}:
                inferred_pixel_size = math.sqrt(abs(float(grid.transform.a * grid.transform.e)))
                pixel_size_source = "projected GeoTIFF pixel spacing"
        except Exception:
            inferred_pixel_size = None
    slope = None
    slope_label = "not computed: provide pixel size in metres or a projected GeoTIFF grid"
    if inferred_pixel_size is not None and metric_units:
        finite_product = np.where(np.isfinite(product), product, np.nanmedian(product))
        slope = slope_degrees(finite_product, float(inferred_pixel_size))
        np.save(run_dir / "slope_degrees.npy", slope, allow_pickle=False)
        write_display(slope, run_dir / "slope_degrees.png", "turbo")
        slope_label = f"slope in degrees from {pixel_size_source} ({inferred_pixel_size:g} m per pixel)"

    viewer_surface = product
    if not np.isfinite(viewer_surface).all():
        viewer_surface = np.where(np.isfinite(viewer_surface), viewer_surface, float(np.nanmedian(viewer_surface)))
    create_viewer(rgb, viewer_surface, run_dir / "terrain_viewer.html", product_label, slope)
    export_obj(rgb, viewer_surface, run_dir / "terrain_mesh.zip", product_label)
    metadata = {
        "input_name": input_path.name,
        "input_shape": list(rgb.shape),
        "input_georeferenced": grid.georeferenced,
        "input_crs": str(grid.crs) if grid.crs else None,
        "input_transform": list(grid.transform)[:6] if grid.transform else None,
        "relative_depth_output": {"units": "unitless", "description": "Frozen model output before Stage 12 inverse mapping and scale/shift calibration."},
        "height_output": calibration,
        "height_product_label": product_label,
        "dem": dem_info,
        "slope": slope_label,
        "exports": {"relative_depth_npy": "relative_depth.npy", "relative_depth_preview": "relative_depth.png", "height_map_npy": "height_map.npy", "height_map_preview": "height_map.png", "height_map_geotiff": "height_map.tif" if (run_dir / "height_map.tif").exists() else None, "dsm_npy": "dsm.npy" if (run_dir / "dsm.npy").exists() else None, "dsm_geotiff": "dsm.tif" if (run_dir / "dsm.tif").exists() else None, "slope_preview": "slope_degrees.png" if slope is not None else None, "terrain_viewer": "terrain_viewer.html", "mesh_bundle": "terrain_mesh.zip"},
        "accuracy_note": "This research visualization is not a surveyed or independently validated absolute elevation product. Ordinary PNG/JPG has no georeferencing; no CRS or transform is assigned.",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    bundle_files = [p for p in run_dir.iterdir() if p.is_file() and p.name != "result_bundle.zip"]
    import zipfile
    with zipfile.ZipFile(run_dir / "result_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in bundle_files:
            archive.write(file, arcname=file.name)
    return metadata


def _png_b64(image: Image.Image) -> str:
    buffer = io.BytesIO(); image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def create_viewer(rgb: np.ndarray, height: np.ndarray, output: Path, height_label: str, slope: np.ndarray | None = None, max_size: int = 256) -> Path:
    ratio = min(1, max_size / max(height.shape))
    size = (max(2, int(round(height.shape[1] * ratio))), max(2, int(round(height.shape[0] * ratio))))
    small_h = np.asarray(Image.fromarray(height.astype(np.float32), mode="F").resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
    small_rgb = Image.fromarray(rgb, "RGB").resize(size, Image.Resampling.BILINEAR)
    n = size[0]; rows = size[1]
    slope_units = "degrees" if slope is not None else "relative gradient per pixel"
    if slope is None:
        gy, gx = np.gradient(height.astype(np.float64))
        raw_gradient = np.hypot(gx, gy)
        small_slope = np.asarray(Image.fromarray(raw_gradient.astype(np.float32), mode="F").resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
    else:
        small_slope = np.asarray(Image.fromarray(slope.astype(np.float32), mode="F").resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
    heights = json.dumps(small_h.ravel().astype(float).tolist(), separators=(",", ":"))
    slopes = json.dumps(small_slope.ravel().astype(float).tolist(), separators=(",", ":"))
    rgb64 = _png_b64(small_rgb)
    slope_img = Image.fromarray((np.clip(small_slope / max(float(np.nanmax(small_slope)), 1e-6), 0, 1) * 255).astype(np.uint8)).convert("RGB")
    slope64 = _png_b64(slope_img)
    label_js = json.dumps(height_label)
    html = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DepthWizard terrain</title><style>html,body{{margin:0;background:#101820;color:#eef;font:14px system-ui}}#panel{{position:fixed;z-index:2;top:12px;left:12px;max-width:350px;background:#111d;padding:14px;border-radius:9px;line-height:1.5}}canvas{{display:block}}label{{display:block;margin-top:8px}}select,input{{max-width:200px}}#status{{font-weight:600;color:#9de}}</style></head><body><div id="panel"><b>DepthWizard 3D terrain preview</b><br><span id="unit"></span><br>Drag to orbit · wheel to zoom · W/A/S/D to fly · click terrain to inspect<label>Surface <select id="surface"><option value="rgb">RGB texture</option><option value="slope">Slope / relative gradient</option></select></label><label>Vertical exaggeration <input id="ex" type="range" min="0.2" max="5" value="1" step="0.1"></label><div id="status"></div></div><script type="module">import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';import {{OrbitControls}} from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';const W={n},HGT={rows},H={heights},S={slopes},unit={label_js},slopeUnits={json.dumps(slope_units)};document.querySelector('#unit').textContent='Height values: '+unit+' · slope display: '+slopeUnits;const renderer=new THREE.WebGLRenderer({{antialias:true}});renderer.setSize(innerWidth,innerHeight);document.body.appendChild(renderer.domElement);const scene=new THREE.Scene();scene.background=new THREE.Color(0x101820);const camera=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,.1,5000);camera.position.set(W*.8,Math.max(W,HGT)*.65,W*.8);camera.up.set(0,1,0);const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(W/2,0,HGT/2);controls.enableDamping=true;const geometry=new THREE.BufferGeometry(),positions=[],uvs=[],indices=[];const lo=H.reduce((a,b)=>Math.min(a,b),Infinity),hi=H.reduce((a,b)=>Math.max(a,b),-Infinity),span=Math.max(hi-lo,1e-6);for(let y=0;y<HGT;y++)for(let x=0;x<W;x++){{const i=y*W+x;positions.push(x,(H[i]-lo)*.15,(HGT-1-y));uvs.push(x/(W-1),1-y/(HGT-1));if(x<W-1&&y<HGT-1){{const a=i,b=i+1,c=i+W,d=i+W+1;indices.push(a,c,b,b,c,d)}}}}geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setAttribute('uv',new THREE.Float32BufferAttribute(uvs,2));geometry.setIndex(indices);geometry.computeVertexNormals();const loader=new THREE.TextureLoader(),rgbTex=loader.load('data:image/png;base64,{rgb64}'),slopeTex=loader.load('data:image/png;base64,{slope64}');const material=new THREE.MeshStandardMaterial({{map:rgbTex,side:THREE.DoubleSide}}),mesh=new THREE.Mesh(geometry,material);scene.add(mesh);scene.add(new THREE.HemisphereLight(0xffffff,0x334455,2));const light=new THREE.DirectionalLight(0xffffff,1.4);light.position.set(W,Math.max(W,HGT)*2,HGT);scene.add(light);const status=document.querySelector('#status'),ray=new THREE.Raycaster(),mouse=new THREE.Vector2();renderer.domElement.addEventListener('click',e=>{{mouse.x=e.clientX/innerWidth*2-1;mouse.y=-(e.clientY/innerHeight)*2+1;ray.setFromCamera(mouse,camera);const hit=ray.intersectObject(mesh)[0];if(hit?.uv){{const x=Math.min(W-1,Math.floor(hit.uv.x*(W-1))),y=Math.min(HGT-1,Math.floor((1-hit.uv.y)*(HGT-1))),i=y*W+x;status.textContent=`${{x}}, ${{y}}: ${{H[i].toFixed(2)}} ${{unit}} · ${{slopeUnits}} ${{S[i].toFixed(2)}}`}}}});document.querySelector('#surface').onchange=e=>material.map=e.target.value==='rgb'?rgbTex:slopeTex;document.querySelector('#ex').oninput=e=>{{const k=Number(e.target.value),p=geometry.attributes.position;for(let i=0;i<H.length;i++)p.setY(i,(H[i]-lo)*.15*k);p.needsUpdate=true;geometry.computeVertexNormals()}};const keys={{}};onkeydown=e=>keys[e.key.toLowerCase()]=true;onkeyup=e=>keys[e.key.toLowerCase()]=false;function animate(){{requestAnimationFrame(animate);if(keys.w)camera.translateZ(-.7);if(keys.s)camera.translateZ(.7);if(keys.a)camera.translateX(-.7);if(keys.d)camera.translateX(.7);controls.update();renderer.render(scene,camera)}}animate();onresize=()=>{{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)}};</script></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    return output


def export_obj(rgb: np.ndarray, height: np.ndarray, output_zip: Path, height_label: str, max_size: int = 256) -> Path:
    ratio = min(1, max_size / max(height.shape))
    size = (max(2, int(round(height.shape[1] * ratio))), max(2, int(round(height.shape[0] * ratio))))
    h = np.asarray(Image.fromarray(height.astype(np.float32), mode="F").resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
    texture = Image.fromarray(rgb, "RGB").resize(size, Image.Resampling.BILINEAR)
    rows, cols = h.shape
    obj = ["mtllib terrain.mtl", "usemtl terrain", f"# height values: {height_label}"]
    for y in range(rows):
        for x in range(cols): obj.append(f"v {x:.4f} {y:.4f} {float(h[y,x]):.6f}")
    for y in range(rows):
        for x in range(cols): obj.append(f"vt {x/(cols-1):.6f} {1-y/(rows-1):.6f}")
    for y in range(rows-1):
        for x in range(cols-1):
            a=y*cols+x+1; b=a+1; c=a+cols; d=c+1
            obj.extend((f"f {a}/{a} {c}/{c} {b}/{b}", f"f {b}/{b} {c}/{c} {d}/{d}"))
    mtl = "newmtl terrain\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nmap_Kd rgb_texture.png\n"
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("terrain.obj", "\n".join(obj)+"\n")
        archive.writestr("terrain.mtl", mtl)
        b=io.BytesIO(); texture.save(b,format="PNG"); archive.writestr("rgb_texture.png",b.getvalue())
        archive.writestr("README.txt",f"DepthWizard sampled terrain mesh. Height values: {height_label}. XY coordinates are pixel units unless exported with a geospatial tool; no CRS is embedded in OBJ.\n")
    return output_zip
