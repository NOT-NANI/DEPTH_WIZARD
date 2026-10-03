"""Export a numeric height surface; georeference only from a georeferenced source."""
import json
from pathlib import Path
import numpy as np

def save_surface(array: np.ndarray, output: Path, source_raster: Path | None = None) -> Path:
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Surface must be a finite 2D array")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".tif", ".tiff"}:
        if source_raster is None:
            raise ValueError("GeoTIFF requires --source-raster with CRS and transform")
        try:
            import rasterio
        except ImportError as exc:
            raise RuntimeError("Install rasterio to write GeoTIFF output") from exc
        with rasterio.open(source_raster) as src:
            if (src.height, src.width) != array.shape:
                raise ValueError("Surface dimensions differ from georeferenced source")
            profile=src.profile.copy(); profile.update(driver="GTiff",count=1,dtype="float32",compress="deflate")
            with rasterio.open(output,"w",**profile) as dst: dst.write(array.astype(np.float32),1)
    else:
        np.save(output, array.astype(np.float32), allow_pickle=False)
        output=output.with_suffix(".npy")
    metadata={"shape":list(array.shape),"dtype":"float32","georeferenced":False,"source_raster":None}
    if source_raster is not None:
        try:
            import rasterio
            with rasterio.open(source_raster) as src:
                metadata.update(georeferenced=bool(src.crs),source_raster=str(source_raster),crs=str(src.crs),transform=list(src.transform)[:6])
        except ImportError:
            metadata.update(source_raster=str(source_raster),georeferencing_metadata="rasterio unavailable")
    output.with_suffix(output.suffix+".json").write_text(json.dumps(metadata,indent=2)+"\n")
    return output
