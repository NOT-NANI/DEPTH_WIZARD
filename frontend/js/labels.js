// Scientific labels, calibration options, export catalogue and formatting helpers.
// Wording here is deliberately conservative: relative depth is unitless; calibrated values are estimates.

export const CALIBRATION_MODES = {
  stage12_validation: {
    title: 'Stage 12 validation-calibrated estimate',
    short: 'Estimated height',
    expectedUnits: 'm (estimated)',
    description:
      'Frozen Experiment B output mapped with the Stage 12 validation-scene scale/shift. Produces an estimated height surface — not surveyed absolute elevation.',
  },
  relative: {
    title: 'Unitless relative output',
    short: 'Relative only',
    expectedUnits: 'unitless',
    description:
      'Normalised model output with no metric mapping. Use to inspect relative structure without implying metres.',
  },
  custom: {
    title: 'Custom scale / shift',
    short: 'Custom mapping',
    expectedUnits: 'user-specified',
    description:
      'Applies your own scale × (Stage 12 inverse-mapped model height) + shift. Units and accuracy depend entirely on your calibration reference.',
  },
};

/** Labels for backend height_output.mode values */
export const BACKEND_MODE_LABELS = {
  stage12_validation_calibrated: 'Stage 12 validation-calibrated estimate',
  relative_output: 'Unitless relative output',
  custom_scale_shift: 'Custom scale / shift',
};

/**
 * Export catalogue keyed by backend `exports` keys. Unknown keys still render generically.
 * group: data | preview | model
 */
export const EXPORT_CATALOG = {
  relative_depth_npy: { group: 'data', title: 'Relative-depth grid', kind: 'relative', desc: 'Float32 NumPy array on the source pixel grid. Unitless model output.' },
  height_map_npy: { group: 'data', title: 'Height map grid', kind: 'height', desc: 'Float32 NumPy array of the selected height interpretation.' },
  height_map_geotiff: { group: 'data', title: 'Height map GeoTIFF', kind: 'height', desc: 'Inherits the source raster CRS/transform. Only produced for raster input with Rasterio.' },
  dsm_npy: { group: 'data', title: 'DSM grid', kind: 'dsm', desc: 'Supplied ground DEM + estimated height. DEM and calibration errors both propagate.' },
  dsm_geotiff: { group: 'data', title: 'DSM GeoTIFF', kind: 'dsm', desc: 'DSM written on the source raster grid (raster input with Rasterio only).' },
  relative_depth_preview: { group: 'preview', title: 'Relative-depth preview', kind: 'relative', desc: 'Grayscale PNG, 2–98 % stretch. Visualisation only.' },
  height_map_preview: { group: 'preview', title: 'Height map preview', kind: 'height', desc: 'Terrain-colormap PNG, 2–98 % stretch. Visualisation only.' },
  slope_preview: { group: 'preview', title: 'Slope preview', kind: 'slope', desc: 'Slope in degrees (turbo colormap). Requires metre units and pixel spacing.' },
  mesh_bundle: { group: 'model', title: 'Textured terrain mesh', kind: 'mesh', desc: 'OBJ + MTL + RGB texture. Mesh XY uses pixel coordinates; no CRS is embedded.' },
  terrain_viewer: { group: 'model', title: 'Standalone terrain viewer', kind: 'viewer', desc: 'Generated HTML viewer (loads Three.js from a CDN).', open: true },
  result_bundle: { group: 'model', title: 'Full result bundle', kind: 'bundle', desc: 'ZIP of every artifact for this run, including metadata.' },
  metadata: { group: 'model', title: 'Run metadata', kind: 'meta', desc: 'JSON with calibration parameters, labels and the accuracy note.' },
};

export const EXPORT_GROUPS = [
  ['data', 'Data grids'],
  ['preview', 'Previews'],
  ['model', '3D & bundles'],
];

export function humanizeKey(key) {
  return key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return '—';
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB'];
  let v = bytes / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[i]}`;
}

export function formatDuration(ms) {
  const s = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

/** Format a measured value without implying more precision than the float32 grid carries. */
export function formatValue(v, digits = 4) {
  if (v === null || v === undefined) return '—';
  if (!Number.isFinite(v)) return 'no data';
  const abs = Math.abs(v);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e6)) return v.toExponential(3);
  return v.toFixed(digits);
}

export function isMetric(units) {
  return typeof units === 'string' && /^m\b/.test(units.trim());
}

export function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
