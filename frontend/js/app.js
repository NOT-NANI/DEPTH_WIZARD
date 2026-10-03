// DepthWizard frontend controller: add image → choose interpretation → process → inspect → export.
import {
  ApiError, BACKEND_COMMAND, MAX_UPLOAD_BYTES, apiBaseUrl, downloadFile, fileUrl,
  getHealth, getRun, getSurface, processImage, samplePoint,
} from './api.js';
import {
  BACKEND_MODE_LABELS, CALIBRATION_MODES, EXPORT_CATALOG, EXPORT_GROUPS,
  escapeHtml, formatBytes, formatDuration, formatValue, humanizeKey, isMetric,
} from './labels.js';
import { colormapGradient, percentiles } from './colormaps.js';
import { MapView } from './mapview.js';

const $ = (sel) => document.querySelector(sel);
const SURFACE_MAX = 512;
const RECENT_KEY = 'depthwizard.recentRuns';
const IMAGE_EXT = ['png', 'jpg', 'jpeg'];
const TIFF_EXT = ['tif', 'tiff'];

const state = {
  health: null,
  healthOk: false,
  file: null, // {file, ext, width, height, previewUrl, isTiff}
  dem: null,
  job: null,
  run: null,
  runName: '',
  layers: [],
  activeLayer: null,
  viewMode: '2d',
  surfaces: new Map(), // kind -> Promise<surface>
  terrain: null,
  terrainReady: false,
  rgbTexture: null,
  sample: null,
  history: [],
  sampleSeq: 0,
  downloaded: false,
};

// ---------------------------------------------------------------- utilities
function toast(message, type = 'info', ms = 5000) {
  const el = document.createElement('div');
  el.className = `toast${type === 'error' ? ' toast--error' : type === 'warn' ? ' toast--warn' : ''}`;
  el.textContent = message;
  $('#toasts').append(el);
  setTimeout(() => el.remove(), ms);
}

function setMsg(el, text, warn = false) {
  el.hidden = !text;
  el.textContent = text || '';
  el.classList.toggle('is-warn', !!warn);
}

function extOf(name) {
  const m = /\.([^.]+)$/.exec(name || '');
  return m ? m[1].toLowerCase() : '';
}

function calibrationMode() {
  return document.querySelector('input[name="calibration"]:checked').value;
}

function expectedHeightUnits() {
  const mode = calibrationMode();
  if (mode === 'custom') return $('#custom-metres').checked ? 'm (user-specified)' : 'user-specified';
  return CALIBRATION_MODES[mode].expectedUnits;
}

function unitChipClass(units, mode) {
  if (mode === 'relative' || mode === 'relative_output' || units === 'unitless') return 'chip chip--relative';
  if (isMetric(units)) return 'chip chip--height';
  return 'chip chip--neutral';
}

function loadImage(url, crossOrigin = false) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    if (crossOrigin) img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Image failed to load: ${url}`));
    img.src = url;
  });
}

// ---------------------------------------------------------------- flow rail
function updateFlow() {
  let done = 0;
  if (state.file && !state.file.error) done = 2;
  if (state.run) done = Math.max(done, 3);
  if (state.run && state.sample) done = 4;
  if (state.run && state.downloaded) done = 5;
  const order = ['image', 'calibration', 'process', 'inspect', 'export'];
  order.forEach((key, i) => {
    const li = document.querySelector(`[data-flow="${key}"]`);
    li.dataset.state = i < done ? 'done' : i === done ? 'current' : 'pending';
    li.querySelector('a').toggleAttribute('aria-current', i === done);
  });
}

// ---------------------------------------------------------------- backend health
async function checkHealth() {
  const pill = $('#backend-status');
  pill.dataset.state = 'checking';
  $('#backend-status-text').textContent = 'Checking backend…';
  try {
    const h = await getHealth();
    state.health = h;
    state.healthOk = h.status === 'ok';
    const ckpt = !!h.model_checkpoint_available;
    pill.dataset.state = state.healthOk && ckpt ? 'ok' : 'warn';
    $('#backend-status-text').textContent = !state.healthOk ? `Backend: ${h.status}` : ckpt ? 'Backend online' : 'Checkpoint missing';
    $('#health-service').textContent = h.status;
    $('#health-checkpoint').innerHTML = ckpt ? '<span class="chip chip--ok">available</span>' : '<span class="chip chip--off">missing</span>';
    $('#health-rasterio').innerHTML = h.rasterio_available ? '<span class="chip chip--ok">installed</span>' : '<span class="chip chip--off">not installed</span>';
    $('#health-storage').innerHTML = `<code>${escapeHtml(h.run_storage || '—')}</code>`;
    $('#backend-banner').hidden = true;
  } catch (err) {
    state.health = null;
    state.healthOk = false;
    pill.dataset.state = 'error';
    $('#backend-status-text').textContent = 'Backend offline';
    for (const id of ['#health-service', '#health-checkpoint', '#health-rasterio', '#health-storage']) $(id).textContent = '—';
    $('#backend-banner-text').textContent = err instanceof ApiError && err.network
      ? `No response from ${apiBaseUrl()}. Start the backend from the project folder:`
      : `${err.message}.`;
    $('#backend-command').textContent = BACKEND_COMMAND;
    $('#backend-banner').hidden = false;
  }
  applyCapabilities();
  validate();
}

function applyCapabilities() {
  const rasterio = !!state.health?.rasterio_available;
  const known = !!state.health;
  const badge = $('#rasterio-badge');
  badge.className = `chip ${!known ? 'chip--neutral' : rasterio ? 'chip--ok' : 'chip--off'}`;
  badge.textContent = !known ? 'backend offline' : rasterio ? 'Rasterio available' : 'GeoTIFF/DEM unavailable';
  $('#tif-format-note').textContent = rasterio ? '· GeoTIFF' : '· GeoTIFF needs Rasterio';
  $('#image-input').accept = rasterio
    ? '.png,.jpg,.jpeg,.tif,.tiff,image/png,image/jpeg,image/tiff'
    : '.png,.jpg,.jpeg,image/png,image/jpeg';
  const demInput = $('#dem-input');
  demInput.disabled = known && !rasterio;
  $('#dem-unavailable').hidden = !(known && !rasterio);
  $('#dem-aligned').disabled = known && !rasterio;
  if (demInput.disabled && state.dem) {
    demInput.value = '';
    state.dem = null;
  }
}

// ---------------------------------------------------------------- image selection
async function handleFiles(list) {
  if (!list || !list.length) return;
  if (list.length > 1) toast('Only the first file is used.', 'warn');
  const file = list[0];
  const msg = $('#image-msg');
  setMsg(msg, '');
  const ext = extOf(file.name);
  const rasterio = !!state.health?.rasterio_available;
  let error = '';
  if (TIFF_EXT.includes(ext) && !rasterio) {
    error = 'GeoTIFF input needs Rasterio on the backend, which is not installed. Use a PNG/JPG, or install Rasterio (./venv/bin/python -m pip install rasterio) and restart the backend.';
  } else if (!IMAGE_EXT.includes(ext) && !TIFF_EXT.includes(ext)) {
    error = `Unsupported format${ext ? ` “.${ext}”` : ''}. Use PNG or JPG${rasterio ? ', or an RGB GeoTIFF' : ''}.`;
  } else if (file.size === 0) {
    error = 'This file is empty (0 bytes).';
  } else if (file.size > MAX_UPLOAD_BYTES) {
    error = `This file is ${formatBytes(file.size)}; the backend accepts uploads up to 150 MB.`;
  }
  if (error) {
    setMsg(msg, error);
    return;
  }

  clearFile(false);
  const info = { file, ext, isTiff: TIFF_EXT.includes(ext), width: null, height: null, previewUrl: null };
  if (!info.isTiff) {
    const url = URL.createObjectURL(file);
    try {
      const img = await loadImage(url);
      info.width = img.naturalWidth;
      info.height = img.naturalHeight;
      info.previewUrl = url;
    } catch {
      URL.revokeObjectURL(url);
      setMsg(msg, 'This file could not be decoded as an image. It may be corrupted or not actually a PNG/JPG.');
      return;
    }
    const mp = (info.width * info.height) / 1e6;
    if (info.width < 32 || info.height < 32) {
      setMsg(msg, `Very small image (${info.width} × ${info.height} px). Output will be coarse.`, true);
    } else if (mp > 40) {
      setMsg(msg, `Large image (${mp.toFixed(0)} MP). Inference and export on CPU may take several minutes.`, true);
    }
  }
  state.file = info;
  renderFileCard();
  validate();
  updateFlow();
}

function renderFileCard() {
  const f = state.file;
  $('#dropzone').hidden = !!f;
  $('#file-card').hidden = !f;
  if (!f) return;
  const thumbWrap = $('#file-thumb-wrap');
  const thumb = $('#file-thumb');
  thumbWrap.querySelector('.thumb-text')?.remove();
  if (f.previewUrl) {
    thumb.hidden = false;
    thumb.src = f.previewUrl;
  } else {
    thumb.hidden = true;
    const t = document.createElement('span');
    t.className = 'thumb-text';
    t.textContent = 'GeoTIFF preview after processing';
    thumbWrap.append(t);
  }
  $('#file-name').textContent = f.file.name;
  $('#file-name').title = f.file.name;
  $('#file-dims').textContent = f.width ? `${f.width} × ${f.height} px` : 'read by backend';
  $('#file-size').textContent = formatBytes(f.file.size);
  $('#file-format').textContent = f.isTiff ? 'GeoTIFF' : f.ext === 'png' ? 'PNG' : 'JPEG';
}

function clearFile(render = true) {
  if (state.file?.previewUrl) URL.revokeObjectURL(state.file.previewUrl);
  state.file = null;
  $('#image-input').value = '';
  if (render) {
    setMsg($('#image-msg'), '');
    renderFileCard();
    validate();
    updateFlow();
  }
}

// ---------------------------------------------------------------- calibration + validation
function onCalibrationChange() {
  const mode = calibrationMode();
  $('#custom-fields').hidden = mode !== 'custom';
  const units = expectedHeightUnits();
  const chip = $('#units-preview');
  chip.textContent = units;
  chip.className = unitChipClass(units, mode);
  validate();
}

function readNumber(input) {
  const raw = input.value.trim();
  if (raw === '') return { empty: true, value: null };
  const value = Number(raw);
  return { empty: false, value, valid: Number.isFinite(value) };
}

/** Returns a blocking reason or '' when ready. Also updates inline messages. */
function validate() {
  const mode = calibrationMode();
  let reason = '';

  // custom calibration
  let customMsg = '';
  if (mode === 'custom') {
    const s = readNumber($('#custom-scale'));
    const t = readNumber($('#custom-shift'));
    $('#custom-scale').setAttribute('aria-invalid', String(s.empty || !s.valid || s.value === 0));
    $('#custom-shift').setAttribute('aria-invalid', String(t.empty || !t.valid));
    if (s.empty || !s.valid || t.empty || !t.valid) customMsg = 'Enter numeric values for both scale and shift.';
    else if (s.value === 0) customMsg = 'Scale must be non-zero; a zero scale produces a flat surface.';
  }
  setMsg($('#custom-msg'), customMsg);

  // pixel size
  const px = readNumber($('#pixel-size'));
  let pxReason = '';
  if (!px.empty && (!px.valid || px.value <= 0)) pxReason = 'Pixel size must be a positive number of metres.';
  $('#pixel-size').setAttribute('aria-invalid', String(!!pxReason));
  const metric = isMetric(expectedHeightUnits());
  $('#pixel-hint').textContent = pxReason
    || (!px.empty && !metric
      ? 'Slope will not be computed: the selected height interpretation is not in metres.'
      : 'Slope is only computed when the height output is in metres and pixel spacing is known.');
  $('#pixel-hint').style.color = pxReason ? 'var(--red)' : '';

  // DEM
  let demMsg = '';
  const dem = state.dem;
  if (dem) {
    if (!TIFF_EXT.includes(extOf(dem.name))) demMsg = 'The ground DEM must be a GeoTIFF (.tif/.tiff).';
    else if (!state.health?.rasterio_available) demMsg = 'DEM processing is unavailable: Rasterio is not installed on the backend.';
    else if (!metric) demMsg = 'DEM integration requires a metre-based height interpretation (Stage 12 estimate, or custom with metres).';
    else if (state.file && !state.file.isTiff && !$('#dem-aligned').checked) demMsg = 'Confirm that the DEM has the same dimensions and is pixel-aligned with your PNG/JPG.';
    else if (state.file && (state.file.file.size + dem.size) > MAX_UPLOAD_BYTES) demMsg = 'Image + DEM exceed the 150 MB upload limit.';
  }
  setMsg($('#dem-msg'), demMsg);

  if (!state.file) reason = 'Add an image to continue.';
  else if (!state.health) reason = 'Backend unreachable — start it, then use Retry in the banner.';
  else if (!state.health.model_checkpoint_available) reason = 'The backend reports the frozen model checkpoint is missing.';
  else if (customMsg) reason = customMsg;
  else if (pxReason) reason = pxReason;
  else if (demMsg) reason = demMsg;

  const btn = $('#process-btn');
  const busy = !!state.job;
  btn.disabled = !!reason || busy;
  $('#process-hint').textContent = busy
    ? 'Processing…'
    : reason || `Ready: ${state.file.file.name} · ${CALIBRATION_MODES[mode].short} (${expectedHeightUnits()})`;
  return reason;
}

// ---------------------------------------------------------------- workspace states
function showState(name) {
  for (const s of ['empty', 'processing', 'error', 'results']) $(`#state-${s}`).hidden = s !== name;
  if (name !== 'results' && state.terrain) state.terrain.setActive(false);
  if (name === 'results' && state.viewMode === '3d' && state.terrain) state.terrain.setActive(true);
  if (name === 'empty') renderRecent();
}

function setPhase(id, phaseState, text) {
  const li = $(`#phase-${id}`);
  li.dataset.state = phaseState;
  if (text !== undefined) {
    const meta = { upload: '#upload-text', infer: '#infer-text', load: '#load-text' }[id];
    $(meta).textContent = text;
  }
}

function showError(err, title = 'Processing failed') {
  $('#error-title').textContent = title;
  $('#error-text').textContent = err?.message || String(err);
  $('#error-endpoint').textContent = err?.endpoint ? `Endpoint: ${err.endpoint}${err.status ? ` · HTTP ${err.status}` : ''}` : '';
  showState('error');
}

// ---------------------------------------------------------------- processing
function buildForm() {
  const fd = new FormData();
  fd.append('image', state.file.file, state.file.file.name);
  const mode = calibrationMode();
  fd.append('calibration', mode);
  if (mode === 'custom') {
    fd.append('scale', $('#custom-scale').value.trim());
    fd.append('shift', $('#custom-shift').value.trim());
    if ($('#custom-metres').checked) fd.append('custom_units_m', 'true');
  }
  const px = $('#pixel-size').value.trim();
  if (px) fd.append('pixel_size_m', px);
  if (state.dem) {
    fd.append('dem', state.dem, state.dem.name);
    if ($('#dem-aligned').checked) fd.append('alignment_confirmed', 'true');
  }
  return fd;
}

async function startProcessing() {
  if (validate() || state.job) return;
  const fileName = state.file.file.name;
  const procImg = $('#proc-image');
  procImg.hidden = !state.file.previewUrl;
  if (state.file.previewUrl) procImg.src = state.file.previewUrl;
  $('#proc-file').textContent = fileName;
  $('#upload-bar').style.width = '0%';
  $('#upload-bar').parentElement.setAttribute('aria-valuenow', '0');
  setPhase('upload', 'active', 'Starting…');
  setPhase('infer', 'pending', 'Waiting for upload');
  setPhase('load', 'pending', '—');
  showState('processing');
  window.scrollTo({ top: 0, behavior: 'smooth' });

  let timer = null;
  let inferStart = 0;
  const job = processImage(buildForm(), {
    onUploadProgress(loaded, total) {
      if (!total) {
        setPhase('upload', 'active', `${formatBytes(loaded)} sent`);
        return;
      }
      const pct = Math.min(100, Math.round((loaded / total) * 100));
      $('#upload-bar').style.width = `${pct}%`;
      $('#upload-bar').parentElement.setAttribute('aria-valuenow', String(pct));
      setPhase('upload', 'active', `${formatBytes(loaded)} of ${formatBytes(total)} · ${pct}%`);
    },
    onUploadComplete() {
      $('#upload-bar').style.width = '100%';
      setPhase('upload', 'done', 'Upload complete');
      inferStart = performance.now();
      setPhase('infer', 'active', '0:00 elapsed · waiting for backend response');
      timer = setInterval(() => {
        setPhase('infer', 'active', `${formatDuration(performance.now() - inferStart)} elapsed · waiting for backend response`);
      }, 500);
    },
  });
  state.job = job;
  validate();

  try {
    const meta = await job.promise;
    clearInterval(timer);
    if (!inferStart) setPhase('upload', 'done', 'Upload complete');
    setPhase('infer', 'done', inferStart ? `Completed in ${formatDuration(performance.now() - inferStart)}` : 'Completed');
    setPhase('load', 'active', 'Fetching previews and surface grid…');
    await showRun(meta, fileName, { fromProcessing: true });
    setPhase('load', 'done', 'Done');
    toast(`Run ${meta.run_id} complete.`);
  } catch (err) {
    clearInterval(timer);
    if (err instanceof ApiError && err.aborted) {
      toast(err.message, 'warn', 7000);
      showState(state.run ? 'results' : 'empty');
    } else {
      for (const p of ['upload', 'infer', 'load']) {
        if ($(`#phase-${p}`).dataset.state === 'active') setPhase(p, 'error');
      }
      showError(err);
    }
  } finally {
    state.job = null;
    validate();
  }
}

// ---------------------------------------------------------------- recent runs
function recentRuns() {
  try {
    return JSON.parse(localStorage.getItem(RECENT_KEY) || '[]');
  } catch {
    return [];
  }
}

function rememberRun(meta, name) {
  const list = recentRuns().filter((r) => r.id !== meta.run_id);
  const [h, w] = meta.input_shape || [];
  list.unshift({ id: meta.run_id, name, w, h, mode: meta.height_output?.mode, t: Date.now() });
  try {
    localStorage.setItem(RECENT_KEY, JSON.stringify(list.slice(0, 8)));
  } catch {
    /* storage unavailable */
  }
}

function forgetRun(id) {
  try {
    localStorage.setItem(RECENT_KEY, JSON.stringify(recentRuns().filter((r) => r.id !== id)));
  } catch {
    /* ignore */
  }
}

function renderRecent() {
  const list = recentRuns();
  $('#recent-runs').hidden = !list.length;
  const ul = $('#recent-list');
  ul.replaceChildren(
    ...list.map((r) => {
      const li = document.createElement('li');
      const b = document.createElement('button');
      b.type = 'button';
      b.innerHTML = `<span>${escapeHtml(r.name || 'Run')}</span><small>${escapeHtml(r.id)} · ${r.w ?? '?'}×${r.h ?? '?'} · ${escapeHtml(BACKEND_MODE_LABELS[r.mode]?.split(' ')[0] || '')}</small>`;
      b.addEventListener('click', () => openRun(r.id));
      li.append(b);
      return li;
    }),
  );
}

async function openRun(id) {
  if (!/^[0-9a-f]{12}$/.test(id)) {
    toast('Invalid run ID in URL.', 'error');
    return;
  }
  try {
    const meta = await getRun(id);
    const name = recentRuns().find((r) => r.id === id)?.name || meta.input_name || id;
    await showRun(meta, name);
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) forgetRun(id);
    toast(`Could not open run ${id}: ${err.message}`, 'error', 8000);
    history.replaceState(null, '', location.pathname + location.search);
    if (!state.run) showState('empty');
  }
}

// ---------------------------------------------------------------- results
function surface(kind) {
  if (!state.run) return Promise.reject(new Error('No run loaded'));
  if (!state.surfaces.has(kind)) {
    const p = getSurface(state.run.run_id, kind, SURFACE_MAX);
    p.catch(() => state.surfaces.delete(kind));
    state.surfaces.set(kind, p);
  }
  return state.surfaces.get(kind);
}

function heightKindLabel(meta) {
  const mode = meta.height_output?.mode;
  if (mode === 'relative_output') return 'Height output (unitless)';
  if (mode === 'custom_scale_shift') return 'Custom height';
  return 'Height estimate';
}

function buildLayers(meta) {
  const ex = meta.exports || {};
  const units = meta.height_output?.units || 'unknown';
  const layers = [{ key: 'rgb', label: 'RGB input', file: 'input_preview.png', swatch: 'linear-gradient(135deg,#d77a61,#5aa469,#4f7cac)' }];
  if (ex.relative_depth_preview) layers.push({ key: 'relative', label: 'Relative depth', file: ex.relative_depth_preview, cmap: 'gray', units: 'unitless', chip: 'chip--relative' });
  if (ex.height_map_preview) layers.push({ key: 'height', label: heightKindLabel(meta), file: ex.height_map_preview, cmap: 'terrain', units, chip: unitChipClass(units, meta.height_output?.mode).replace('chip ', '') });
  if (ex.dsm_npy) layers.push({ key: 'dsm', label: 'DSM (DEM + estimate)', file: 'dsm.png', cmap: 'terrain', units: 'm (DEM + estimate)', chip: 'chip--dsm' });
  if (ex.slope_preview) layers.push({ key: 'slope', label: 'Slope', file: ex.slope_preview, cmap: 'turbo', units: 'degrees', chip: 'chip--neutral' });
  for (const l of layers) if (l.cmap) l.swatch = colormapGradient(l.cmap, '135deg');
  return layers;
}

async function showRun(meta, name, { fromProcessing = false } = {}) {
  state.run = meta;
  state.runName = name || meta.input_name;
  state.surfaces = new Map();
  state.sample = null;
  state.history = [];
  state.downloaded = false;
  state.rgbTexture = null;
  state.terrainReady = false;
  rememberRun(meta, state.runName);
  history.replaceState(null, '', `#run=${meta.run_id}`);

  const [H, W] = meta.input_shape || [0, 0];
  const ho = meta.height_output || {};
  $('#run-id').textContent = meta.run_id;
  $('#run-name').textContent = state.runName;
  $('#run-grid').textContent = `${W} × ${H} px`;
  $('#run-mode').textContent = BACKEND_MODE_LABELS[ho.mode] || ho.mode || '—';
  const unitsEl = $('#run-units');
  unitsEl.textContent = ho.units || 'unknown';
  unitsEl.className = unitChipClass(ho.units, ho.mode);
  $('#run-geo').textContent = meta.input_georeferenced
    ? `From source raster: ${meta.input_crs || 'CRS present'}`
    : 'None — pixel grid only (no CRS / transform)';
  $('#accuracy-note').textContent = meta.accuracy_note || 'No accuracy note returned by the backend.';
  const labels = [meta.height_product_label, ho.label].filter((v, i, a) => v && a.indexOf(v) === i);
  $('#height-label').textContent = labels.join(' ');

  // 2D
  state.layers = buildLayers(meta);
  renderLayerTabs();
  state.map.setSource(W, H);
  state.map.setBase(fileUrl(meta.run_id, 'input_preview.png'), `RGB input preview for ${state.runName}`);
  const defaultLayer = state.layers.find((l) => l.key === 'height') ? 'height' : 'rgb';

  // 3D controls
  const ex = meta.exports || {};
  const kindSel = $('#surface-kind');
  kindSel.replaceChildren();
  const addOpt = (sel, value, text) => sel.append(new Option(text, value));
  addOpt(kindSel, 'height', `${heightKindLabel(meta)} · ${ho.units || '?'}`);
  if (ex.relative_depth_npy) addOpt(kindSel, 'relative', 'Relative depth · unitless');
  if (ex.dsm_npy) addOpt(kindSel, 'dsm', 'DSM · DEM + estimate');
  const colorSel = $('#color-mode');
  colorSel.replaceChildren();
  addOpt(colorSel, 'rgb', 'RGB texture');
  addOpt(colorSel, 'ramp', 'Value colormap');
  if (ex.slope_preview) addOpt(colorSel, 'slope', 'Slope (degrees)');
  $('#exaggeration').value = '1';
  $('#exaggeration-val').textContent = '1.00×';
  const viewer = $('#viewer-link');
  viewer.parentElement.hidden = !ex.terrain_viewer;
  if (ex.terrain_viewer) viewer.href = fileUrl(meta.run_id, ex.terrain_viewer);
  if (state.terrain) state.terrain.clearMarker();

  resetInspector();
  renderExports(meta);
  renderDetails(meta);

  // prefetch the primary surface (legend values + 3D)
  const primary = surface('height').catch((err) => {
    toast(`Surface grid unavailable: ${err.message}`, 'warn', 7000);
  });

  showState('results');
  setViewMode(state.viewMode === '3d' ? '3d' : '2d');
  try {
    await selectLayer(defaultLayer);
  } catch (err) {
    toast(err.message, 'warn');
  }
  if (fromProcessing) await primary;
  updateFlow();
  if (fromProcessing) $('#results-stage').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderLayerTabs() {
  const wrap = $('#layer-tabs');
  wrap.replaceChildren(
    ...state.layers.map((l) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'layer-tab';
      b.setAttribute('role', 'tab');
      b.dataset.layer = l.key;
      b.setAttribute('aria-selected', 'false');
      b.tabIndex = -1;
      b.innerHTML = `<span class="swatch" style="background:${l.swatch}"></span>${escapeHtml(l.label)}`;
      b.addEventListener('click', () => selectLayer(l.key).catch((e) => toast(e.message, 'warn')));
      return b;
    }),
  );
}

async function selectLayer(key) {
  const layer = state.layers.find((l) => l.key === key) || state.layers[0];
  state.activeLayer = layer;
  document.querySelectorAll('.layer-tab').forEach((b) => {
    const on = b.dataset.layer === layer.key;
    b.setAttribute('aria-selected', String(on));
    b.tabIndex = on ? 0 : -1;
  });
  const isRgb = layer.key === 'rgb';
  $('#opacity-wrap').hidden = isRgb;
  const opacity = Number($('#layer-opacity').value) / 100;
  renderLegend(layer);
  await state.map.setLayer(isRgb ? null : fileUrl(state.run.run_id, layer.file), opacity, `${layer.label} preview`);
}

async function renderLegend(layer) {
  const legend = $('#legend');
  const note = $('#legend-note');
  if (!layer.cmap) {
    legend.hidden = true;
    note.textContent = 'RGB input as decoded by the backend (EXIF orientation applied). Click any pixel to inspect.';
    return;
  }
  legend.hidden = false;
  $('#legend-title').innerHTML = `${escapeHtml(layer.label)} <span class="chip ${layer.chip}">${escapeHtml(layer.units)}</span>`;
  $('#legend-bar').style.background = colormapGradient(layer.cmap);
  $('#legend-lo').textContent = 'low';
  $('#legend-hi').textContent = 'high';
  note.textContent = 'Preview uses a 2–98 % contrast stretch for display only. Click a pixel for exact values.';
  try {
    const s = await surface(layer.key);
    if (state.activeLayer !== layer) return;
    const [p2, p98] = percentiles(s.data, [2, 98]);
    $('#legend-lo').textContent = `≈ ${formatValue(p2, 2)}`;
    $('#legend-hi').textContent = `≈ ${formatValue(p98, 2)}`;
    note.textContent = `Preview uses a 2–98 % contrast stretch for display only. Legend endpoints are approximate (from a ${s.width} × ${s.height} ${s.resampling === 'bilinear' ? 'downsampled' : ''} grid, units: ${s.units}). Click a pixel for exact values.`;
  } catch {
    /* legend keeps qualitative labels */
  }
}

// ---------------------------------------------------------------- 3D
function setViewMode(mode) {
  state.viewMode = mode;
  const is3d = mode === '3d';
  $('#mode-2d').setAttribute('aria-selected', String(!is3d));
  $('#mode-3d').setAttribute('aria-selected', String(is3d));
  $('#map-panel').hidden = is3d;
  $('#terrain-panel').hidden = !is3d;
  $('#layer-tabs').hidden = is3d;
  if (is3d) ensureTerrain();
  else {
    state.terrain?.setActive(false);
    state.map.fit();
  }
}

async function ensureTerrain() {
  if (!state.run) return;
  if (!state.terrain) {
    const { TerrainView } = await import('./terrain3d.js').catch((err) => {
      showTerrainFallback(`3D module failed to load (${err.message}).`);
      return {};
    });
    if (!TerrainView) return;
    state.terrain = new TerrainView($('#terrain-stage'), { onPick: (x, y) => inspect(x, y, '3d') });
    if (!state.terrain.init()) {
      showTerrainFallback('WebGL is not available in this browser, so the interactive 3D terrain cannot be shown.');
      return;
    }
  }
  if (!state.terrain.ready) return;
  state.terrain.setActive(state.viewMode === '3d');
  if (!state.terrainReady) await loadTerrainSurface();
}

function showTerrainFallback(text) {
  const el = $('#terrain-fallback');
  el.hidden = false;
  el.innerHTML = `<p>${escapeHtml(text)}</p><p>Use the 2D maps, or open the generated <a href="${$('#viewer-link').href}" target="_blank" rel="noopener">terrain_viewer.html</a>.</p>`;
}

async function loadTerrainSurface() {
  const runId = state.run.run_id;
  const kind = $('#surface-kind').value || 'height';
  const loading = $('#terrain-loading');
  $('#terrain-loading-text').textContent = 'Loading surface grid…';
  loading.hidden = false;
  try {
    const [s, tex] = await Promise.all([
      surface(kind),
      state.rgbTexture || loadImage(`${fileUrl(runId, 'input_preview.png')}?texture=1`, true).catch(() => null),
    ]);
    if (state.run?.run_id !== runId) return;
    state.terrain.setSurface(s);
    if (tex && !state.rgbTexture) {
      state.rgbTexture = tex;
      state.terrain.setTexture(tex);
    } else if (!tex) {
      toast('RGB texture could not be loaded; showing colormap instead.', 'warn');
      $('#color-mode').value = 'ramp';
    }
    await applyColorMode();
    if (state.sample) state.terrain.setMarker(state.sample.x, state.sample.y);
    state.terrainReady = true;
    $('#terrain-note').textContent = `Mesh ${s.width} × ${s.height} vertices${s.resampling === 'bilinear' ? `, bilinearly downsampled from the ${s.originalWidth} × ${s.originalHeight} source grid` : ' (source resolution)'}. XY are pixel units; the vertical scale is normalised for display × exaggeration. Surface units: ${s.units}. Clicks map to the nearest source pixel and are sampled exactly.`;
  } catch (err) {
    showTerrainFallback(`Surface could not be loaded: ${err.message}`);
  } finally {
    loading.hidden = true;
  }
}

async function applyColorMode() {
  const mode = $('#color-mode').value;
  const t = state.terrain;
  if (!t?.surface) return;
  const legend = $('#terrain-legend');
  if (mode === 'slope') {
    try {
      const slope = await surface('slope');
      t.setColorMode('slope', slope);
      const [, p98] = percentiles(slope.data, [0, 98]);
      legend.innerHTML = `<span>Slope</span><span class="mono">0°</span><span class="legend-bar" style="background:${colormapGradient('turbo')}"></span><span class="mono">≥ ${formatValue(p98, 1)}°</span>`;
      legend.hidden = false;
    } catch (err) {
      toast(`Slope surface unavailable: ${err.message}`, 'warn');
      $('#color-mode').value = 'rgb';
      t.setColorMode('rgb');
      legend.hidden = true;
    }
  } else if (mode === 'ramp') {
    t.setColorMode('ramp');
    const s = t.surface;
    const cmap = { height: 'terrain', dsm: 'terrain', relative: 'viridis' }[s.kind] || 'viridis';
    legend.innerHTML = `<span>${escapeHtml(s.units)}</span><span class="mono">${formatValue(t.rampRange.lo, 2)}</span><span class="legend-bar" style="background:${colormapGradient(cmap)}"></span><span class="mono">${formatValue(t.rampRange.hi, 2)}</span>`;
    legend.hidden = false;
  } else {
    t.setColorMode('rgb');
    legend.hidden = true;
  }
}

// ---------------------------------------------------------------- inspection
function resetInspector() {
  $('#inspect-empty').hidden = false;
  $('#inspect-result').hidden = true;
  setMsg($('#inspect-error'), '');
  $('#sample-x').value = '';
  $('#sample-y').value = '';
  const [H, W] = state.run?.input_shape || [0, 0];
  $('#sample-x').max = String(Math.max(0, W - 1));
  $('#sample-y').max = String(Math.max(0, H - 1));
  $('#sample-history-wrap').hidden = true;
  $('#sample-history').replaceChildren();
  state.map.clearMarker();
}

async function inspect(x, y, source = '2d') {
  if (!state.run) return;
  const [H, W] = state.run.input_shape;
  if (!Number.isInteger(x) || !Number.isInteger(y) || x < 0 || y < 0 || x >= W || y >= H) {
    setMsg($('#inspect-error'), `Pixel out of range. x must be 0–${W - 1} and y 0–${H - 1}.`);
    return;
  }
  setMsg($('#inspect-error'), '');
  state.map.setMarker(x, y);
  state.terrain?.setMarker(x, y);
  $('#sample-x').value = String(x);
  $('#sample-y').value = String(y);
  const seq = ++state.sampleSeq;
  const result = $('#inspect-result');
  result.style.opacity = '0.55';
  try {
    const data = await samplePoint(state.run.run_id, x, y);
    if (seq !== state.sampleSeq) return;
    state.sample = { x, y, data, source };
    renderSample(data, source);
    pushHistory(data);
    updateFlow();
  } catch (err) {
    if (seq !== state.sampleSeq) return;
    setMsg($('#inspect-error'), `Sample failed: ${err.message}`);
  } finally {
    if (seq === state.sampleSeq) result.style.opacity = '';
  }
}

function reading(cls, label, chipCls, chipText, value, extra = '') {
  return `<div class="reading reading--${cls}"><dt><span>${escapeHtml(label)}</span><span class="chip ${chipCls}">${escapeHtml(chipText)}</span></dt><dd>${value}${extra}</dd></div>`;
}

function renderSample(d, source) {
  const meta = state.run;
  const [H, W] = meta.input_shape;
  $('#inspect-empty').hidden = true;
  $('#inspect-result').hidden = false;
  $('#ins-x').textContent = d.x;
  $('#ins-y').textContent = d.y;
  $('#ins-grid').textContent = `of ${W} × ${H} source grid · zero-based`;
  const hu = d.height_units || meta.height_output?.units || 'unknown';
  const parts = [];
  parts.push(reading('relative', 'Relative depth', 'chip--relative', d.relative_depth_units || 'unitless', escapeHtml(formatValue(d.relative_depth, 4))));
  parts.push(reading('height', heightKindLabel(meta), unitChipClass(hu, meta.height_output?.mode).replace('chip ', ''), hu, escapeHtml(formatValue(d.height, 3))));
  if ('dsm' in d) {
    parts.push(reading('dsm', 'DSM', 'chip--dsm', 'm · DEM + estimate', d.dsm === null ? 'no data <small>(outside DEM)</small>' : escapeHtml(formatValue(d.dsm, 3))));
  }
  if ('slope_degrees' in d) {
    parts.push(reading('slope', 'Slope', 'chip--neutral', 'degrees', `${escapeHtml(formatValue(d.slope_degrees, 2))}<small>°</small>`));
  }
  $('#ins-readings').innerHTML = parts.join('');
  $('#ins-source').textContent = source === '3d'
    ? 'Picked on the downsampled 3D mesh, mapped to the nearest source pixel, and read exactly from the full-resolution grids.'
    : 'Read exactly from the full-resolution grids at this source pixel.';
}

function pushHistory(d) {
  state.history = [d, ...state.history.filter((h) => h.x !== d.x || h.y !== d.y)].slice(0, 6);
  $('#sample-history-wrap').hidden = state.history.length < 2;
  $('#sample-history').replaceChildren(
    ...state.history.map((h) => {
      const li = document.createElement('li');
      const b = document.createElement('button');
      b.type = 'button';
      b.innerHTML = `<span>(${h.x}, ${h.y})</span><span>${escapeHtml(formatValue(h.height, 2))} ${escapeHtml(h.height_units || '')}</span>`;
      b.setAttribute('aria-label', `Re-select pixel ${h.x}, ${h.y}`);
      b.addEventListener('click', () => inspect(h.x, h.y, '2d'));
      li.append(b);
      return li;
    }),
  );
}

// ---------------------------------------------------------------- exports + details
function exportChip(entry, meta) {
  const units = meta.height_output?.units || 'unknown';
  switch (entry?.kind) {
    case 'relative': return '<span class="chip chip--relative">unitless</span>';
    case 'height': return `<span class="${unitChipClass(units, meta.height_output?.mode)}">${escapeHtml(units)}</span>`;
    case 'dsm': return '<span class="chip chip--dsm">m · DEM + estimate</span>';
    case 'slope': return '<span class="chip chip--neutral">degrees</span>';
    default: return '';
  }
}

function renderExports(meta) {
  const ex = meta.exports || {};
  const groups = new Map(EXPORT_GROUPS.map(([g]) => [g, []]));
  for (const [key, file] of Object.entries(ex)) {
    if (!file) continue;
    const entry = EXPORT_CATALOG[key];
    groups.get(entry?.group || 'model').push({ key, file, entry });
  }
  const wrap = $('#export-groups');
  wrap.replaceChildren();
  for (const [g, title] of EXPORT_GROUPS) {
    const items = groups.get(g);
    if (!items.length) continue;
    const section = document.createElement('div');
    section.className = 'export-group';
    section.innerHTML = `<h3>${escapeHtml(title)}</h3>`;
    const ul = document.createElement('ul');
    ul.className = 'export-list';
    for (const { key, file, entry } of items) {
      const li = document.createElement('li');
      li.className = 'export-item';
      const name = entry?.title || humanizeKey(key);
      li.innerHTML = `<div><div class="ex-title">${escapeHtml(name)} ${exportChip(entry, meta)}</div><div class="ex-file">${escapeHtml(file)}</div></div><p class="ex-desc">${escapeHtml(entry?.desc || 'Artifact generated by the backend for this run.')}</p>`;
      if (entry?.open) {
        const a = document.createElement('a');
        a.className = 'btn btn--ghost btn--sm';
        a.href = fileUrl(meta.run_id, file);
        a.target = '_blank';
        a.rel = 'noopener';
        a.textContent = 'Open ↗';
        a.setAttribute('aria-label', `Open ${name} in a new tab`);
        li.append(a);
      } else {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'btn btn--ghost btn--sm';
        b.textContent = 'Download';
        b.setAttribute('aria-label', `Download ${name} (${file})`);
        b.addEventListener('click', () => doDownload(b, meta.run_id, file));
        li.append(b);
      }
      ul.append(li);
    }
    section.append(ul);
    wrap.append(section);
  }

  const missing = [];
  const rasterio = state.health?.rasterio_available;
  if (!ex.height_map_geotiff) {
    const why = !meta.input_georeferenced ? 'input has no CRS/transform' : '';
    missing.push(`GeoTIFF (${[why, rasterio === false ? 'Rasterio not installed' : ''].filter(Boolean).join('; ') || 'not produced'})`);
  }
  if (!ex.dsm_npy) missing.push('DSM (no ground DEM supplied)');
  if (!ex.slope_preview) missing.push(`slope (${meta.slope || 'not computed'})`);
  $('#export-missing').textContent = missing.length ? `Not produced for this run: ${missing.join(' · ')}.` : '';
}

async function doDownload(btn, runId, file) {
  const label = btn.textContent;
  btn.classList.add('is-busy');
  btn.textContent = 'Downloading…';
  try {
    const size = await downloadFile(runId, file);
    state.downloaded = true;
    updateFlow();
    toast(`Saved ${file} (${formatBytes(size)}).`);
  } catch (err) {
    toast(`Download failed for ${file}: ${err.message}`, 'error', 8000);
  } finally {
    btn.classList.remove('is-busy');
    btn.textContent = label;
  }
}

function renderDetails(meta) {
  const ho = meta.height_output || {};
  const rows = [
    ['Run ID', `<code>${escapeHtml(meta.run_id)}</code>`],
    ['Stored input name', escapeHtml(meta.input_name)],
    ['Input shape (H × W × C)', escapeHtml((meta.input_shape || []).join(' × '))],
    ['Georeferenced', meta.input_georeferenced ? 'Yes (from source raster)' : 'No'],
    ['CRS', meta.input_crs ? `<code>${escapeHtml(meta.input_crs)}</code>` : 'None assigned'],
    ['Transform', meta.input_transform ? `<code>${escapeHtml(meta.input_transform.join(', '))}</code>` : 'None assigned'],
    ['Relative depth', `${escapeHtml(meta.relative_depth_output?.units || 'unitless')} — ${escapeHtml(meta.relative_depth_output?.description || '')}`],
    ['Height mode', escapeHtml(BACKEND_MODE_LABELS[ho.mode] || ho.mode || '—')],
    ['Height units', escapeHtml(ho.units || '—')],
    ['Scale', ho.scale === null || ho.scale === undefined ? '—' : `<code>${escapeHtml(String(ho.scale))}</code>`],
    ['Shift', ho.shift_m === null || ho.shift_m === undefined ? '—' : `<code>${escapeHtml(String(ho.shift_m))}</code>`],
    ['Stage 12 inverse range (m)', Number.isFinite(ho.target_inverse_min_m) ? `<code>${formatValue(ho.target_inverse_min_m, 3)} … ${formatValue(ho.target_inverse_max_m, 3)}</code>` : '—'],
    ['Model', escapeHtml(ho.model_id || '—')],
    ['Checkpoint', ho.checkpoint ? `<code>${escapeHtml(ho.checkpoint)}</code>` : '—'],
    ['Height product', escapeHtml(meta.height_product_label || '—')],
    ['Slope', escapeHtml(meta.slope || '—')],
    ['DEM', meta.dem ? escapeHtml(meta.dem.description || 'supplied') : 'None supplied'],
  ];
  $('#run-details').innerHTML = rows.map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${v}</dd>`).join('');
}

// ---------------------------------------------------------------- events
function bind() {
  const input = $('#image-input');
  const dz = $('#dropzone');
  dz.addEventListener('click', () => input.click());
  dz.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      input.click();
    }
  });
  input.addEventListener('change', () => handleFiles(input.files));
  ['dragenter', 'dragover'].forEach((t) => dz.addEventListener(t, (e) => {
    e.preventDefault();
    dz.classList.add('is-over');
  }));
  ['dragleave', 'dragend', 'drop'].forEach((t) => dz.addEventListener(t, () => dz.classList.remove('is-over')));
  dz.addEventListener('drop', (e) => {
    e.preventDefault();
    handleFiles(e.dataTransfer?.files);
  });
  // Prevent accidental navigation when a file is dropped outside the drop zone.
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer?.files?.length && !state.job) handleFiles(e.dataTransfer.files);
  });
  $('#replace-file').addEventListener('click', () => input.click());
  $('#remove-file').addEventListener('click', () => clearFile());

  document.querySelectorAll('input[name="calibration"]').forEach((r) => r.addEventListener('change', onCalibrationChange));
  ['#custom-scale', '#custom-shift', '#pixel-size'].forEach((s) => $(s).addEventListener('input', validate));
  $('#custom-metres').addEventListener('change', onCalibrationChange);
  $('#dem-input').addEventListener('change', (e) => {
    state.dem = e.target.files[0] || null;
    validate();
  });
  $('#dem-aligned').addEventListener('change', validate);
  $('#process-btn').addEventListener('click', startProcessing);
  $('#cancel-btn').addEventListener('click', () => state.job?.abort());
  $('#error-retry').addEventListener('click', () => {
    showState(state.run ? 'results' : 'empty');
    checkHealth().then(() => {
      if (!validate()) startProcessing();
    });
  });
  $('#error-dismiss').addEventListener('click', () => showState(state.run ? 'results' : 'empty'));

  // backend popover
  const pill = $('#backend-status');
  const pop = $('#backend-popover');
  const closePop = () => {
    pop.hidden = true;
    pill.setAttribute('aria-expanded', 'false');
  };
  pill.addEventListener('click', (e) => {
    e.stopPropagation();
    pop.hidden = !pop.hidden;
    pill.setAttribute('aria-expanded', String(!pop.hidden));
  });
  document.addEventListener('click', (e) => {
    if (!pop.hidden && !pop.contains(e.target)) closePop();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !pop.hidden) {
      closePop();
      pill.focus();
    }
  });
  $('#health-recheck').addEventListener('click', checkHealth);
  $('#banner-retry').addEventListener('click', checkHealth);

  // view mode + layers (arrow-key navigation inside tablists)
  $('#mode-2d').addEventListener('click', () => setViewMode('2d'));
  $('#mode-3d').addEventListener('click', () => setViewMode('3d'));
  for (const list of [$('.segmented'), $('#layer-tabs')]) {
    list.addEventListener('keydown', (e) => {
      if (!['ArrowLeft', 'ArrowRight'].includes(e.key)) return;
      const tabs = [...list.querySelectorAll('[role="tab"]')];
      const i = tabs.indexOf(document.activeElement);
      if (i < 0) return;
      const next = tabs[(i + (e.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length];
      next.focus();
      next.click();
    });
  }
  $('#layer-opacity').addEventListener('input', (e) => {
    const v = Number(e.target.value);
    $('#layer-opacity-val').textContent = `${v}%`;
    state.map.setOpacity(v / 100);
  });

  // 3D controls
  $('#surface-kind').addEventListener('change', () => {
    state.terrainReady = false;
    if (state.terrain?.ready) loadTerrainSurface();
  });
  $('#color-mode').addEventListener('change', applyColorMode);
  $('#exaggeration').addEventListener('input', (e) => {
    const k = Number(e.target.value);
    $('#exaggeration-val').textContent = `${k.toFixed(2)}×`;
    state.terrain?.setExaggeration(k);
  });
  document.querySelectorAll('[data-nudge]').forEach((b) => b.addEventListener('click', () => state.terrain?.nudge(b.dataset.nudge)));
  $('#reset-view').addEventListener('click', () => state.terrain?.resetView());
  $('#auto-orbit').addEventListener('click', (e) => {
    const on = e.currentTarget.getAttribute('aria-pressed') !== 'true';
    e.currentTarget.setAttribute('aria-pressed', String(on));
    state.terrain?.setAutoRotate(on);
  });
  $('#fullscreen-3d').addEventListener('click', () => {
    const el = $('#terrain-stage');
    if (document.fullscreenElement) document.exitFullscreen();
    else el.requestFullscreen?.().catch(() => toast('Fullscreen is not available here.', 'warn'));
  });

  $('#sample-form').addEventListener('submit', (e) => {
    e.preventDefault();
    const x = Number($('#sample-x').value);
    const y = Number($('#sample-y').value);
    if ($('#sample-x').value === '' || $('#sample-y').value === '') {
      setMsg($('#inspect-error'), 'Enter both x (column) and y (row).');
      return;
    }
    inspect(x, y, '2d');
  });

  window.addEventListener('hashchange', () => {
    const m = /^#run=([0-9a-f]{12})$/.exec(location.hash);
    if (m && m[1] !== state.run?.run_id) openRun(m[1]);
  });
}

// ---------------------------------------------------------------- boot
function init() {
  $('#backend-url').textContent = apiBaseUrl();
  state.map = new MapView(
    {
      stage: $('#map-stage'), frame: $('#map-frame'), base: $('#map-base'), layer: $('#map-layer'),
      cursor: $('#map-cursor'), marker: $('#map-marker'), hover: $('#map-hover'), loading: $('#map-loading'),
    },
    (x, y) => inspect(x, y, '2d'),
  );
  bind();
  onCalibrationChange();
  renderRecent();
  updateFlow();
  checkHealth().then(() => {
    const m = /^#run=([0-9a-f]{12})$/.exec(location.hash);
    if (m) openRun(m[1]);
  });
}

init();
