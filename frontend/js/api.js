// Small reusable client for the DepthWizard backend (scripts/run_demo.py).
// Every network call in the UI goes through this module so error handling is uniform.
import { API_BASE_URL } from '../config.js';

export const BACKEND_COMMAND = './venv/bin/python scripts/run_demo.py --host 127.0.0.1 --port 8765';
export const MAX_UPLOAD_BYTES = 150 * 1024 * 1024; // backend limit for the whole multipart body
export const SURFACE_KINDS = ['relative', 'height', 'dsm', 'slope'];

const base = API_BASE_URL.replace(/\/+$/, '');

export class ApiError extends Error {
  /**
   * @param {string} message human-readable message
   * @param {{status?: number, endpoint?: string, network?: boolean, aborted?: boolean}} details
   */
  constructor(message, { status = 0, endpoint = '', network = false, aborted = false } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.endpoint = endpoint;
    this.network = network;
    this.aborted = aborted;
  }
}

export function apiBaseUrl() {
  return base;
}

function networkError(endpoint) {
  return new ApiError(
    `Cannot reach the DepthWizard backend at ${base}. Start it from the project folder with: ${BACKEND_COMMAND}`,
    { endpoint, network: true },
  );
}

async function errorFromResponse(response, endpoint) {
  let detail = '';
  try {
    const text = await response.text();
    try {
      detail = JSON.parse(text).error || text;
    } catch {
      detail = text;
    }
  } catch {
    /* ignore body read failures */
  }
  detail = (detail || response.statusText || 'Request failed').toString().trim();
  return new ApiError(`${detail} (HTTP ${response.status})`, { status: response.status, endpoint });
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(base + path, options);
  } catch (err) {
    if (err?.name === 'AbortError') throw new ApiError('Request cancelled', { endpoint: path, aborted: true });
    throw networkError(path);
  }
  if (!response.ok) throw await errorFromResponse(response, path);
  return response;
}

async function requestJson(path, options) {
  const response = await request(path, options);
  try {
    return await response.json();
  } catch {
    throw new ApiError('Backend returned a response that is not valid JSON', { status: response.status, endpoint: path });
  }
}

/** GET /api/health → {status, model_checkpoint_available, rasterio_available, run_storage} */
export function getHealth({ signal } = {}) {
  return requestJson('/api/health', { signal, cache: 'no-store' });
}

/** GET /api/runs/{id} → run metadata */
export function getRun(runId) {
  return requestJson(`/api/runs/${encodeURIComponent(runId)}`, { cache: 'no-store' });
}

/** GET /api/runs/{id}/sample?x&y → exact source-grid values */
export function samplePoint(runId, x, y) {
  const q = new URLSearchParams({ x: String(x), y: String(y) });
  return requestJson(`/api/runs/${encodeURIComponent(runId)}/sample?${q}`);
}

const littleEndianHost = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

/**
 * GET /api/runs/{id}/surface → decoded float32 grid plus header metadata.
 * The backend may bilinearly downsample; exact values must come from samplePoint().
 */
export async function getSurface(runId, kind = 'height', maxSize = 512) {
  if (!SURFACE_KINDS.includes(kind)) throw new ApiError(`Unsupported surface kind: ${kind}`);
  const q = new URLSearchParams({ kind, max_size: String(maxSize) });
  const path = `/api/runs/${encodeURIComponent(runId)}/surface?${q}`;
  const response = await request(path);
  const header = (name) => response.headers.get(name);
  const width = Number(header('X-Surface-Width'));
  const height = Number(header('X-Surface-Height'));
  const format = header('X-Surface-Format') || '';
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 2 || height < 2) {
    throw new ApiError('Surface response is missing X-Surface-Width/X-Surface-Height headers (check backend CORS exposure)', { endpoint: path });
  }
  if (format && format !== 'float32-le-row-major') {
    throw new ApiError(`Unexpected surface format "${format}"`, { endpoint: path });
  }
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength !== width * height * 4) {
    throw new ApiError(`Surface size mismatch: expected ${width * height * 4} bytes, received ${buffer.byteLength}`, { endpoint: path });
  }
  let data;
  if (littleEndianHost) {
    data = new Float32Array(buffer);
  } else {
    const view = new DataView(buffer);
    data = new Float32Array(width * height);
    for (let i = 0; i < data.length; i++) data[i] = view.getFloat32(i * 4, true);
  }
  return {
    kind,
    data,
    width,
    height,
    originalWidth: Number(header('X-Original-Width')) || width,
    originalHeight: Number(header('X-Original-Height')) || height,
    units: header('X-Surface-Units') || 'unknown',
    resampling: header('X-Surface-Resampling') || 'unknown',
    format: format || 'float32-le-row-major',
  };
}

/** URL for an artifact served by GET /files/{run_id}/{filename} */
export function fileUrl(runId, filename) {
  return `${base}/files/${encodeURIComponent(runId)}/${encodeURIComponent(filename)}`;
}

/** Fetch an artifact and save it with its original filename (cross-origin safe). */
export async function downloadFile(runId, filename) {
  const path = `/files/${encodeURIComponent(runId)}/${encodeURIComponent(filename)}`;
  const response = await request(path);
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
  return blob.size;
}

/**
 * POST /api/process via XMLHttpRequest so real upload progress can be reported.
 * The API gives no inference progress, so callers get only: upload progress → upload done → response.
 * @returns {{promise: Promise<object>, abort: () => void}}
 */
export function processImage(formData, { onUploadProgress, onUploadComplete } = {}) {
  const endpoint = '/api/process';
  const xhr = new XMLHttpRequest();
  const promise = new Promise((resolve, reject) => {
    xhr.open('POST', base + endpoint);
    xhr.responseType = 'text';
    xhr.upload.onprogress = (e) => {
      if (onUploadProgress) onUploadProgress(e.loaded, e.lengthComputable ? e.total : 0);
    };
    xhr.upload.onload = () => onUploadComplete && onUploadComplete();
    xhr.onerror = () => reject(networkError(endpoint));
    xhr.onabort = () => reject(new ApiError('Stopped waiting for the backend. The server may still finish this run.', { endpoint, aborted: true }));
    xhr.onload = () => {
      let body = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* handled below */
      }
      if (xhr.status >= 200 && xhr.status < 300 && body && body.run_id) {
        resolve(body);
      } else {
        const detail = (body && body.error) || xhr.responseText || xhr.statusText || 'Processing failed';
        reject(new ApiError(`${detail} (HTTP ${xhr.status})`, { status: xhr.status, endpoint }));
      }
    };
    xhr.send(formData);
  });
  return { promise, abort: () => xhr.abort() };
}
