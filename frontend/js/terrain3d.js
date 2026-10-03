// Interactive 3D terrain built from the backend /surface grid with Three.js (vendored r160,
// the same version the backend's generated terrain_viewer.html uses).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { sampleColormap, percentiles, finiteRange } from './colormaps.js';

const WORLD = 100; // longer side of the mesh in scene units (pixel grid, not metres)
const BASE_RELIEF = 12; // display-normalised vertical range at 1× exaggeration
const RAMP_BY_KIND = { height: 'terrain', dsm: 'terrain', relative: 'viridis', slope: 'turbo' };

export class TerrainView {
  /**
   * @param {HTMLElement} container element that receives the canvas
   * @param {{onPick?: (x:number, y:number) => void}} options onPick receives source-grid column/row
   */
  constructor(container, { onPick } = {}) {
    this.container = container;
    this.onPick = onPick;
    this.surface = null;
    this.slope = null;
    this.textureCanvas = null;
    this.colorMode = 'rgb';
    this.exaggeration = 1;
    this.marker = null;
    this.markerSrc = null;
    this.active = false;
    this.ready = false;
  }

  /** Create renderer lazily. Returns false when WebGL is unavailable. */
  init() {
    if (this.ready) return true;
    try {
      this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: false });
    } catch {
      return false;
    }
    const r = this.renderer;
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.domElement.className = 'terrain-canvas';
    r.domElement.tabIndex = 0;
    r.domElement.setAttribute('role', 'application');
    r.domElement.setAttribute(
      'aria-label',
      '3D terrain. Drag to orbit, scroll or pinch to zoom, right-drag or arrow keys to pan. Click to inspect a point.',
    );
    this.container.append(r.domElement);

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x0a1016, 220, 520);
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 2000);
    this.controls = new OrbitControls(this.camera, r.domElement);
    Object.assign(this.controls, {
      enableDamping: true,
      dampingFactor: 0.08,
      screenSpacePanning: false,
      maxPolarAngle: Math.PI * 0.495,
      minDistance: 8,
      autoRotateSpeed: 0.8,
    });
    this.controls.listenToKeyEvents(r.domElement);

    this.scene.add(new THREE.HemisphereLight(0xe3f1ff, 0x1a2630, 1.5));
    const sun = new THREE.DirectionalLight(0xffffff, 1.7);
    sun.position.set(-70, 120, 50);
    this.scene.add(sun);

    const grid = new THREE.GridHelper(WORLD * 1.6, 32, 0x2a4152, 0x18262f);
    grid.position.y = -0.6;
    this.scene.add(grid);
    this.grid = grid;

    this.material = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.92, metalness: 0, side: THREE.DoubleSide });
    this.raycaster = new THREE.Raycaster();

    let down = null;
    r.domElement.addEventListener('pointerdown', (e) => {
      down = { x: e.clientX, y: e.clientY, button: e.button };
    });
    r.domElement.addEventListener('pointerup', (e) => {
      if (!down || down.button !== 0) return;
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
      down = null;
      if (moved < 5) this._pick(e);
    });

    this.resizeObserver = new ResizeObserver(() => this._resize());
    this.resizeObserver.observe(this.container);
    this._resize();
    this.ready = true;
    return true;
  }

  get canvas() {
    return this.renderer?.domElement;
  }

  setActive(active) {
    this.active = active;
    if (!this.ready) return;
    if (active) {
      this._resize();
      this.renderer.setAnimationLoop(() => {
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
      });
    } else {
      this.renderer.setAnimationLoop(null);
    }
  }

  /** @param {{data: Float32Array, width:number, height:number, originalWidth:number, originalHeight:number, kind:string, units:string}} surface */
  setSurface(surface) {
    this.surface = surface;
    const { data, width: w, height: h } = surface;
    const [lo, hi] = finiteRange(data);
    this.range = { lo, hi, span: Math.max(hi - lo, 1e-9) };
    const [p2, p98] = percentiles(data, [2, 98]);
    this.rampRange = { lo: p2, hi: p98 > p2 ? p98 : p2 + 1e-9 };

    const step = WORLD / (Math.max(w, h) - 1);
    this.step = step;
    const n = w * h;
    const positions = new Float32Array(n * 3);
    const uvs = new Float32Array(n * 2);
    for (let j = 0; j < h; j++) {
      for (let i = 0; i < w; i++) {
        const k = j * w + i;
        positions[k * 3] = (i - (w - 1) / 2) * step;
        positions[k * 3 + 2] = (j - (h - 1) / 2) * step;
        uvs[k * 2] = i / (w - 1);
        uvs[k * 2 + 1] = 1 - j / (h - 1);
      }
    }
    const IndexArray = n > 65535 ? Uint32Array : Uint16Array;
    const indices = new IndexArray((w - 1) * (h - 1) * 6);
    let t = 0;
    for (let j = 0; j < h - 1; j++) {
      for (let i = 0; i < w - 1; i++) {
        const a = j * w + i;
        const b = a + 1;
        const c = a + w;
        const d = c + 1;
        indices[t++] = a; indices[t++] = c; indices[t++] = b;
        indices[t++] = b; indices[t++] = c; indices[t++] = d;
      }
    }
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    this.mesh = new THREE.Mesh(geometry, this.material);
    this.scene.add(this.mesh);
    this._applyHeights();
    this._applyColors();
    this.resetView();
  }

  setTexture(image) {
    // Downscale to a safe GPU texture size and keep it in a canvas so flipY behaves consistently.
    const max = Math.min(2048, this.renderer?.capabilities.maxTextureSize || 2048);
    const iw = image.naturalWidth || image.width;
    const ih = image.naturalHeight || image.height;
    const ratio = Math.min(1, max / Math.max(iw, ih));
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(iw * ratio));
    canvas.height = Math.max(1, Math.round(ih * ratio));
    canvas.getContext('2d').drawImage(image, 0, 0, canvas.width, canvas.height);
    if (this.texture) this.texture.dispose();
    this.texture = new THREE.CanvasTexture(canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.texture.anisotropy = Math.min(8, this.renderer.capabilities.getMaxAnisotropy());
    this._applyMaterial();
  }

  /** mode: 'rgb' | 'ramp' | 'slope' */
  setColorMode(mode, slopeSurface = null) {
    this.colorMode = mode;
    if (slopeSurface) this.slope = slopeSurface;
    this._applyColors();
  }

  setExaggeration(k) {
    this.exaggeration = k;
    if (!this.mesh) return;
    if (this._pendingRelief) return;
    this._pendingRelief = requestAnimationFrame(() => {
      this._pendingRelief = null;
      this._applyHeights();
      if (this.markerSrc) this.setMarker(this.markerSrc.x, this.markerSrc.y);
    });
  }

  setAutoRotate(on) {
    if (this.controls) this.controls.autoRotate = on;
  }

  resetView() {
    if (!this.surface || !this.camera) return;
    const { width: w, height: h } = this.surface;
    const ex = (w - 1) * this.step;
    const ez = (h - 1) * this.step;
    const relief = BASE_RELIEF * this.exaggeration;
    const target = new THREE.Vector3(0, relief * 0.35, 0);
    const radius = 0.5 * Math.hypot(ex, ez, relief);
    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const fit = this.camera.aspect < 1 ? radius / Math.sin(fov / 2) / this.camera.aspect : radius / Math.sin(fov / 2);
    const dir = new THREE.Vector3(0.18, 0.62, 0.78).normalize();
    this.camera.position.copy(target).addScaledVector(dir, fit * 0.92);
    this.controls.target.copy(target);
    this.controls.maxDistance = fit * 4;
    this.camera.near = Math.max(0.05, fit / 500);
    this.camera.far = fit * 10;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  /** action: zoomIn | zoomOut | rotateLeft | rotateRight */
  nudge(action) {
    if (!this.controls) return;
    const offset = this.camera.position.clone().sub(this.controls.target);
    if (action === 'zoomIn' || action === 'zoomOut') {
      const f = action === 'zoomIn' ? 0.8 : 1.25;
      const len = THREE.MathUtils.clamp(offset.length() * f, this.controls.minDistance, this.controls.maxDistance);
      offset.setLength(len);
    } else {
      offset.applyAxisAngle(new THREE.Vector3(0, 1, 0), THREE.MathUtils.degToRad(action === 'rotateLeft' ? 15 : -15));
    }
    this.camera.position.copy(this.controls.target).add(offset);
    this.controls.update();
  }

  /** Place the inspection marker at a source-grid column/row. */
  setMarker(srcX, srcY) {
    if (!this.surface || !this.mesh) return;
    this.markerSrc = { x: srcX, y: srcY };
    const { width: w, height: h, originalWidth: ow, originalHeight: oh } = this.surface;
    const i = Math.round((srcX / Math.max(ow - 1, 1)) * (w - 1));
    const j = Math.round((srcY / Math.max(oh - 1, 1)) * (h - 1));
    const pos = this.mesh.geometry.attributes.position;
    const k = j * w + i;
    if (!this.marker) {
      const group = new THREE.Group();
      const mat = new THREE.MeshBasicMaterial({ color: 0xffc35a, depthTest: false, transparent: true });
      const pin = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 7, 8), mat);
      pin.position.y = 3.5;
      const head = new THREE.Mesh(new THREE.SphereGeometry(0.9, 20, 14), mat);
      head.position.y = 7.2;
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(0.9, 1.4, 32),
        new THREE.MeshBasicMaterial({ color: 0xffc35a, side: THREE.DoubleSide, depthTest: false, transparent: true, opacity: 0.85 }),
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = 0.05;
      group.add(pin, head, ring);
      group.renderOrder = 10;
      group.traverse((o) => (o.renderOrder = 10));
      this.marker = group;
      this.scene.add(group);
    }
    this.marker.position.set(pos.getX(k), pos.getY(k), pos.getZ(k));
    this.marker.visible = true;
  }

  clearMarker() {
    this.markerSrc = null;
    if (this.marker) this.marker.visible = false;
  }

  dispose() {
    this.setActive(false);
    this.resizeObserver?.disconnect();
    this.mesh?.geometry.dispose();
    this.material?.dispose();
    this.texture?.dispose();
    this.renderer?.dispose();
    this.renderer?.domElement.remove();
    this.ready = false;
  }

  // ---- internals ----
  _applyHeights() {
    const { data } = this.surface;
    const { lo, span } = this.range;
    const pos = this.mesh.geometry.attributes.position;
    const scale = (BASE_RELIEF * this.exaggeration) / span;
    for (let k = 0; k < data.length; k++) {
      const v = data[k];
      pos.array[k * 3 + 1] = Number.isFinite(v) ? (v - lo) * scale : 0;
    }
    pos.needsUpdate = true;
    this.mesh.geometry.computeVertexNormals();
    this.mesh.geometry.computeBoundingSphere();
    this.mesh.geometry.computeBoundingBox();
  }

  _applyColors() {
    if (!this.mesh) return;
    const colors = this.mesh.geometry.attributes.color;
    const { data, width: w, height: h } = this.surface;
    if (this.colorMode === 'ramp') {
      const cmap = RAMP_BY_KIND[this.surface.kind] || 'viridis';
      const { lo, hi } = this.rampRange;
      for (let k = 0; k < data.length; k++) this._writeColor(colors.array, k, data[k], lo, hi, cmap);
    } else if (this.colorMode === 'slope' && this.slope) {
      const s = this.slope;
      const [, p98] = percentiles(s.data, [0, 98]);
      this.slopeRange = { lo: 0, hi: p98 > 0 ? p98 : 1 };
      for (let j = 0; j < h; j++) {
        const sj = Math.round((j / (h - 1)) * (s.height - 1));
        for (let i = 0; i < w; i++) {
          const si = Math.round((i / (w - 1)) * (s.width - 1));
          this._writeColor(colors.array, j * w + i, s.data[sj * s.width + si], 0, this.slopeRange.hi, 'turbo');
        }
      }
    }
    colors.needsUpdate = true;
    this._applyMaterial();
  }

  _writeColor(arr, k, v, lo, hi, cmap) {
    if (!Number.isFinite(v)) {
      arr[k * 3] = arr[k * 3 + 1] = arr[k * 3 + 2] = 0.18;
      return;
    }
    const [r, g, b] = sampleColormap(cmap, (v - lo) / (hi - lo));
    arr[k * 3] = r;
    arr[k * 3 + 1] = g;
    arr[k * 3 + 2] = b;
  }

  _applyMaterial() {
    const useTexture = this.colorMode === 'rgb' && this.texture;
    this.material.map = useTexture ? this.texture : null;
    this.material.vertexColors = !useTexture && this.colorMode !== 'rgb';
    this.material.color.set(useTexture || this.material.vertexColors ? 0xffffff : 0x8fa5b5);
    this.material.needsUpdate = true;
  }

  _pick(event) {
    if (!this.mesh || !this.surface) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(ndc, this.camera);
    const hit = this.raycaster.intersectObject(this.mesh, false)[0];
    if (!hit?.uv) return;
    const { originalWidth: ow, originalHeight: oh } = this.surface;
    const x = Math.min(ow - 1, Math.max(0, Math.round(hit.uv.x * (ow - 1))));
    const y = Math.min(oh - 1, Math.max(0, Math.round((1 - hit.uv.y) * (oh - 1))));
    this.onPick?.(x, y);
  }

  _resize() {
    if (!this.renderer) return;
    const { clientWidth: w, clientHeight: h } = this.container;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }
}
