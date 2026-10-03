// 2D map viewer: fits the source grid into the stage, stacks a layer over the RGB preview,
// converts pointer/keyboard positions to exact zero-based source pixel column/row.

export class MapView {
  /**
   * @param {object} els DOM elements
   * @param {(x:number, y:number) => void} onPick
   */
  constructor(els, onPick) {
    Object.assign(this, els);
    this.onPick = onPick;
    this.W = 0;
    this.H = 0;
    this.kbd = null; // keyboard crosshair position
    this._bind();
    new ResizeObserver(() => this.fit()).observe(this.stage);
  }

  setSource(width, height) {
    this.W = width;
    this.H = height;
    this.kbd = null;
    this.clearMarker();
    this.fit();
  }

  fit() {
    if (!this.W || !this.H) return;
    const pad = 24;
    const sw = Math.max(50, this.stage.clientWidth - pad * 2);
    const sh = Math.max(50, this.stage.clientHeight - pad * 2);
    const scale = Math.min(sw / this.W, sh / this.H);
    this.frame.style.width = `${Math.max(1, Math.floor(this.W * scale))}px`;
    this.frame.style.height = `${Math.max(1, Math.floor(this.H * scale))}px`;
  }

  setBase(url, alt) {
    this.base.src = url;
    if (alt) this.base.alt = alt;
  }

  /** Show a layer over the base. url=null shows only the base. Resolves when loaded. */
  setLayer(url, opacity = 1, alt = '') {
    return new Promise((resolve, reject) => {
      if (!url) {
        this.layer.hidden = true;
        this.layer.removeAttribute('src');
        this.loading.hidden = true;
        resolve();
        return;
      }
      this.layer.hidden = false;
      this.layer.alt = alt;
      this.layer.style.opacity = String(opacity);
      if (this.layer.src === url && this.layer.complete && this.layer.naturalWidth) {
        resolve();
        return;
      }
      this.loading.hidden = false;
      this.layer.onload = () => {
        this.loading.hidden = true;
        resolve();
      };
      this.layer.onerror = () => {
        this.loading.hidden = true;
        this.layer.hidden = true;
        reject(new Error('Preview image could not be loaded from the backend'));
      };
      this.layer.src = url;
    });
  }

  setOpacity(opacity) {
    this.layer.style.opacity = String(opacity);
  }

  setMarker(x, y) {
    this.kbd = { x, y };
    this.marker.hidden = false;
    if (!this.marker.firstChild) this.marker.append(document.createElement('span'));
    this.marker.style.left = `${((x + 0.5) / this.W) * 100}%`;
    this.marker.style.top = `${((y + 0.5) / this.H) * 100}%`;
  }

  clearMarker() {
    this.marker.hidden = true;
    this.cursor.hidden = true;
  }

  _pixelFromEvent(e) {
    const r = this.frame.getBoundingClientRect();
    const fx = (e.clientX - r.left) / r.width;
    const fy = (e.clientY - r.top) / r.height;
    if (fx < 0 || fy < 0 || fx >= 1 || fy >= 1) return null;
    return {
      x: Math.min(this.W - 1, Math.max(0, Math.floor(fx * this.W))),
      y: Math.min(this.H - 1, Math.max(0, Math.floor(fy * this.H))),
    };
  }

  _showHover(p) {
    this.hover.textContent = p ? `x ${p.x}, y ${p.y}` : 'x —, y —';
  }

  _showKbdCursor() {
    if (!this.kbd) return;
    this.cursor.hidden = false;
    this.cursor.style.left = `${((this.kbd.x + 0.5) / this.W) * 100}%`;
    this.cursor.style.top = `${((this.kbd.y + 0.5) / this.H) * 100}%`;
    this._showHover(this.kbd);
  }

  _bind() {
    let down = null;
    this.frame.addEventListener('pointermove', (e) => this._showHover(this._pixelFromEvent(e)));
    this.frame.addEventListener('pointerleave', () => this._showHover(null));
    this.frame.addEventListener('pointerdown', (e) => {
      down = { x: e.clientX, y: e.clientY };
    });
    this.frame.addEventListener('pointerup', (e) => {
      if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 6) return;
      down = null;
      const p = this._pixelFromEvent(e);
      if (p) {
        this.cursor.hidden = true;
        this.onPick(p.x, p.y);
      }
    });
    this.frame.addEventListener('keydown', (e) => {
      if (!this.W) return;
      const step = e.shiftKey ? 10 : 1;
      const moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
      if (moves[e.key]) {
        e.preventDefault();
        if (!this.kbd) this.kbd = { x: Math.floor(this.W / 2), y: Math.floor(this.H / 2) };
        else {
          this.kbd = {
            x: Math.min(this.W - 1, Math.max(0, this.kbd.x + moves[e.key][0])),
            y: Math.min(this.H - 1, Math.max(0, this.kbd.y + moves[e.key][1])),
          };
        }
        this._showKbdCursor();
      } else if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        if (!this.kbd) this.kbd = { x: Math.floor(this.W / 2), y: Math.floor(this.H / 2) };
        this.cursor.hidden = true;
        this.onPick(this.kbd.x, this.kbd.y);
      }
    });
    this.frame.addEventListener('focus', () => this._showKbdCursor());
    this.frame.addEventListener('blur', () => {
      this.cursor.hidden = true;
    });
  }
}
