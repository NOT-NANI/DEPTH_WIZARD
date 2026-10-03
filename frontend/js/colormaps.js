// Colormaps matching the backend previews (matplotlib "terrain", "turbo", gray) plus viridis.
// Used for 3D vertex colouring and on-screen legends.

const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);

const MAPS = {
  terrain: [
    [0, [0.2, 0.2, 0.6]],
    [0.15, [0, 0.6, 1]],
    [0.25, [0, 0.8, 0.4]],
    [0.5, [1, 1, 0.6]],
    [0.75, [0.5, 0.36, 0.33]],
    [1, [1, 1, 1]],
  ],
  gray: [
    [0, [0, 0, 0]],
    [1, [1, 1, 1]],
  ],
  viridis: ['#440154', '#472d7b', '#3b528b', '#2c728e', '#21918c', '#28ae80', '#5ec962', '#addc30', '#fde725'].map(
    (c, i, a) => [i / (a.length - 1), hex(c)],
  ),
  turbo: [
    '#30123b', '#4145ab', '#4675ed', '#39a2fc', '#1bcfd4', '#24eca6', '#61fc6c', '#a4fc3b',
    '#d1e834', '#f3c63a', '#fe9b2d', '#f36315', '#d93806', '#b11901', '#7a0403',
  ].map((c, i, a) => [i / (a.length - 1), hex(c)]),
};

/** Interpolate a colormap at t∈[0,1] → [r,g,b] in 0..1 */
export function sampleColormap(name, t) {
  const stops = MAPS[name] || MAPS.gray;
  const v = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0));
  for (let i = 1; i < stops.length; i++) {
    const [p1, c1] = stops[i];
    if (v <= p1) {
      const [p0, c0] = stops[i - 1];
      const f = p1 === p0 ? 0 : (v - p0) / (p1 - p0);
      return [c0[0] + (c1[0] - c0[0]) * f, c0[1] + (c1[1] - c0[1]) * f, c0[2] + (c1[2] - c0[2]) * f];
    }
  }
  return stops[stops.length - 1][1];
}

/** CSS linear-gradient for a legend bar */
export function colormapGradient(name, direction = 'to right') {
  const stops = [];
  for (let i = 0; i <= 16; i++) {
    const t = i / 16;
    const [r, g, b] = sampleColormap(name, t);
    stops.push(`rgb(${Math.round(r * 255)} ${Math.round(g * 255)} ${Math.round(b * 255)}) ${(t * 100).toFixed(1)}%`);
  }
  return `linear-gradient(${direction}, ${stops.join(', ')})`;
}

/** Robust percentiles over finite values (sorted copy). */
export function percentiles(data, ps) {
  const finite = [];
  for (let i = 0; i < data.length; i++) if (Number.isFinite(data[i])) finite.push(data[i]);
  if (!finite.length) return ps.map(() => NaN);
  const sorted = Float32Array.from(finite).sort();
  return ps.map((p) => {
    const idx = Math.min(sorted.length - 1, Math.max(0, Math.round((p / 100) * (sorted.length - 1))));
    return sorted[idx];
  });
}

export function finiteRange(data) {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (Number.isFinite(v)) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return Number.isFinite(lo) ? [lo, hi] : [0, 1];
}
