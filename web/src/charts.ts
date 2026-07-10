// Tiny hand-rolled canvas charts. No chart library: we draw exactly what the
// exhibits need, crisply, on HiDPI.

export interface Series {
  label: string;
  points: [number, number][]; // x, y
  color: string;
}

export function setupCanvas(canvas: HTMLCanvasElement, cssW: number, cssH: number) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

const AXIS = "#3a4150";
const LABEL = "#8a93a3";

export function drawLineChart(
  canvas: HTMLCanvasElement,
  series: Series[],
  opts: {
    w: number;
    h: number;
    logY?: boolean;
    yLabel?: string;
    xLabel?: string;
    marker?: number; // x position of scrub marker
    yMin?: number;
    yMax?: number;
  },
) {
  const { w, h } = opts;
  const ctx = setupCanvas(canvas, w, h);
  ctx.clearRect(0, 0, w, h);
  const padL = 46;
  const padB = 24;
  const padT = 8;
  const padR = 10;

  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = opts.yMin ?? Infinity;
  let yMax = opts.yMax ?? -Infinity;
  for (const s of series) {
    for (const [x, y] of s.points) {
      xMin = Math.min(xMin, x);
      xMax = Math.max(xMax, x);
      if (opts.yMin === undefined) yMin = Math.min(yMin, y);
      if (opts.yMax === undefined) yMax = Math.max(yMax, y);
    }
  }
  if (!isFinite(xMin) || !isFinite(yMin)) return;
  if (opts.logY) {
    yMin = Math.max(yMin, 1e-7);
    yMax = Math.max(yMax, yMin * 10);
  }
  if (yMax === yMin) yMax = yMin + 1;

  const ty = (y: number) => {
    let f: number;
    if (opts.logY) f = (Math.log10(y) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin));
    else f = (y - yMin) / (yMax - yMin);
    return padT + (1 - f) * (h - padT - padB);
  };
  const tx = (x: number) => padL + ((x - xMin) / (xMax - xMin || 1)) * (w - padL - padR);

  // axes
  ctx.strokeStyle = AXIS;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, h - padB);
  ctx.lineTo(w - padR, h - padB);
  ctx.stroke();

  ctx.fillStyle = LABEL;
  ctx.font = "10px system-ui";
  // y ticks
  const ticks = opts.logY
    ? logTicks(yMin, yMax)
    : [yMin, yMin + (yMax - yMin) / 2, yMax];
  for (const t of ticks) {
    const y = ty(t);
    ctx.fillText(fmt(t), 4, y + 3);
    ctx.strokeStyle = "#1c212b";
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
    ctx.stroke();
  }
  // x ticks
  for (const f of [0, 0.5, 1]) {
    const x = xMin + f * (xMax - xMin);
    ctx.fillText(fmt(x), tx(x) - 8, h - 8);
  }
  if (opts.xLabel) ctx.fillText(opts.xLabel, w - padR - 40, h - 8);

  for (const s of series) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    let first = true;
    for (const [x, y] of s.points) {
      const yy = opts.logY ? Math.max(y, yMin) : y;
      if (first) {
        ctx.moveTo(tx(x), ty(yy));
        first = false;
      } else ctx.lineTo(tx(x), ty(yy));
    }
    ctx.stroke();
  }

  // legend
  let lx = padL + 8;
  for (const s of series) {
    ctx.fillStyle = s.color;
    ctx.fillRect(lx, padT + 2, 10, 3);
    ctx.fillStyle = LABEL;
    ctx.fillText(s.label, lx + 14, padT + 8);
    lx += 14 + ctx.measureText(s.label).width + 16;
  }

  if (opts.marker !== undefined) {
    ctx.strokeStyle = "#5eead4";
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(tx(opts.marker), padT);
    ctx.lineTo(tx(opts.marker), h - padB);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

function logTicks(lo: number, hi: number): number[] {
  const out: number[] = [];
  for (let e = Math.ceil(Math.log10(lo)); e <= Math.floor(Math.log10(hi)); e++)
    out.push(Math.pow(10, e));
  return out.length ? out : [lo, hi];
}

function fmt(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 10000) return `${Math.round(v / 1000)}k`;
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(1);
  if (a >= 0.001) return v.toFixed(3);
  return v.toExponential(0);
}

export function drawBars(
  canvas: HTMLCanvasElement,
  values: number[],
  opts: { w: number; h: number; highlight?: Set<number>; color?: string; logX?: boolean },
) {
  const { w, h } = opts;
  const ctx = setupCanvas(canvas, w, h);
  ctx.clearRect(0, 0, w, h);
  const pad = 6;
  const max = Math.max(...values, 1e-9);
  const bw = (w - 2 * pad) / values.length;
  for (let i = 0; i < values.length; i++) {
    const bh = (values[i] / max) * (h - 2 * pad);
    ctx.fillStyle = opts.highlight?.has(i) ? "#f0abfc" : (opts.color ?? "#5eead4");
    ctx.fillRect(pad + i * bw, h - pad - bh, Math.max(bw - 1, 1), bh);
  }
}

// heatmap for attention patterns / embedding similarity
export function drawHeatmap(
  canvas: HTMLCanvasElement,
  data: Float32Array,
  rows: number,
  cols: number,
  opts: { w: number; h: number; symmetric?: boolean },
) {
  const { w, h } = opts;
  const ctx = setupCanvas(canvas, w, h);
  ctx.clearRect(0, 0, w, h);
  let max = 1e-9;
  let min = 0;
  for (const v of data) {
    max = Math.max(max, v);
    if (opts.symmetric) min = Math.min(min, v);
  }
  if (opts.symmetric) max = Math.max(max, -min);
  const cw = w / cols;
  const ch = h / rows;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = data[r * cols + c];
      if (opts.symmetric) {
        const f = Math.min(Math.abs(v) / max, 1);
        ctx.fillStyle = v >= 0 ? `rgba(94,234,212,${f})` : `rgba(240,171,252,${f})`;
      } else {
        const f = Math.min(v / max, 1);
        ctx.fillStyle = `rgba(94,234,212,${f})`;
      }
      ctx.fillRect(c * cw, r * ch, Math.ceil(cw), Math.ceil(ch));
    }
  }
}
