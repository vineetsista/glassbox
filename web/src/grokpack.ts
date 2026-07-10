// Parser for the .grokpack binary written by grok/export_web.py.

export interface GrokHeader {
  format: string;
  p: number;
  d_model: number;
  n_heads: number;
  n_ckpts: number;
  key_freqs: number[];
  n_freq: number;
  n_probe: number;
  probe_pairs: [number, number][];
  train_frac: number;
  record_fields: { name: string; dtype: string; count: number; labels?: string[] }[];
}

export interface GrokCheckpoint {
  step: number;
  metrics: Record<string, number>;
  embDft: Float32Array; // [n_freq]
  emb: Float32Array; // [(p+1) * d_model]
  attnProbe: Float32Array; // [n_probe * n_heads * 3]
}

export interface GrokPack {
  header: GrokHeader;
  ckpts: GrokCheckpoint[];
}

function f16ToF32(bits: number): number {
  const sign = (bits & 0x8000) ? -1 : 1;
  const exp = (bits >> 10) & 0x1f;
  const frac = bits & 0x3ff;
  if (exp === 0) return sign * Math.pow(2, -14) * (frac / 1024);
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

function readF16Array(view: DataView, offset: number, count: number): Float32Array {
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = f16ToF32(view.getUint16(offset + 2 * i, true));
  return out;
}

export function parseGrokpack(buf: ArrayBuffer): GrokPack {
  const view = new DataView(buf);
  const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 4));
  if (magic !== "GRPK") throw new Error("not a grokpack file");
  const version = view.getUint32(4, true);
  if (version !== 1) throw new Error(`unsupported grokpack version ${version}`);
  const hlen = view.getUint32(8, true);
  const header: GrokHeader = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buf, 12, hlen)),
  );

  const metricLabels = header.record_fields.find((f) => f.name === "metrics")!.labels!;
  const ckpts: GrokCheckpoint[] = [];
  let off = 12 + hlen;
  for (let c = 0; c < header.n_ckpts; c++) {
    const step = view.getUint32(off, true);
    off += 4;
    const metrics: Record<string, number> = {};
    for (let m = 0; m < metricLabels.length; m++) {
      metrics[metricLabels[m]] = view.getFloat32(off, true);
      off += 4;
    }
    const embDft = new Float32Array(header.n_freq);
    for (let i = 0; i < header.n_freq; i++) {
      embDft[i] = view.getFloat32(off, true);
      off += 4;
    }
    const embCount = (header.p + 1) * header.d_model;
    const emb = readF16Array(view, off, embCount);
    off += embCount * 2;
    const probeCount = header.n_probe * header.n_heads * 3;
    const attnProbe = readF16Array(view, off, probeCount);
    off += probeCount * 2;
    ckpts.push({ step, metrics, embDft, emb, attnProbe });
  }
  return { header, ckpts };
}

// cosine similarity matrix of number-token embeddings, downsampled to `n` rows
export function embCosSim(ck: GrokCheckpoint, p: number, dModel: number, n: number): Float32Array {
  const stride = Math.max(1, Math.floor(p / n));
  const idx: number[] = [];
  for (let i = 0; i < p; i += stride) idx.push(i);
  const m = idx.length;
  const out = new Float32Array(m * m);
  const norms = idx.map((i) => {
    let s = 0;
    for (let d = 0; d < dModel; d++) s += ck.emb[i * dModel + d] ** 2;
    return Math.sqrt(s) || 1;
  });
  for (let a = 0; a < m; a++) {
    for (let b = 0; b < m; b++) {
      let dot = 0;
      for (let d = 0; d < dModel; d++)
        dot += ck.emb[idx[a] * dModel + d] * ck.emb[idx[b] * dModel + d];
      out[a * m + b] = dot / (norms[a] * norms[b]);
    }
  }
  return out;
}
