// Wrapper around the Emscripten module (engine/build-wasm/glassbox.js).
// The module is copied into public/assets/ by scripts/build_site.sh and
// loaded dynamically so the app still renders if the engine is missing.

export interface FeatureAct {
  feature: number;
  value: number;
}

interface EmModule {
  ccall: (name: string, ret: string | null, args: string[], vals: unknown[]) => unknown;
  _malloc: (n: number) => number;
  _free: (p: number) => void;
  HEAPU8: Uint8Array;
}

export class GlassboxEngine {
  private m: EmModule;

  private constructor(m: EmModule) {
    this.m = m;
  }

  static async load(assetBase = "assets"): Promise<GlassboxEngine> {
    // resolve against the PAGE url, not the bundled module url — a bare
    // relative specifier would resolve inside dist/assets/ and 404
    const base = new URL(`${assetBase}/`, document.baseURI).href;
    const url = `${base}glassbox.js`;
    const factory = (await import(/* @vite-ignore */ url)).default as (opts?: object) => Promise<EmModule>;
    const m = await factory({
      locateFile: (f: string) => `${base}${f}`,
    });
    const gbxResp = await fetch(`${base}model.gbx`);
    if (!gbxResp.ok) throw new Error(`model.gbx: HTTP ${gbxResp.status}`);
    const bytes = new Uint8Array(await gbxResp.arrayBuffer());
    const ptr = m._malloc(bytes.length);
    m.HEAPU8.set(bytes, ptr);
    const rc = m.ccall("gbx_load", "number", ["number", "number"], [ptr, bytes.length]) as number;
    m._free(ptr);
    if (rc !== 0) throw new Error("gbx_load failed (corrupt model.gbx?)");
    return new GlassboxEngine(m);
  }

  private call(name: string, ret: "number" | "string" | null, args: string[] = [], vals: unknown[] = []) {
    return this.m.ccall(name, ret, args, vals);
  }

  vocabSize(): number {
    return this.call("gbx_vocab_size", "number") as number;
  }
  ctxLen(): number {
    return this.call("gbx_ctx_len", "number") as number;
  }
  hasSae(): boolean {
    return (this.call("gbx_has_sae", "number") as number) === 1;
  }
  nFeatures(): number {
    return this.call("gbx_n_features", "number") as number;
  }
  eotId(): number {
    return this.call("gbx_eot_id", "number") as number;
  }

  tokenize(text: string): number[] {
    const n = this.call("gbx_tokenize", "number", ["string"], [text]) as number;
    const ids: number[] = [];
    for (let i = 0; i < n; i++)
      ids.push(this.call("gbx_token_at", "number", ["number"], [i]) as number);
    return ids;
  }
  tokenStr(id: number): string {
    return this.call("gbx_token_str", "string", ["number"], [id]) as string;
  }
  text(): string {
    return this.call("gbx_text", "string") as string;
  }

  seed(v: number) {
    this.call("gbx_seed", null, ["number", "number"], [Math.floor(v / 2 ** 32), v >>> 0]);
  }
  reset() {
    this.call("gbx_reset", null);
  }
  /** feed one token; returns false when the context is full */
  feed(token: number): boolean {
    return (this.call("gbx_feed", "number", ["number"], [token]) as number) >= 0;
  }
  sample(temperature: number, topK: number): number {
    return this.call(
      "gbx_sample",
      "number",
      ["number", "number"],
      [temperature, topK],
    ) as number;
  }

  /** SAE features at the hook layer for the last token fed */
  lastFeatures(): FeatureAct[] {
    const n = this.call("gbx_feature_count", "number") as number;
    const out: FeatureAct[] = [];
    for (let i = 0; i < n; i++) {
      out.push({
        feature: this.call("gbx_feature_id", "number", ["number"], [i]) as number,
        value: this.call("gbx_feature_val", "number", ["number"], [i]) as number,
      });
    }
    return out;
  }

  setSteering(feature: number, multiplier: number) {
    this.call("gbx_set_steering", null, ["number", "number"], [feature, multiplier]);
  }
  clearSteering() {
    this.call("gbx_clear_steering", null);
  }
}
