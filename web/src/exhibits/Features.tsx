import { useEffect, useMemo, useRef, useState } from "react";
import { drawBars } from "../charts";
import { GlassboxEngine } from "../engine";

interface IndexEntry {
  id: number;
  density: number;
  max_act: number;
  top_tokens: string[];
  n_examples: number;
}
interface FeatureIndex {
  n_features: number;
  k: number;
  hook: string;
  scanned_tokens: number;
  features: IndexEntry[];
}
interface Example {
  tokens: string[];
  acts: number[];
  max_pos: number;
  max_act: number;
}
interface FeatureDetail {
  id: number;
  density: number;
  max_act: number;
  hist: { edges: number[]; counts: number[] };
  logit_lens: { promoted: [string, number][]; suppressed: [string, number][] };
  examples: Example[];
}

function TokenStrip({ ex }: { ex: Example }) {
  const max = Math.max(ex.max_act, 1e-6);
  return (
    <div className="tokens">
      {ex.tokens.map((t, i) => {
        const a = ex.acts[i] / max;
        const bg = a > 0 ? `rgba(94, 234, 212, ${Math.min(a, 1) * 0.55})` : "transparent";
        return (
          <span key={i} style={{ background: bg }} title={`act ${ex.acts[i].toFixed(3)}`}>
            {t}
          </span>
        );
      })}
    </div>
  );
}

export default function Features() {
  const [index, setIndex] = useState<FeatureIndex | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [sel, setSel] = useState<number | null>(null);
  const [detail, setDetail] = useState<FeatureDetail | null>(null);
  const [sortBy, setSortBy] = useState<"density" | "max_act" | "id">("density");
  const [filter, setFilter] = useState("");
  const histCanvas = useRef<HTMLCanvasElement>(null);

  // live probe
  const [engine, setEngine] = useState<GlassboxEngine | null>(null);
  const [probeText, setProbeText] = useState("Once upon a time, there was a little girl named Lily.");
  const [probeActs, setProbeActs] = useState<{ tok: string; act: number }[] | null>(null);

  useEffect(() => {
    fetch("assets/features/index.json")
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} - dashboards not baked yet?`);
        return r.json();
      })
      .then((j: FeatureIndex) => setIndex(j))
      .catch((e) => setErr(String(e)));
    GlassboxEngine.load().then(setEngine).catch(() => setEngine(null));
  }, []);

  useEffect(() => {
    if (sel === null) return;
    setDetail(null);
    fetch(`assets/features/f${sel}.json`)
      .then((r) => r.json())
      .then(setDetail)
      .catch((e) => setErr(String(e)));
  }, [sel]);

  useEffect(() => {
    if (!detail || !histCanvas.current) return;
    drawBars(histCanvas.current, detail.hist.counts.map((c) => Math.log10(1 + c)), {
      w: 480,
      h: 110,
      color: "#5eead4",
    });
  }, [detail]);

  const sorted = useMemo(() => {
    if (!index) return [];
    const fs = index.features.filter(
      (f) =>
        !filter ||
        String(f.id).includes(filter) ||
        f.top_tokens.some((t) => t.toLowerCase().includes(filter.toLowerCase())),
    );
    return [...fs].sort((a, b) =>
      sortBy === "id" ? a.id - b.id : sortBy === "density" ? b.density - a.density : b.max_act - a.max_act,
    );
  }, [index, sortBy, filter]);

  function runProbe() {
    if (!engine || sel === null) return;
    const ids = engine.tokenize(probeText).slice(0, engine.ctxLen());
    engine.clearSteering();
    engine.reset();
    const acts: { tok: string; act: number }[] = [];
    for (const id of ids) {
      if (!engine.feed(id)) break;
      const feats = engine.lastFeatures();
      const hit = feats.find((f) => f.feature === sel);
      acts.push({ tok: engine.tokenStr(id), act: hit ? hit.value : 0 });
    }
    setProbeActs(acts);
  }

  if (err)
    return (
      <div className="panel">
        <h2>Look inside</h2>
        <p className="err">{err}</p>
        <p className="note">
          Bake dashboards with <code>python -m sae.dashboards</code> after SAE training, then{" "}
          <code>scripts/build_site.sh</code>.
        </p>
      </div>
    );
  if (!index) return <div className="loading">loading feature index...</div>;

  return (
    <>
      <div className="panel">
        <h2>Look inside: {index.n_features} SAE features</h2>
        <p className="note">
          A top-k sparse autoencoder (k={index.k}) was trained on <code>{index.hook}</code>{" "}
          activations over {(index.scanned_tokens / 1e6).toFixed(1)}M tokens of TinyStories.
          Each feature is a direction in the residual stream; below are its strongest firing
          contexts (green = activation strength), activation histogram, and which output tokens
          the feature pushes toward or away from.
        </p>
        <div className="controls">
          <label>sort</label>
          <select value={sortBy} onChange={(e) => setSortBy(e.target.value as never)}>
            <option value="density">by density</option>
            <option value="max_act">by max activation</option>
            <option value="id">by id</option>
          </select>
          <label>filter</label>
          <input type="text" value={filter} placeholder="token or id" onChange={(e) => setFilter(e.target.value)} />
        </div>
        <div className="feature-grid">
          {sorted.slice(0, 400).map((f) => (
            <div
              key={f.id}
              className={`feature-card ${sel === f.id ? "sel" : ""}`}
              onClick={() => setSel(f.id)}
            >
              <div className="fid">#{f.id}</div>
              <div className="toks">{f.top_tokens.join(" ")}</div>
              <div style={{ color: "var(--dim)" }}>{(f.density * 100).toFixed(2)}%</div>
            </div>
          ))}
        </div>
      </div>

      {sel !== null && (
        <div className="panel">
          <h2>feature #{sel}</h2>
          {!detail && <div className="loading">loading...</div>}
          {detail && (
            <>
              <div>
                <span className="badge">density {(detail.density * 100).toFixed(3)}%</span>
                <span className="badge">max act {detail.max_act.toFixed(2)}</span>
              </div>
              <div className="row">
                <div>
                  <h3>max-activating examples</h3>
                  {detail.examples.slice(0, 8).map((ex, i) => (
                    <div key={i} style={{ marginBottom: 10 }}>
                      <TokenStrip ex={ex} />
                    </div>
                  ))}
                  {detail.examples.length === 0 && <p className="note">feature never fired in the scan (dead)</p>}
                </div>
                <div>
                  <h3>activation histogram (log counts)</h3>
                  <canvas className="chart" ref={histCanvas} />
                  <h3>logit lens (direct path)</h3>
                  <div className="row">
                    <table className="lens">
                      <tbody>
                        {detail.logit_lens.promoted.slice(0, 8).map(([t, v]) => (
                          <tr key={t + v}>
                            <td style={{ color: "#5eead4" }}>{JSON.stringify(t)}</td>
                            <td className="val">+{v.toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <table className="lens">
                      <tbody>
                        {detail.logit_lens.suppressed.slice(0, 8).map(([t, v]) => (
                          <tr key={t + v}>
                            <td style={{ color: "#f0abfc" }}>{JSON.stringify(t)}</td>
                            <td className="val">-{v.toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="note">
                    Approximation: decoder direction times unembedding, ignoring the layers after
                    the hook. See METHODS.md.
                  </p>
                </div>
              </div>

              <h3>probe this feature live {engine ? "" : "(engine not loaded)"}</h3>
              <div className="controls">
                <textarea
                  className="prompt"
                  value={probeText}
                  onChange={(e) => setProbeText(e.target.value)}
                  style={{ flex: 1, minHeight: 44 }}
                />
                <button disabled={!engine} onClick={runProbe}>
                  run
                </button>
              </div>
              {probeActs && (
                <div className="tokens">
                  {probeActs.map((p, i) => {
                    const max = Math.max(...probeActs.map((q) => q.act), 1e-6);
                    const a = p.act / max;
                    return (
                      <span
                        key={i}
                        style={{ background: a > 0 ? `rgba(94,234,212,${a * 0.55})` : "transparent" }}
                        title={`act ${p.act.toFixed(3)}`}
                      >
                        {p.tok}
                      </span>
                    );
                  })}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </>
  );
}
