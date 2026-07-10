import { useEffect, useRef, useState } from "react";
import { GlassboxEngine, type FeatureAct } from "../engine";

interface SteerSpec {
  feature: number;
  mult: number;
}
interface IndexEntry {
  id: number;
  top_tokens: string[];
  density: number;
}

export default function Steering() {
  const [engine, setEngine] = useState<GlassboxEngine | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("Once upon a time");
  const [temp, setTemp] = useState(0.8);
  const [seed, setSeed] = useState(1);
  const [maxNew, setMaxNew] = useState(120);
  const [steers, setSteers] = useState<SteerSpec[]>([]);
  const [featQuery, setFeatQuery] = useState("");
  const [featIndex, setFeatIndex] = useState<IndexEntry[]>([]);
  const [baseline, setBaseline] = useState("");
  const [steered, setSteered] = useState("");
  const [running, setRunning] = useState(false);
  const [liveFeats, setLiveFeats] = useState<FeatureAct[]>([]);
  const cancelRef = useRef(false);

  useEffect(() => {
    GlassboxEngine.load()
      .then(setEngine)
      .catch((e) => setErr(String(e)));
    fetch("assets/features/index.json")
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => j && setFeatIndex(j.features))
      .catch(() => {});
  }, []);

  async function generate(withSteering: boolean, onText: (s: string) => void): Promise<void> {
    if (!engine) return;
    engine.clearSteering();
    if (withSteering) for (const s of steers) engine.setSteering(s.feature, s.mult);
    engine.seed(seed);
    engine.reset();
    const ids = engine.tokenize(prompt).slice(0, engine.ctxLen() - 2);
    for (const id of ids) engine.feed(id);
    for (let i = 0; i < maxNew; i++) {
      if (cancelRef.current) break;
      const next = engine.sample(temp, 40);
      if (next < 0 || next === engine.eotId()) break;
      if (!engine.feed(next)) break;
      if (withSteering) setLiveFeats(engine.lastFeatures());
      onText(engine.text());
      // yield to the UI thread every few tokens
      if (i % 4 === 0) await new Promise((r) => setTimeout(r, 0));
    }
    engine.clearSteering();
  }

  async function run() {
    if (!engine || running) return;
    setRunning(true);
    cancelRef.current = false;
    setBaseline("");
    setSteered("");
    await generate(false, setBaseline);
    await generate(true, setSteered);
    setRunning(false);
  }

  function addSteer(id: number) {
    if (steers.some((s) => s.feature === id)) return;
    setSteers([...steers, { feature: id, mult: 4 }]);
  }

  const matches = featQuery
    ? featIndex
        .filter(
          (f) =>
            String(f.id) === featQuery ||
            f.top_tokens.some((t) => t.toLowerCase().includes(featQuery.toLowerCase())),
        )
        .slice(0, 8)
    : [];

  if (err)
    return (
      <div className="panel">
        <h2>Steer it</h2>
        <p className="err">engine failed to load: {err}</p>
        <p className="note">
          The WASM engine + model bundle are built by <code>scripts/build_site.sh</code> (needs a
          trained LM checkpoint).
        </p>
      </div>
    );

  return (
    <>
      <div className="panel">
        <h2>Steer it {engine ? "" : "(loading engine...)"}</h2>
        <p className="note">
          Same seed, same prompt, two generations: the left one is the model as trained; the
          right one has your feature edits applied to the residual stream mid-network
          (multiplier 0 = ablate the feature, &gt;1 = amplify it). The model is tiny and
          CPU-trained - expect simple stories, not literature. That is the honest artifact.
        </p>
        <textarea className="prompt" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
        <div className="controls">
          <label>temperature</label>
          <input type="number" step={0.1} min={0} max={2} value={temp} onChange={(e) => setTemp(Number(e.target.value))} style={{ width: 70 }} />
          <label>seed</label>
          <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} style={{ width: 90 }} />
          <label>tokens</label>
          <input type="number" value={maxNew} min={10} max={200} onChange={(e) => setMaxNew(Number(e.target.value))} style={{ width: 70 }} />
          <button className="primary" disabled={!engine || running} onClick={run}>
            {running ? "generating..." : "generate both"}
          </button>
          {running && (
            <button className="ghost" onClick={() => (cancelRef.current = true)}>
              stop
            </button>
          )}
        </div>

        <h3>feature edits</h3>
        <div className="controls">
          <input
            type="text"
            placeholder="search features by token or id..."
            value={featQuery}
            onChange={(e) => setFeatQuery(e.target.value)}
            style={{ flex: 1 }}
          />
        </div>
        {matches.length > 0 && (
          <div className="feature-grid" style={{ maxHeight: 130 }}>
            {matches.map((f) => (
              <div key={f.id} className="feature-card" onClick={() => addSteer(f.id)}>
                <div className="fid">#{f.id}</div>
                <div className="toks">{f.top_tokens.join(" ")}</div>
              </div>
            ))}
          </div>
        )}
        <div className="steer-list">
          {steers.map((s, i) => (
            <div className="steer-item" key={s.feature}>
              <span>#{s.feature}</span>
              <input
                type="range"
                min={0}
                max={10}
                step={0.5}
                value={s.mult}
                onChange={(e) => {
                  const next = [...steers];
                  next[i] = { ...s, mult: Number(e.target.value) };
                  setSteers(next);
                }}
              />
              <span style={{ width: 46 }}>{s.mult === 0 ? "ablate" : `x${s.mult}`}</span>
              <button className="ghost" onClick={() => setSteers(steers.filter((x) => x.feature !== s.feature))}>
                remove
              </button>
            </div>
          ))}
          {steers.length === 0 && <p className="note">no edits - search above and click a feature to add one</p>}
        </div>
      </div>

      <div className="row">
        <div className="panel">
          <h3>baseline</h3>
          <div className="gen-output">{baseline || " "}</div>
        </div>
        <div className="panel">
          <h3>steered ({steers.length} edit{steers.length === 1 ? "" : "s"})</h3>
          <div className="gen-output">{steered || " "}</div>
        </div>
      </div>

      {liveFeats.length > 0 && (
        <div className="panel">
          <h3>features active at the last generated token</h3>
          <div>
            {liveFeats
              .slice()
              .sort((a, b) => b.value - a.value)
              .slice(0, 16)
              .map((f) => (
                <span key={f.feature} className="badge">
                  #{f.feature} {f.value.toFixed(2)}
                </span>
              ))}
          </div>
        </div>
      )}
    </>
  );
}
