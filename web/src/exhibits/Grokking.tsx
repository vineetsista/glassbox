import { useEffect, useRef, useState } from "react";
import { drawBars, drawHeatmap, drawLineChart } from "../charts";
import { embCosSim, parseGrokpack, type GrokPack } from "../grokpack";

const ASSET = "assets/grok.grokpack";

export default function Grokking() {
  const [pack, setPack] = useState<GrokPack | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);

  const accCanvas = useRef<HTMLCanvasElement>(null);
  const lossCanvas = useRef<HTMLCanvasElement>(null);
  const dftCanvas = useRef<HTMLCanvasElement>(null);
  const simCanvas = useRef<HTMLCanvasElement>(null);
  const attnCanvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    fetch(ASSET)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} - export not built yet?`);
        return r.arrayBuffer();
      })
      .then((buf) => {
        const p = parseGrokpack(buf);
        setPack(p);
        setIdx(p.ckpts.length - 1);
      })
      .catch((e) => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (!playing || !pack) return;
    const t = setInterval(() => {
      setIdx((i) => {
        if (i + 1 >= pack.ckpts.length) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, 60);
    return () => clearInterval(t);
  }, [playing, pack]);

  useEffect(() => {
    if (!pack) return;
    const ck = pack.ckpts[idx];
    const marker = ck.step;

    if (accCanvas.current)
      drawLineChart(
        accCanvas.current,
        [
          {
            label: "train acc",
            color: "#5eead4",
            points: pack.ckpts.map((c) => [c.step, c.metrics.train_acc]),
          },
          {
            label: "test acc",
            color: "#f0abfc",
            points: pack.ckpts.map((c) => [c.step, c.metrics.test_acc]),
          },
        ],
        { w: 520, h: 200, marker, xLabel: "step", yMin: 0, yMax: 1 },
      );
    if (lossCanvas.current)
      drawLineChart(
        lossCanvas.current,
        [
          {
            label: "test loss",
            color: "#f0abfc",
            points: pack.ckpts.map((c) => [c.step, Math.max(c.metrics.full_test, 1e-6)]),
          },
          {
            label: "restricted (key freqs only)",
            color: "#5eead4",
            points: pack.ckpts.map((c) => [c.step, Math.max(c.metrics.restricted_test, 1e-6)]),
          },
          {
            label: "excluded (key freqs removed)",
            color: "#fbbf24",
            points: pack.ckpts.map((c) => [c.step, Math.max(c.metrics.excluded_train, 1e-6)]),
          },
        ],
        { w: 520, h: 200, logY: true, marker, xLabel: "step" },
      );
    if (dftCanvas.current)
      drawBars(dftCanvas.current, Array.from(ck.embDft), {
        w: 520,
        h: 140,
        highlight: new Set(pack.header.key_freqs),
      });
    if (simCanvas.current) {
      const n = 56;
      const sim = embCosSim(ck, pack.header.p, pack.header.d_model, n);
      const m = Math.round(Math.sqrt(sim.length));
      drawHeatmap(simCanvas.current, sim, m, m, { w: 260, h: 260, symmetric: true });
    }
    if (attnCanvas.current) {
      // probe attention: rows = probes, cols = heads x 3 positions
      const { n_probe, n_heads } = pack.header;
      drawHeatmap(attnCanvas.current, ck.attnProbe, n_probe, n_heads * 3, {
        w: 260,
        h: 260,
      });
    }
  }, [pack, idx]);

  if (err)
    return (
      <div className="panel">
        <h2>Watch it grok</h2>
        <p className="err">could not load {ASSET}: {err}</p>
        <p className="note">
          Build the export first: <code>python -m grok.export_web</code> then{" "}
          <code>scripts/build_site.sh</code>.
        </p>
      </div>
    );
  if (!pack) return <div className="loading">loading grokking run...</div>;

  const ck = pack.ckpts[idx];
  const m = ck.metrics;
  const phase =
    m.test_acc > 0.99 ? "generalized (grokked)" : m.train_acc > 0.99 ? "memorized, not yet general" : "learning the training set";

  return (
    <>
      <div className="panel">
        <h2>Watch it grok</h2>
        <p className="note">
          A 1-layer transformer trained on <code>(a + b) mod 113</code> from only{" "}
          {Math.round(pack.header.train_frac * 100)}% of all pairs, checkpointed every 100 steps.
          It first memorizes (train accuracy hits 100% while test stays near 0), then - thousands
          of steps later - <em>grokks</em>: test accuracy snaps to 100% as the memorization
          circuit is replaced by trig identities on {pack.header.key_freqs.length} key Fourier
          frequencies. Drag the slider.
        </p>
        <div className="controls">
          <button onClick={() => setPlaying(!playing)}>{playing ? "pause" : "play"}</button>
          <input
            type="range"
            min={0}
            max={pack.ckpts.length - 1}
            value={idx}
            onChange={(e) => setIdx(Number(e.target.value))}
            style={{ flex: 1 }}
          />
        </div>
        <div>
          <span className="badge">step {ck.step}</span>
          <span className="badge">train acc {(m.train_acc * 100).toFixed(1)}%</span>
          <span className="badge">test acc {(m.test_acc * 100).toFixed(1)}%</span>
          <span className="badge">weight norm {m.wnorm.toFixed(0)}</span>
          <span className="badge" style={{ color: "#5eead4" }}>
            {phase}
          </span>
        </div>
      </div>

      <div className="row">
        <div className="panel">
          <h3>accuracy</h3>
          <canvas className="chart" ref={accCanvas} />
        </div>
        <div className="panel">
          <h3>progress measures (log)</h3>
          <canvas className="chart" ref={lossCanvas} />
          <p className="note">
            The restricted loss (only the key-frequency components of the logits) starts tracking
            the full test loss BEFORE test accuracy moves: the periodic circuit grows quietly
            inside a memorizing network.
          </p>
        </div>
      </div>

      <div className="row">
        <div className="panel">
          <h3>embedding spectrum (DFT over token index)</h3>
          <canvas className="chart" ref={dftCanvas} />
          <p className="note">
            Each bar is one frequency; pink bars are the key frequencies of the final model. Watch
            the spectrum collapse from noise onto a few spikes as it grokks.
          </p>
        </div>
        <div className="panel">
          <h3>embedding cosine similarity / attention probe</h3>
          <div style={{ display: "flex", gap: 12 }}>
            <canvas className="chart" ref={simCanvas} style={{ maxWidth: 260 }} />
            <canvas className="chart" ref={attnCanvas} style={{ maxWidth: 260 }} />
          </div>
          <p className="note">
            Left: cos-sim between number embeddings (periodic bands appear post-grok). Right: how
            the '=' position attends to a and b on {pack.header.n_probe} fixed probe pairs (rows),
            per head.
          </p>
        </div>
      </div>
    </>
  );
}
