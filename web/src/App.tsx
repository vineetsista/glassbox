import { useState } from "react";
import Grokking from "./exhibits/Grokking";
import Features from "./exhibits/Features";
import Steering from "./exhibits/Steering";

type Tab = "grokking" | "features" | "steering";

const TABS: { id: Tab; label: string; blurb: string }[] = [
  {
    id: "grokking",
    label: "1 - Watch it grok",
    blurb: "Scrub through 30k training steps of a 1-layer transformer learning modular addition.",
  },
  {
    id: "features",
    label: "2 - Look inside",
    blurb: "Browse the sparse-autoencoder features of a tiny story-telling transformer.",
  },
  {
    id: "steering",
    label: "3 - Steer it",
    blurb: "Generate stories in your browser and turn individual features up or down.",
  },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("grokking");
  return (
    <>
      <header className="site">
        <h1>
          GLASS<span>BOX</span>
        </h1>
        <div className="sub">
          an interpretability instrument - every model here was trained from scratch; inference
          runs in your browser via WASM
        </div>
      </header>
      <nav className="tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={tab === t.id ? "active" : ""}
            onClick={() => setTab(t.id)}
            title={t.blurb}
          >
            {t.label}
          </button>
        ))}
      </nav>
      {tab === "grokking" && <Grokking />}
      {tab === "features" && <Features />}
      {tab === "steering" && <Steering />}
      <footer className="site">
        GLASSBOX - from-scratch transformers, SAEs, and a C++/WASM engine. Honest by construction:
        every artifact regenerates from a script and a seed; limitations are documented in the
        repo. No data leaves your browser.
      </footer>
    </>
  );
}
