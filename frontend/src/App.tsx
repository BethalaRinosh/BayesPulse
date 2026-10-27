import { useEffect, useMemo, useState, type FormEvent } from "react";
import {
  Activity, ArrowDownRight, ArrowUpRight, Beaker, ChevronDown,
  CircleHelp, FlaskConical, HeartPulse, Info, LoaderCircle, RotateCcw,
  ShieldCheck, Sparkles, Waves,
} from "lucide-react";

type CategoryOption = { value: number; label: string };
type Feature = {
  name: string;
  kind: "continuous" | "categorical";
  description: string;
  min?: number;
  max?: number;
  default: number;
  options?: CategoryOption[];
};
type Metadata = {
  name: string;
  dataset: { name: string; records: number };
  target: { class_0: string; class_1: string };
  features: Feature[];
  limitations: string[];
};
type Evidence = {
  feature: string;
  value: number | null;
  log_likelihood_ratio: number;
  evidence: "missing" | "toward_class_1" | "toward_class_0" | "neutral";
};
type Prediction = {
  predicted_class: number;
  probabilities: { class_0: number; class_1: number };
  prior_log_odds: number;
  total_log_odds: number;
  feature_evidence: Evidence[];
  updating_trace: { evidence_included: string; class_0: number; class_1: number }[];
  interpretation: string;
};

const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const prettyName: Record<string, string> = {
  age: "Age", trestbps: "Resting blood pressure", chol: "Serum cholesterol",
  thalach: "Maximum heart rate", oldpeak: "ST depression", sex: "Sex",
  cp: "Chest-pain type", fbs: "Fasting blood sugar", restecg: "Resting ECG",
  exang: "Exercise-induced angina", slope: "ST slope", ca: "Major vessels", thal: "Thalassemia",
};
const continuousUnits: Record<string, string> = {
  age: "years", trestbps: "mm Hg", chol: "mg/dL", thalach: "bpm", oldpeak: "mm",
};

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}
function signed(value: number) {
  return `${value > 0 ? "+" : ""}${value.toFixed(3)}`;
}

export default function App() {
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loadingMetadata, setLoadingMetadata] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState("");
  const [apiReady, setApiReady] = useState(false);
  const [showTrace, setShowTrace] = useState(false);

  useEffect(() => {
    let active = true;
    Promise.all([
      fetch(`${API_URL}/api/v1/metadata`),
      fetch(`${API_URL}/api/v1/health`),
    ])
      .then(async ([metadataResponse, healthResponse]) => {
        if (!metadataResponse.ok) throw new Error("Could not load the predictor schema.");
        const data = (await metadataResponse.json()) as Metadata;
        if (!active) return;
        setMetadata(data);
        setValues(Object.fromEntries(data.features.map((feature) => [feature.name, String(feature.default)])));
        setApiReady(healthResponse.ok);
      })
      .catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : "Could not connect to the BayesPulse API.");
      })
      .finally(() => {
        if (active) setLoadingMetadata(false);
      });
    return () => { active = false; };
  }, []);

  const filledCount = useMemo(
    () => metadata?.features.filter((feature) => values[feature.name]?.trim() !== "").length ?? 0,
    [metadata, values],
  );

  function updateValue(name: string, value: string) {
    setValues((current) => ({ ...current, [name]: value }));
  }

  function resetForm() {
    if (!metadata) return;
    setValues(Object.fromEntries(metadata.features.map((feature) => [feature.name, String(feature.default)])));
    setPrediction(null);
    setError("");
  }

  async function runPrediction(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!metadata) return;
    setPredicting(true);
    setError("");
    try {
      const features = Object.fromEntries(
        metadata.features.map((feature) => {
          const raw = values[feature.name]?.trim() ?? "";
          return [feature.name, raw === "" ? null : Number(raw)];
        }),
      );
      const response = await fetch(`${API_URL}/api/v1/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      });
      const body = await response.json();
      if (!response.ok) {
        const detail = typeof body.detail === "string" ? body.detail : "Please check the entered values.";
        throw new Error(detail);
      }
      setPrediction(body as Prediction);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Prediction request failed. Check that the API is running.");
    } finally {
      setPredicting(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="BayesPulse home">
          <span className="brand-mark"><Waves size={21} strokeWidth={2.3} /></span>
          <span>bayes<span className="brand-light">pulse</span><sup>LAB</sup></span>
        </a>
        <nav className="top-nav" aria-label="Main navigation">
          <a className="nav-active" href="#workspace">Workspace</a>
          <a href="#method">How it works</a>
        </nav>
        <div className="api-status">
          <span className={`status-dot ${apiReady ? "is-online" : "is-offline"}`} />
          <span>{apiReady ? "API connected" : "Connecting to API"}</span>
        </div>
      </header>

      <main>
        <section className="hero">
          <div className="hero-copy">
            <div className="eyebrow"><span className="eyebrow-line" /> PROBABILITY, MADE VISIBLE</div>
            <h1>See the probability.<br /><span>Understand the why.</span></h1>
            <p className="hero-description">
              Explore how a hybrid Naive Bayes model updates its belief as evidence arrives, one feature at a time.
            </p>
            <div className="hero-meta">
              <span><Beaker size={15} /> From-scratch hybrid Naive Bayes</span>
              <span className="meta-divider" />
              <span><Activity size={15} /> {metadata ? `${metadata.dataset.records} observations` : "Cleveland dataset"}</span>
            </div>
          </div>
          <div className="hero-visual" aria-hidden="true">
            <div className="orbit orbit-outer" />
            <div className="orbit orbit-inner" />
            <div className="visual-core"><Waves size={37} strokeWidth={1.5} /></div>
            <div className="visual-node node-a"><span className="node-dot" /> PRIOR</div>
            <div className="visual-node node-b"><span className="node-dot" /> EVIDENCE</div>
            <div className="visual-node node-c"><span className="node-dot" /> POSTERIOR</div>
            <div className="visual-caption">P(H | E) <span>←</span> P(E | H) · P(H)</div>
          </div>
        </section>

        <div className="workspace-heading" id="workspace">
          <div>
            <div className="section-kicker">THE SANDBOX</div>
            <h2>Build a scenario</h2>
            <p>Change the inputs and watch the model's belief respond.</p>
          </div>
          <div className="sample-chip"><span className="sample-chip-dot" /> EDUCATIONAL MODEL</div>
        </div>

        <div className="workspace-grid">
          <section className="panel input-panel">
            <div className="panel-heading">
              <div className="panel-icon"><FlaskConical size={18} /></div>
              <div><h3>Input features</h3><p>Dataset-coded values</p></div>
              <button type="button" className="icon-button reset-button" onClick={resetForm} title="Reset to dataset defaults" aria-label="Reset to dataset defaults"><RotateCcw size={16} /></button>
            </div>
            {loadingMetadata ? (
              <div className="loading-state"><LoaderCircle className="spin" size={21} /> Loading feature schema…</div>
            ) : metadata ? (
              <form onSubmit={runPrediction}>
                <div className="form-progress"><span>{filledCount} of {metadata.features.length} fields set</span><span className="progress-track"><i style={{ width: `${(filledCount / metadata.features.length) * 100}%` }} /></span></div>
                <div className="feature-grid">
                  {metadata.features.map((feature, index) => (
                    <label className={`field field-${feature.kind}`} key={feature.name}>
                      <span className="field-topline"><span>{prettyName[feature.name] ?? feature.name}</span><span className="field-index">{String(index + 1).padStart(2, "0")}</span></span>
                      {feature.kind === "categorical" ? (
                        <span className="select-wrap">
                          <select value={values[feature.name] ?? ""} onChange={(event) => updateValue(feature.name, event.target.value)}>
                            <option value="">Unknown / missing</option>
                            {feature.options?.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                          </select>
                          <ChevronDown size={15} />
                        </span>
                      ) : (
                        <span className="number-wrap">
                          <input
                            type="number"
                            step="any"
                            min={feature.min}
                            max={feature.max}
                            value={values[feature.name] ?? ""}
                            onChange={(event) => updateValue(feature.name, event.target.value)}
                            placeholder="Unknown"
                          />
                          <span className="field-unit">{continuousUnits[feature.name] ?? ""}</span>
                        </span>
                      )}
                      <span className="field-description">{feature.description}</span>
                    </label>
                  ))}
                </div>
                <div className="form-note"><Info size={15} /><span>Leave a field blank to treat that feature as missing. Numeric ranges reflect the dataset, not medical reference ranges.</span></div>
                <button className="primary-button" type="submit" disabled={predicting || loadingMetadata}>
                  {predicting ? <><LoaderCircle className="spin" size={17} /> Calculating posterior…</> : <><Sparkles size={17} /> Run probability analysis <ArrowUpRight size={16} /></>}
                </button>
              </form>
            ) : (
              <div className="error-state"><CircleHelp size={22} /><strong>API unavailable</strong><span>Start the FastAPI server, then refresh this page.</span></div>
            )}
          </section>

          <section className="panel results-panel" aria-live="polite">
            <div className="panel-heading">
              <div className="panel-icon panel-icon-teal"><Activity size={18} /></div>
              <div><h3>Posterior readout</h3><p>What the model believes, given these inputs</p></div>
              <span className="live-label"><span /> LIVE</span>
            </div>
            {!prediction ? (
              <div className="empty-results">
                <div className="empty-illustration">
                  <div className="empty-ring ring-one" /><div className="empty-ring ring-two" />
                  <div className="empty-center"><HeartPulse size={32} /></div>
                  <span className="floating-point point-one" /><span className="floating-point point-two" /><span className="floating-point point-three" />
                </div>
                <div className="empty-title">Your results will live here.</div>
                <p>Run an analysis to reveal the predicted class, class probabilities, and how each feature shifts the model's log-odds.</p>
                <div className="empty-hint"><Sparkles size={14} /> No prediction has been run yet</div>
              </div>
            ) : (
              <div className="result-content">
                <div className="result-label"><span className="result-pulse" /> MODEL PREDICTION</div>
                <div className="result-main">
                  <div><div className="result-class">{prediction.predicted_class === 1 ? "Class 1" : "Class 0"}</div><div className="result-subtitle">{prediction.predicted_class === 1 ? metadata?.target.class_1 : metadata?.target.class_0}</div></div>
                  <div className="result-percent">{percent(prediction.probabilities[`class_${prediction.predicted_class}` as "class_0" | "class_1"])}</div>
                </div>
                <div className="probability-block">
                  <div className="probability-heading"><span>Class probabilities</span><span>100% total</span></div>
                  <div className="stacked-bar"><div className="bar-class-zero" style={{ width: percent(prediction.probabilities.class_0) }} /><div className="bar-class-one" style={{ width: percent(prediction.probabilities.class_1) }} /></div>
                  <div className="probability-legend">
                    <div><span className="legend-dot dot-zero" /><span>Class 0</span><strong>{percent(prediction.probabilities.class_0)}</strong></div>
                    <div><span className="legend-dot dot-one" /><span>Class 1</span><strong>{percent(prediction.probabilities.class_1)}</strong></div>
                  </div>
                </div>
                <div className="odds-grid">
                  <div className="odds-card"><span>Prior log-odds</span><strong>{signed(prediction.prior_log_odds)}</strong><small>Before feature evidence</small></div>
                  <div className="odds-card"><span>Total log-odds</span><strong>{signed(prediction.total_log_odds)}</strong><small>After feature evidence</small></div>
                </div>
                <div className="evidence-section">
                  <div className="evidence-heading"><div><h4>Evidence by feature</h4><p>Log-likelihood ratio · class 1 relative to class 0</p></div><span>{prediction.feature_evidence.filter((item) => item.evidence !== "missing").length} used</span></div>
                  <div className="evidence-list">
                    {prediction.feature_evidence.filter((item) => item.evidence !== "missing").slice().sort((a, b) => Math.abs(b.log_likelihood_ratio) - Math.abs(a.log_likelihood_ratio)).map((item) => {
                      const positive = item.log_likelihood_ratio > 0;
                      const width = Math.min(100, Math.abs(item.log_likelihood_ratio) / Math.max(1, ...prediction.feature_evidence.map((entry) => Math.abs(entry.log_likelihood_ratio))) * 100);
                      return <div className="evidence-row" key={item.feature}>
                        <span className="evidence-name">{prettyName[item.feature] ?? item.feature}</span>
                        <div className="evidence-track"><i className={positive ? "evidence-positive" : "evidence-negative"} style={{ width: `${width}%` }} /></div>
                        <span className={`evidence-value ${positive ? "value-positive" : "value-negative"}`}>{signed(item.log_likelihood_ratio)}</span>
                      </div>;
                    })}
                  </div>
                  <div className="evidence-key"><span><ArrowUpRight size={13} /> Toward class 1</span><span><ArrowDownRight size={13} /> Toward class 0</span></div>
                </div>
                <button className="trace-toggle" type="button" onClick={() => setShowTrace((current) => !current)}>{showTrace ? "Hide posterior progression" : "Explore posterior progression"}<ChevronDown className={showTrace ? "chevron-open" : ""} size={16} /></button>
                {showTrace && <div className="trace-list">{prediction.updating_trace.map((step, index) => <div className="trace-row" key={`${step.evidence_included}-${index}`}><span className="trace-index">{String(index).padStart(2, "0")}</span><span>{step.evidence_included}</span><div className="trace-mini"><i style={{ width: percent(step.class_1) }} /></div><strong>{percent(step.class_1)}</strong></div>)}</div>}
                <p className="interpretation"><Info size={14} /> {prediction.interpretation}</p>
              </div>
            )}
          </section>
        </div>

        {error && <div className="toast-error" role="alert"><CircleHelp size={17} /><span>{error}</span><button type="button" onClick={() => setError("")} aria-label="Dismiss error">×</button></div>}

        <section className="method-strip" id="method">
          <div className="method-icon"><ShieldCheck size={20} /></div>
          <div className="method-copy"><strong>A model you can inspect, not just query.</strong><span>Hybrid Naive Bayes, transparent evidence contributions, and a step-by-step posterior trail.</span></div>
          <div className="method-tags"><span>NumPy</span><span>FastAPI</span><span>React</span></div>
        </section>
        <section className="disclaimer"><Info size={16} /><p><strong>Research and learning only.</strong> This is an educational demonstration using a historical Cleveland heart-disease dataset. It is not clinically validated, not a medical device, and must not be used for diagnosis or care decisions. Do not enter real patient information.</p></section>
      </main>
      <footer><a className="footer-brand" href="#">bayespulse<span>.</span></a><span>Probability, made visible.</span><span>Educational model · Not for clinical use</span></footer>
    </div>
  );
}
