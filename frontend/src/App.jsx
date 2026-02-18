import React, { useState, useRef } from "react";
import { createJob, getJob, cancelJob } from "./api"; // <-- add cancelJob (optional)
import VtkViewer from "./components/VtkViewer";
import ErrorBoundary from "./components/ErrorBoundary";
import ThreePlyViewer from "./components/ThreePlyViewer";
import FallbackBox from "./components/FallbackBox";
import VtkProbe from "./components/VtkProbe";
import "./styles.css";

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [domain, setDomain] = useState("heat_transfer");
  const [parametersText, setParametersText] = useState("{}");

  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState("");
  

  // NEW: track current job + cancellation
  const currentJobIdRef = useRef(null);
  const abortRef = useRef(null);

  const isDone = response?.status === "completed";
  const artifactUrl =
    response?.result?.artifacts?.find((a) => a.name.endsWith(".vtp"))?.url ??
    response?.result?.artifacts?.[0]?.url;

  const API_BASE = "https://techchallenge3-s0p9.onrender.com"; // import.meta.env.VITE_API_BASE || "http://localhost:8000";
  const fullUrl = artifactUrl ? `${API_BASE}${artifactUrl}` : null;

  const defaultFieldForJob = (job) => {
    if (!job) return "temperature";
    if (job.result?.default_field) return job.result.default_field;
    if (job.domain === "solid_mechanics") return "u_mag";
    if (job.domain === "heat_transfer") return "temperature";
    if (Array.isArray(job.result?.fields) && job.result.fields.length) return job.result.fields[0];
    if (job.result?.field) return job.result.field;
    return "temperature";
  };

  const fieldName = defaultFieldForJob(response);
  const artifactSubtitle = fullUrl ? `Artifact: ${artifactUrl}` : "No artifact yet";

  React.useEffect(() => {
    const onErr = (e) => {
      const msg = e?.message || String(e);
      document.body.setAttribute("data-global-error", msg);
      console.error("window.onerror:", e);
    };
    const onRej = (e) => {
      const msg = e?.reason?.message || String(e?.reason || e);
      document.body.setAttribute("data-global-error", msg);
      console.error("unhandledrejection:", e);
    };
    window.addEventListener("error", onErr);
    window.addEventListener("unhandledrejection", onRej);
    return () => {
      window.removeEventListener("error", onErr);
      window.removeEventListener("unhandledrejection", onRej);
    };
  }, []);

  async function onRun() {
    if (!prompt || !prompt.trim()) {
      setError("Please introduce a prompt");
      return;
    }

    // If user clicks Run while another run is active, cancel the previous polling.
    if (loading) return;

    setLoading(true);
    setError("");
    setResponse(null);

    let parameters = {};
    try {
      parameters = parametersText.trim() ? JSON.parse(parametersText) : {};
    } catch {
      setLoading(false);
      setError("Parameters must be valid JSON.");
      return;
    }

    // NEW: create an abort controller for this run/polling session
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const { job_id } = await createJob({ prompt, domain, parameters });
      currentJobIdRef.current = job_id;

      // Poll until completed/failed OR cancelled
      while (true) {
        // Stop immediately if cancelled
        if (controller.signal.aborted) {
          setResponse((prev) => ({
            ...(prev || {}),
            id: job_id,
            status: "cancelled",
            message: "Cancelled by user (polling stopped).",
            progress: prev?.progress ?? 0,
          }));
          break;
        }

        const job = await getJob(job_id);
        setResponse(job);

        if (job.status === "completed" || job.status === "failed" || job.status === "cancelled") {
          break;
        }

        await new Promise((r) => setTimeout(r, 400));
      }
    } catch (e) {
      // If abort triggered during await, treat as cancelled
      if (abortRef.current?.signal?.aborted) {
        setResponse((prev) => ({
          ...(prev || {}),
          status: "cancelled",
          message: "Cancelled by user.",
        }));
      } else {
        setError(e.message || "Unknown error");
      }
    } finally {
      setLoading(false);
      // clear refs if this was the active run
      abortRef.current = null;
      currentJobIdRef.current = null;
    }
  }

  // NEW: cancel handler
  async function onCancel() {
    // stop UI polling immediately
    if (abortRef.current) abortRef.current.abort();

    const jobId = currentJobIdRef.current;
    if (!jobId) return;

    // optional: tell backend to cancel the actual simulation
    try {
      if (typeof cancelJob === "function") {
        await cancelJob(jobId);
      }
    } catch (e) {
      // Not fatal; UI is already stopped.
      console.warn("cancelJob failed (backend may not support it):", e);
    }
  }

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1>KronosAI TechChallenge: Physics Simulator</h1>
          <p className="sub">Agentic finite-element simulations.</p>
        </div>
      </header>

      {typeof document !== "undefined" && document.body?.getAttribute("data-global-error") && (
        <div className="alert error" style={{ whiteSpace: "pre-wrap" }}>
          Global error: {document.body.getAttribute("data-global-error")}
        </div>
      )}

      <main className="grid">
        <section className="card">
          <h2>Prompt</h2>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={5}
            placeholder="e.g., Simulate the heat transfer through a thin 30 nm conductor."
          />

          <div className="row">
            <label>
              Domain
              <select value={domain} onChange={(e) => setDomain(e.target.value)}>
                <option value="heat_transfer">heat_transfer</option>
                <option value="solid_mechanics">solid_mechanics</option>
                <option value="fluid">fluid</option>
              </select>
            </label>

            <div style={{ display: "flex", gap: 10 }}>
              <button className="btn primary" onClick={onRun} disabled={loading}>
                {loading ? "Running..." : "Run simulation"}
              </button>

              {/* NEW: Cancel button */}
              <button className="btn" onClick={onCancel} disabled={!loading}>
                Cancel
              </button>
            </div>
          </div>

          {error && <div className="alert error">{error}</div>}
        </section>

        <section className="card">
          <h2>Response</h2>
          {response && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 13, color: "#a7b0c0", marginBottom: 6 }}>
                Status: <b style={{ color: "white" }}>{response.status}</b> — {response.message}
              </div>

              <div
                style={{
                  height: 10,
                  borderRadius: 999,
                  overflow: "hidden",
                  border: "1px solid rgba(255,255,255,0.12)",
                  background: "rgba(0,0,0,0.25)",
                }}
              >
                <div
                  style={{
                    width: `${Math.round((response.progress || 0) * 100)}%`,
                    height: "100%",
                    background: "rgba(255,213,74,0.95)",
                    transition: "width 150ms linear",
                  }}
                />
              </div>
            </div>
          )}

          <pre className="code">{response ? JSON.stringify(response, null, 2) : "No response yet."}</pre>

          <h3 style={{ marginTop: 16 }}>3D Viewer</h3>
          {fullUrl ? (
            <VtkViewer url={fullUrl} fieldName={fieldName} />
          ) : (
            <FallbackBox subtitle={artifactSubtitle} />
          )}
        </section>
      </main>

      <footer className="footer">
        <span>Joan Vendrell Gallart</span>
      </footer>
    </div>
  );
}
