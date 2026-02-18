const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function createJob(payload) {
  const res = await fetch(`${API_BASE}/api/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getJob(jobId) {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function cancelJob(jobId) {
  // ✅ use same base + consistent /api prefix
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}/cancel`, {
    method: "POST",
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`Cancel failed (${res.status}): ${txt}`);
  }
  return res.json().catch(() => ({}));
}
