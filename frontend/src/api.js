const API_BASE = import.meta.env.VITE_API_BASE || "";

export async function fetchCatalogs() {
  const res = await fetch(`${API_BASE}/api/catalogs`);
  if (!res.ok) throw new Error("Failed to load catalogs");
  return res.json();
}

function formatApiDetail(detail) {
  if (!detail) return "";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => (typeof item === "string" ? item : item?.msg || JSON.stringify(item)))
      .join("; ");
  }
  try {
    return JSON.stringify(detail);
  } catch {
    return String(detail);
  }
}

export async function searchQuestions(body) {
  let res;
  try {
    res = await fetch(`${API_BASE}/api/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    throw new Error(
      `Cannot reach API (${err.message}). Is the backend running on port 8000?`
    );
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const detail = formatApiDetail(err.detail);
    throw new Error(detail || `Search failed (HTTP ${res.status})`);
  }
  return res.json();
}
