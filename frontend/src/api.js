const API_BASE = import.meta.env.VITE_API_BASE || "";

export async function fetchCatalogs() {
  const res = await fetch(`${API_BASE}/api/catalogs`);
  if (!res.ok) throw new Error("Failed to load catalogs");
  return res.json();
}

export async function searchQuestions(body) {
  const res = await fetch(`${API_BASE}/api/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Search failed");
  }
  return res.json();
}
