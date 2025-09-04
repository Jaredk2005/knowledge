export async function fetchAlerts(limit: number = 50) {
  const res = await fetch(`http://localhost:8000/api/alerts?limit=${limit}`);
  if (!res.ok) throw new Error('Failed to fetch alerts');
  const json = await res.json();
  return json.items ?? [];
}

export async function healthCheck() {
  const res = await fetch('http://localhost:8000/api/health');
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}
