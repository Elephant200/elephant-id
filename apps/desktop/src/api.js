// Thin client for the Alphaphant sidecar API.

export const API_BASE =
  (typeof window !== 'undefined' && window.alphaphant?.apiBase) ||
  'http://127.0.0.1:8756';

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status}`;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return response.json();
}

export const getHealth = () => request('/health');
export const getCatalog = () => request('/catalog');
export const getElephant = (name) => request(`/catalog/${encodeURIComponent(name)}`);
export const listSightings = () => request('/sightings');
export const getSighting = (id) => request(`/sightings/${encodeURIComponent(id)}`);
export const getAnalysis = (id) => request(`/sightings/${encodeURIComponent(id)}/analysis`);
export const createSighting = (folder) =>
  request('/sightings', { method: 'POST', body: JSON.stringify({ folder }) });
export const approveEvidence = (id, leftCandidateId, rightCandidateId) =>
  request(`/sightings/${encodeURIComponent(id)}/approve-evidence`, {
    method: 'POST',
    body: JSON.stringify({
      left_candidate_id: leftCandidateId,
      right_candidate_id: rightCandidateId,
    }),
  });
export const matchSighting = (id, topN = 8) =>
  request(`/sightings/${encodeURIComponent(id)}/match`, {
    method: 'POST',
    body: JSON.stringify({ top_n: topN }),
  });
export const decideSighting = (id, action, elephantName = null) =>
  request(`/sightings/${encodeURIComponent(id)}/decision`, {
    method: 'POST',
    body: JSON.stringify({ action, elephant_name: elephantName }),
  });

async function uploadDevImage(path, file) {
  const body = new FormData();
  body.append('file', file);
  const response = await fetch(`${API_BASE}${path}`, { method: 'POST', body });
  if (!response.ok) {
    let detail = `${response.status}`;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return response.json();
}

async function uploadDevImagePair(path, fileA, fileB) {
  const body = new FormData();
  body.append('file_a', fileA);
  body.append('file_b', fileB);
  const response = await fetch(`${API_BASE}${path}`, { method: 'POST', body });
  if (!response.ok) {
    let detail = `${response.status}`;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return response.json();
}

export const devAnalyze = (file) => uploadDevImage('/dev/analyze', file);
export const devSam3 = (file) => uploadDevImage('/dev/sam3', file);
export const devTearProfile = (file) => uploadDevImage('/dev/tear-profile', file);
export const devTearMatch = (fileA, fileB) =>
  uploadDevImagePair('/dev/tear-match', fileA, fileB);
export const devPhotoAnalysis = (file) => uploadDevImage('/dev/photo-analysis', file);

export const imageUrl = (path) =>
  path ? `${API_BASE}/image?path=${encodeURIComponent(path)}` : null;

export const selectFolder = () =>
  window.alphaphant?.selectFolder ? window.alphaphant.selectFolder() : Promise.resolve(null);

export const hasNativeFolderPicker = () =>
  typeof window !== 'undefined' && Boolean(window.alphaphant?.selectFolder);
