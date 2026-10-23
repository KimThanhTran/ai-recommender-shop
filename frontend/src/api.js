// `/api` is proxied to the FastAPI service in development (vite.config.js) and by nginx in Docker.
const BASE = import.meta.env.VITE_API_BASE ?? '/api';

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function get(path, params, signal) {
  const query = params ? `?${new URLSearchParams(params)}` : '';
  let response;
  try {
    response = await fetch(`${BASE}${path}${query}`, { signal });
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new ApiError(0, 'Cannot reach the API');
  }
  if (!response.ok) {
    let detail = response.statusText || 'Request failed';
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      // body was not JSON: keep the status text
    }
    throw new ApiError(response.status, detail);
  }
  return response.json();
}

export const api = {
  health: (signal) => get('/health', null, signal),
  recommend: (userId, { k, model }, signal) => get(`/recommend/${userId}`, { k, model }, signal),
  history: (userId, { limit }, signal) => get(`/users/${userId}/history`, { limit }, signal),
  similar: (itemId, { k, model }, signal) => get(`/items/${itemId}/similar`, { k, model }, signal),
  popular: ({ k }, signal) => get('/popular', { k }, signal),
};
