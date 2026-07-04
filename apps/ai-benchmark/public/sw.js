// Offline support for the benchmark. Two goals, both honoring the rule that
// nothing large downloads unless the user starts a run:
//   1. App shell works offline, but shows the latest deploy when online — so
//      the shell is network-first, falling back to cache only when offline.
//   2. Once a run has downloaded the ONNX Runtime, re-runs work fully offline —
//      the runtime is large and effectively immutable, so it is cache-first.
// Model weights are cached separately by modelRunner.js (Cache API), so this
// worker deliberately does NOT touch /models/ — no second copy of large files.

const SHELL_CACHE = 'ai-benchmark-shell-v2';
const RUNTIME_CACHE = 'ai-benchmark-runtime-v1';
const RUNTIME_ORIGIN = 'https://weights.benchmark.elephant-id.org';
const KEEP = new Set([SHELL_CACHE, RUNTIME_CACHE]);

self.addEventListener('install', () => self.skipWaiting());

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      const names = await caches.keys();
      await Promise.all(
        names
          .filter((name) => !KEEP.has(name) && !name.startsWith('ai-benchmark-models'))
          .map((name) => caches.delete(name)),
      );
      await self.clients.claim();
    })(),
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);

  // ONNX Runtime wasm/glue from R2: cache-first so it survives going offline.
  // Populated only when a run actually fetches it — never precached.
  if (url.origin === RUNTIME_ORIGIN && url.pathname.startsWith('/runtime/')) {
    event.respondWith(cacheFirst(request, RUNTIME_CACHE));
    return;
  }

  // App shell (same-origin): latest when online, cached copy when offline.
  if (url.origin === self.location.origin) {
    event.respondWith(networkFirst(request, SHELL_CACHE));
  }
});

async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (response.ok) await cache.put(request, response.clone());
  return response;
}

async function networkFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  try {
    const response = await fetch(request);
    if (response.ok) await cache.put(request, response.clone());
    return response;
  } catch (error) {
    const cached = await cache.match(request);
    if (cached) return cached;
    throw error;
  }
}
