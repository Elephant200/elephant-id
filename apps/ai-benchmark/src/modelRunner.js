import { formatBytes, summarizeSamples } from './stats.js';

const DEFAULT_WARMUPS = 2;
const DEFAULT_SAMPLES = 7;
const MODEL_CACHE = 'ai-benchmark-models-v1';

export async function runModelModule({
  model,
  manifest,
  signal,
  onProgress,
  backend: backendPreference = 'auto',
}) {
  const startedAt = performance.now();
  const download = await downloadModel(model, signal, onProgress);
  const backendResult = await createSessionForBackend({
    manifest,
    modelBytes: download.buffer,
    signal,
    onProgress,
    preference: backendPreference,
  });
  const { session, backend, ort, warnings } = backendResult;
  const inputName = session.inputNames[0];
  const inputShape = resolveInputShape(session, inputName, model.inputShape);
  const input = createSyntheticInput(ort, inputShape);
  const feeds = { [inputName]: input.tensor };
  const samples = [];

  onProgress(`Warmup: ${model.name} on ${backend}`);
  for (let i = 0; i < DEFAULT_WARMUPS; i += 1) {
    throwIfAborted(signal);
    await session.run(feeds);
    await tick();
  }

  for (let i = 0; i < DEFAULT_SAMPLES; i += 1) {
    throwIfAborted(signal);
    onProgress(`Sample ${i + 1}/${DEFAULT_SAMPLES}: ${model.name} on ${backend}`);
    const start = performance.now();
    await session.run(feeds);
    samples.push(performance.now() - start);
    await tick();
  }

  const outputNames = [...session.outputNames];
  releaseSession(session);
  return {
    id: model.id,
    name: model.name,
    task: model.task,
    status: 'complete',
    backend,
    startedAt,
    endedAt: performance.now(),
    modelBytes: model.bytes,
    bytesDownloaded: download.bytesDownloaded,
    downloadMs: download.downloadMs,
    hashVerified: download.hashVerified,
    hashStatus: download.hashStatus,
    sessionCreateMs: backendResult.sessionCreateMs,
    inputShape,
    outputNames,
    samples,
    stats: summarizeSamples(samples),
    warnings,
  };
}

async function createSessionForBackend({
  manifest,
  modelBytes,
  signal,
  onProgress,
  preference,
}) {
  const warnings = [];
  const webgpuRuntime = manifest.runtime?.webgpu;
  const webgpuUsable = Boolean(navigator.gpu) && Boolean(webgpuRuntime?.wasmPaths);

  if (preference === 'webgpu' && !webgpuUsable) {
    throw new Error('WebGPU was requested but is not available in this browser.');
  }
  if (preference !== 'wasm' && webgpuUsable) {
    try {
      onProgress('Trying GPU (WebGPU) backend');
      return await createSession({
        runtime: webgpuRuntime,
        backend: 'webgpu',
        modelBytes,
        signal,
        warnings,
      });
    } catch (error) {
      if (preference === 'webgpu') throw error;
      warnings.push(`WebGPU unavailable for this model: ${error.message}`);
    }
  }

  const wasmRuntime = manifest.runtime?.wasm;
  if (!wasmRuntime?.wasmPaths) {
    throw new Error('Manifest does not define a WASM ONNX Runtime wasmPaths URL.');
  }
  onProgress('Using CPU (WASM) backend');
  return createSession({
    runtime: wasmRuntime,
    backend: 'wasm',
    modelBytes,
    signal,
    warnings,
  });
}

async function createSession({ runtime, backend, modelBytes, signal, warnings }) {
  throwIfAborted(signal);
  const ort = await loadOrt(backend);
  configureOrt(ort, runtime);
  const start = performance.now();
  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: [backend],
    graphOptimizationLevel: 'all',
  });
  return {
    ort,
    session,
    backend,
    warnings,
    sessionCreateMs: performance.now() - start,
  };
}

async function loadOrt(backend) {
  if (backend === 'webgpu') {
    return import('onnxruntime-web/webgpu');
  }
  return import('onnxruntime-web/wasm');
}

function configureOrt(ort, runtime) {
  if (ort.env?.wasm) {
    if (runtime.wasmPaths) ort.env.wasm.wasmPaths = runtime.wasmPaths;
    ort.env.wasm.numThreads = window.crossOriginIsolated
      ? Math.min(4, navigator.hardwareConcurrency || 1)
      : 1;
    ort.env.wasm.simd = true;
  }
}

async function downloadModel(model, signal, onProgress) {
  const start = performance.now();
  const cacheKey = cacheRequest(model);
  const cache = await openCache();
  const cached = await cache?.match(cacheKey);
  if (cached) {
    onProgress(`Loading ${model.name} from browser cache`);
    const buffer = await cached.arrayBuffer();
    return {
      buffer,
      bytesDownloaded: 0,
      downloadMs: performance.now() - start,
      hashVerified: true,
      hashStatus: 'browser cache',
    };
  }

  onProgress(`Downloading ${model.name} (${formatBytes(model.bytes)})`);
  const response = await fetch(model.url, { signal, mode: 'cors' });
  if (!response.ok) {
    throw new Error(`Download failed with HTTP ${response.status}`);
  }
  const buffer = await response.arrayBuffer();
  const bytesDownloaded = buffer.byteLength;
  const downloadMs = performance.now() - start;
  if (model.bytes && model.bytes !== bytesDownloaded) {
    throw new Error(
      `Downloaded ${bytesDownloaded} bytes, expected ${model.bytes} bytes.`,
    );
  }
  const hashStatus = await verifyHash(buffer, model.sha256);
  await cache?.put(cacheKey, new Response(buffer.slice(0)));
  return {
    buffer,
    bytesDownloaded,
    downloadMs,
    hashVerified: hashStatus === 'verified',
    hashStatus,
  };
}

function cacheRequest(model) {
  const pin = model.sha256?.startsWith('replace-with') ? 'unpinned' : model.sha256;
  return new Request(
    `/__model-cache__/${encodeURIComponent(`${model.id}-${model.bytes}-${pin}`)}`,
  );
}

async function openCache() {
  if (!('caches' in window)) return null;
  try {
    return await caches.open(MODEL_CACHE);
  } catch {
    return null;
  }
}

async function verifyHash(buffer, sha256) {
  if (!sha256 || sha256.startsWith('replace-with')) return 'not pinned';
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  const hex = [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
  if (hex !== sha256.toLowerCase()) {
    throw new Error('Downloaded model hash did not match manifest SHA-256.');
  }
  return 'verified';
}

function createSyntheticInput(ort, shape) {
  if (!Array.isArray(shape) || shape.length < 2) {
    throw new Error('Model inputShape must be an array.');
  }
  const length = shape.reduce((product, value) => product * value, 1);
  const data = new Float32Array(length);
  for (let i = 0; i < data.length; i += 1) {
    data[i] = (((i * 17) % 255) / 255 - 0.5) * 2;
  }
  return {
    data,
    tensor: new ort.Tensor('float32', data, shape),
  };
}

function resolveInputShape(session, inputName, fallback) {
  const meta = session.inputMetadata?.find((entry) => entry.name === inputName);
  const modelShape = meta?.isTensor ? meta.shape : null;
  if (!Array.isArray(modelShape) || !modelShape.length) {
    return requireStaticShape(fallback, inputName);
  }
  // The model is the source of truth. Fixed dims come straight from it (so we
  // feed a correctly-sized tensor even when the manifest inputShape is stale);
  // symbolic dims (batch, dynamic H/W) fall back to the manifest per index.
  return modelShape.map((dim, index) => {
    if (typeof dim === 'number' && Number.isInteger(dim) && dim > 0) return dim;
    const fromFallback = fallback?.[index];
    if (Number.isInteger(fromFallback) && fromFallback > 0) return fromFallback;
    throw new Error(
      `Input "${inputName}" dimension ${index} is dynamic (${dim}) with no manifest fallback.`,
    );
  });
}

function requireStaticShape(fallback, inputName) {
  if (
    Array.isArray(fallback) &&
    fallback.length >= 2 &&
    fallback.every((dim) => Number.isInteger(dim) && dim > 0)
  ) {
    return fallback;
  }
  throw new Error(`No usable input shape for "${inputName}" from model or manifest.`);
}

function releaseSession(session) {
  if (typeof session.release === 'function') {
    session.release();
  }
}

function throwIfAborted(signal) {
  if (signal?.aborted) {
    throw new DOMException('Benchmark canceled', 'AbortError');
  }
}

function tick() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}
