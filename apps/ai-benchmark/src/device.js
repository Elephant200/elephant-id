/** Collect as much device/browser detail as the platform will expose. */
export async function collectDeviceInfo() {
  const uaData = navigator.userAgentData;
  const [adapter, highEntropy, storage] = await Promise.all([
    getGpuAdapterInfo(),
    getHighEntropyUa(uaData),
    getStorageEstimate(),
  ]);
  return {
    userAgent: navigator.userAgent,
    browser: describeBrowser(highEntropy, uaData),
    platform: uaData?.platform ?? navigator.platform,
    platformVersion: highEntropy?.platformVersion ?? null,
    cpuArchitecture: highEntropy?.architecture
      ? `${highEntropy.architecture}${highEntropy.bitness ? ` ${highEntropy.bitness}-bit` : ''}`
      : null,
    deviceModel: highEntropy?.model || null,
    logicalCpuCores: navigator.hardwareConcurrency ?? null,
    deviceMemoryGb: navigator.deviceMemory ?? null,
    storage,
    network: getNetworkInfo(),
    crossOriginIsolated: window.crossOriginIsolated,
    wasmFeatures: detectWasmFeatures(),
    webGpuAvailable: Boolean(navigator.gpu),
    gpuAdapter: adapter,
    gpuRenderer: getWebglRenderer(),
    screen: {
      width: window.screen.width,
      height: window.screen.height,
      devicePixelRatio: window.devicePixelRatio,
      colorDepth: window.screen.colorDepth,
    },
    performanceMemory: performance.memory
      ? { jsHeapSizeLimit: performance.memory.jsHeapSizeLimit }
      : null,
  };
}

async function getHighEntropyUa(uaData) {
  if (!uaData?.getHighEntropyValues) return null;
  try {
    return await uaData.getHighEntropyValues([
      'platformVersion',
      'architecture',
      'bitness',
      'model',
      'fullVersionList',
    ]);
  } catch {
    return null;
  }
}

function describeBrowser(highEntropy, uaData) {
  const list = highEntropy?.fullVersionList ?? uaData?.brands ?? [];
  const primary = list.find((entry) => !/Not.?A.?Brand/i.test(entry.brand));
  if (primary) return `${primary.brand} ${primary.version}`;
  return 'not exposed';
}

async function getStorageEstimate() {
  if (!navigator.storage?.estimate) return null;
  try {
    const { quota, usage } = await navigator.storage.estimate();
    return { quota: quota ?? null, usage: usage ?? null };
  } catch {
    return null;
  }
}

function getNetworkInfo() {
  const connection = navigator.connection;
  if (!connection) return null;
  return {
    effectiveType: connection.effectiveType ?? null,
    downlinkMbps: connection.downlink ?? null,
    rttMs: connection.rtt ?? null,
    saveData: connection.saveData ?? null,
  };
}

/** Minimal WASM feature probes; explains why the CPU backend is fast or slow. */
function detectWasmFeatures() {
  const test = (bytes) => {
    try {
      return WebAssembly.validate(new Uint8Array(bytes));
    } catch {
      return false;
    }
  };
  return {
    // SIMD: module using a v128 local.
    simd: test([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10,
      10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
    ]),
    // Threads: module declaring shared memory.
    threads: test([
      0, 97, 115, 109, 1, 0, 0, 0, 5, 4, 1, 3, 1, 1,
    ]),
    sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
  };
}

/** WebGL exposes the real GPU name, which WebGPU adapter.info often hides. */
function getWebglRenderer() {
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') ?? canvas.getContext('webgl');
    if (!gl) return null;
    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    const renderer = debugInfo
      ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL)
      : gl.getParameter(gl.RENDERER);
    const vendor = debugInfo
      ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL)
      : gl.getParameter(gl.VENDOR);
    return { renderer: renderer ?? null, vendor: vendor ?? null };
  } catch {
    return null;
  }
}

async function getGpuAdapterInfo() {
  if (!navigator.gpu) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });
    if (!adapter) return { available: false };
    const info =
      adapter.info ??
      (typeof adapter.requestAdapterInfo === 'function'
        ? await adapter.requestAdapterInfo()
        : null);
    return {
      available: true,
      isFallback: adapter.isFallbackAdapter ?? false,
      vendor: info?.vendor ?? 'not exposed',
      architecture: info?.architecture ?? 'not exposed',
      device: info?.device ?? 'not exposed',
      description: info?.description ?? 'not exposed',
      limits: adapter.limits
        ? {
            maxBufferSize: adapter.limits.maxBufferSize,
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            maxComputeInvocationsPerWorkgroup:
              adapter.limits.maxComputeInvocationsPerWorkgroup,
          }
        : null,
    };
  } catch (error) {
    return { available: false, error: error.message };
  }
}
