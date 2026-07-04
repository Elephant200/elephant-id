export async function collectDeviceInfo() {
  const uaData = navigator.userAgentData;
  const adapter = await getGpuAdapterInfo();
  const memory = performance.memory
    ? {
        jsHeapSizeLimit: performance.memory.jsHeapSizeLimit,
        totalJSHeapSize: performance.memory.totalJSHeapSize,
        usedJSHeapSize: performance.memory.usedJSHeapSize,
      }
    : null;

  return {
    timestamp: new Date().toISOString(),
    userAgent: navigator.userAgent,
    brands: uaData?.brands ?? [],
    platform: uaData?.platform ?? navigator.platform,
    mobile: uaData?.mobile ?? /Mobi|Android/i.test(navigator.userAgent),
    language: navigator.language,
    logicalCpuCores: navigator.hardwareConcurrency ?? null,
    deviceMemoryGb: navigator.deviceMemory ?? null,
    crossOriginIsolated: window.crossOriginIsolated,
    webAssembly: typeof WebAssembly !== 'undefined',
    workers: typeof Worker !== 'undefined',
    webGpuAvailable: Boolean(navigator.gpu),
    gpuAdapter: adapter,
    screen: {
      width: window.screen.width,
      height: window.screen.height,
      devicePixelRatio: window.devicePixelRatio,
    },
    performanceMemory: memory,
  };
}

async function getGpuAdapterInfo() {
  if (!navigator.gpu) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { available: false };
    const info =
      adapter.info ??
      (typeof adapter.requestAdapterInfo === 'function'
        ? await adapter.requestAdapterInfo()
        : null);
    return {
      available: true,
      vendor: info?.vendor ?? 'not exposed',
      architecture: info?.architecture ?? 'not exposed',
      device: info?.device ?? 'not exposed',
      description: info?.description ?? 'not exposed',
      features: Array.from(adapter.features ?? []),
      limits: adapter.limits
        ? {
            maxBufferSize: adapter.limits.maxBufferSize,
            maxComputeInvocationsPerWorkgroup:
              adapter.limits.maxComputeInvocationsPerWorkgroup,
            maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
          }
        : null,
    };
  } catch (error) {
    return { available: false, error: error.message };
  }
}
