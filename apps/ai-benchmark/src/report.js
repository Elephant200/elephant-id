import { formatBytes, formatMs } from './stats.js';
import { gradingLegend, rateModelFps, rateWorkload, TIERS, worstTier } from './rating.js';

/**
 * Graded summary for the UI: an overall verdict tier, per-model grades, a
 * compute-hardware breakdown, and key system facts. The full numbers live in
 * buildEmailReport().
 */
export function buildSummary({ device, modules, status }) {
  const completed = modules.filter((module) => module.status === 'complete');
  const failed = modules.filter((module) => module.status === 'failed');
  const models = completed
    .filter((module) => module.samples?.length)
    .map((module) => ({
      name: module.name,
      backendLabel: backendLabel(module.backend),
      fps: module.stats.fpsP50,
      latencyMs: module.stats.p50,
      tier: rateModelFps(module.stats.fpsP50),
    }));
  const hardware = completed.find((module) => module.id === 'general-hardware');

  return {
    verdict: summaryVerdict({ models, failed, canceled: status === 'canceled' }),
    models,
    compute: computeBreakdown(hardware),
    system: systemFacts(device),
    legend: gradingLegend(),
  };
}

const VERDICT_COPY = {
  excellent: {
    label: 'Excellent for local AI',
    detail: 'Every model ran at real-time speed with room to spare.',
  },
  great: {
    label: 'Great for local AI',
    detail: 'Models run at real-time speed on this device.',
  },
  good: {
    label: 'Good for local AI',
    detail: 'Models run at interactive speed — comfortable for review.',
  },
  acceptable: {
    label: 'Acceptable for local AI',
    detail: 'Models run, but the slowest ones are better suited to batch review.',
  },
  poor: {
    label: 'Struggles with local AI',
    detail: 'At least one model is too slow for a smooth workflow here.',
  },
};

const TONE_BY_TIER = {
  excellent: 'ok',
  great: 'ok',
  good: 'ok',
  acceptable: 'warn',
  poor: 'bad',
};

function summaryVerdict({ models, failed, canceled }) {
  if (canceled) {
    return { tier: null, tone: 'warn', label: 'Run canceled', detail: 'The benchmark was stopped before it finished.' };
  }
  if (!models.length) {
    if (failed.length) {
      return { tier: null, tone: 'bad', label: 'Tests failed', detail: 'No models finished. Open the technical report to see why.' };
    }
    return { tier: null, tone: 'ok', label: 'Hardware checked', detail: 'Pick a model and run it to grade real AI speed on this device.' };
  }
  const tier = worstTier(models.map((model) => model.tier));
  const copy = VERDICT_COPY[tier];
  const bottleneck = [...models].sort((a, b) => a.fps - b.fps)[0];
  const goodCut = TIERS.indexOf('good');
  const goodOrBetter = models.filter((model) => TIERS.indexOf(model.tier) <= goodCut).length;
  const plural = models.length !== 1;
  const coverage = `${goodOrBetter} of ${models.length} selected model${plural ? 's' : ''} ${plural ? 'run' : 'runs'} at Good speed or better.`;
  const detail = `${copy.detail} ${coverage} Graded by the slowest, ${bottleneck.name} (${Math.round(bottleneck.fps)} runs/sec).`;
  return { tier, tone: TONE_BY_TIER[tier], label: copy.label, detail };
}

function computeBreakdown(hardware) {
  if (!hardware?.workloads?.length) return [];
  return hardware.workloads.map((workload) => ({
    id: workload.id,
    name: workload.name,
    throughput: workload.throughput
      ? `${formatThroughput(workload.throughput.value)} ${workload.throughput.unit}`
      : `p50 ${formatMs(workload.stats.p50)}`,
    tier: workload.throughput ? rateWorkload(workload.id, workload.throughput.value) : null,
  }));
}

function systemFacts(device) {
  if (!device) return [];
  const facts = [
    ['Browser', device.browser],
    ['Operating system', joinNonEmpty([device.platform, device.platformVersion])],
    ['CPU', joinNonEmpty([device.cpuArchitecture, coresText(device)])],
    ['GPU', gpuName(device)],
    ['Device memory', device.deviceMemoryGb ? `${device.deviceMemoryGb} GB (min)` : 'not exposed'],
    ['WebGPU', device.webGpuAvailable ? 'available' : 'not available'],
    ['Multithreading', device.crossOriginIsolated ? 'enabled (isolated)' : 'single-thread only'],
    ['WASM SIMD', device.wasmFeatures?.simd ? 'yes' : 'no'],
    ['Cache headroom', storageText(device.storage)],
    ['Network', networkText(device.network)],
  ];
  return facts
    .filter(([, value]) => value && value !== 'not exposed')
    .map(([label, value]) => ({ label, value }));
}

function coresText(device) {
  return device.logicalCpuCores ? `${device.logicalCpuCores} logical cores` : null;
}

function gpuName(device) {
  const renderer = device.gpuRenderer?.renderer;
  if (renderer) return renderer;
  const adapter = device.gpuAdapter;
  if (adapter?.description && adapter.description !== 'not exposed') return adapter.description;
  return device.webGpuAvailable ? 'available (name hidden)' : 'not exposed';
}

function storageText(storage) {
  if (!storage?.quota) return null;
  return `${formatBytes(storage.quota - (storage.usage ?? 0))} free`;
}

function networkText(network) {
  if (!network?.effectiveType) return null;
  const parts = [network.effectiveType];
  if (network.downlinkMbps) parts.push(`~${network.downlinkMbps} Mbps`);
  return parts.join(' · ');
}

function joinNonEmpty(parts) {
  return parts.filter(Boolean).join(' · ') || null;
}

function formatThroughput(value) {
  if (!Number.isFinite(value)) return 'n/a';
  if (value >= 100) return value.toFixed(0);
  if (value >= 10) return value.toFixed(1);
  return value.toFixed(2);
}

function backendLabel(backend) {
  if (backend === 'webgpu') return 'GPU (WebGPU)';
  if (backend === 'wasm') return 'CPU (WASM)';
  return backend;
}

export function buildEmailReport({ device, manifest, modules, status }) {
  const completed = modules.filter((module) => module.status === 'complete');
  const failed = modules.filter((module) => module.status === 'failed');
  const skipped = modules.filter((module) => module.status === 'skipped');
  const canceled = status === 'canceled';
  const verdict = buildVerdict(completed, failed, canceled);

  return [
    'AI Browser Benchmark Report',
    `Run status: ${canceled ? 'Canceled before completion' : 'Complete'}`,
    `Generated: ${new Date().toISOString()}`,
    '',
    'Summary',
    verdict,
    'Method note: this browser ran the selected hardware checks and exact ONNX model tests listed below. Results measure this browser and backend, not a server-side runtime.',
    '',
    'Selected Results',
    ...completed.map(formatModuleSummary),
    ...(failed.length
      ? ['', 'Failed Modules', ...failed.map((module) => `- ${module.name}: ${module.error}`)]
      : []),
    ...(skipped.length
      ? ['', 'Skipped Modules', ...skipped.map((module) => `- ${module.name}: ${module.reason}`)]
      : []),
    '',
    'Environment',
    `- Browser: ${device.browser ?? 'not exposed'}`,
    `- User agent: ${device.userAgent}`,
    `- Operating system: ${joinOrNa([device.platform, device.platformVersion])}`,
    `- CPU: ${joinOrNa([device.cpuArchitecture, device.logicalCpuCores ? `${device.logicalCpuCores} logical cores` : null])}`,
    `- Device model: ${device.deviceModel ?? 'not exposed'}`,
    `- Device memory: ${device.deviceMemoryGb ? `${device.deviceMemoryGb} GB (minimum)` : 'not exposed'}`,
    `- WebGPU: ${device.webGpuAvailable ? 'available' : 'not available'}`,
    `- GPU (WebGL): ${device.gpuRenderer?.renderer ?? 'not exposed'}`,
    `- GPU adapter (WebGPU): ${formatGpu(device.gpuAdapter)}`,
    `- GPU limits: ${formatGpuLimits(device.gpuAdapter)}`,
    `- Cross-origin isolated: ${device.crossOriginIsolated ? 'yes' : 'no'}`,
    `- WASM features: SIMD ${yesNo(device.wasmFeatures?.simd)}, threads ${yesNo(device.wasmFeatures?.threads)}, SharedArrayBuffer ${yesNo(device.wasmFeatures?.sharedArrayBuffer)}`,
    `- Storage: ${formatStorage(device.storage)}`,
    `- Network: ${formatNetwork(device.network)}`,
    `- JS heap limit: ${device.performanceMemory ? formatBytes(device.performanceMemory.jsHeapSizeLimit) : 'not exposed'}`,
    `- Screen: ${device.screen.width} x ${device.screen.height} @ ${device.screen.devicePixelRatio}x, ${device.screen.colorDepth ?? '?'}-bit color`,
    '',
    'Manifest',
    `- Name: ${manifest.name ?? 'unnamed manifest'}`,
    `- Updated: ${manifest.updated ?? 'not specified'}`,
    `- Runtime WASM assets: ${formatBytes(manifest.runtime?.wasm?.bytes ?? 0)}`,
    `- Runtime WebGPU assets: ${formatBytes(manifest.runtime?.webgpu?.bytes ?? 0)}`,
    '',
    'Detailed Results',
    ...completed.flatMap(formatModuleDetails),
  ].join('\n');
}

function buildVerdict(completed, failed, canceled) {
  if (!completed.length) {
    return canceled
      ? 'No completed modules were available before cancellation.'
      : 'No completed modules were available.';
  }
  const modelResults = completed.filter((module) => module.samples?.length);
  const slow = modelResults.filter((module) => module.stats.p95 > 1000);
  const backendText = modelResults.length
    ? `Model modules ran on ${[
        ...new Set(modelResults.map((module) => module.backend)),
      ].join(', ')}.`
    : 'Only the general hardware check completed.';
  if (failed.length || slow.length) {
    return `Usable but watch the slow modules. ${backendText}`;
  }
  return `Good browser-side local CV suitability for the selected modules. ${backendText}`;
}

function formatModuleSummary(module) {
  if (module.id === 'general-hardware') {
    return `- ${module.name}: complete, ${module.workloads.length} hardware workloads measured.`;
  }
  return [
    `- ${module.name}: ${module.backend}, p50 ${formatMs(module.stats.p50)}, p95 ${formatMs(module.stats.p95)}, mean ${formatMs(module.stats.mean)}, p50 FPS-equivalent ${module.stats.fpsP50.toFixed(2)}.`,
    `  Download: ${formatBytes(module.bytesDownloaded)} in ${formatMs(module.downloadMs)}. Session creation: ${formatMs(module.sessionCreateMs)}. Hash: ${module.hashStatus}.`,
  ].join('\n');
}

function formatModuleDetails(module) {
  if (module.id === 'general-hardware') {
    return [
      '',
      `${module.name}`,
      `Backend: ${module.backend}`,
      ...module.workloads.map(
        (workload) =>
          [
            `- ${workload.name}${formatThroughputSuffix(workload.throughput)}`,
            `  ${formatStatsLine(workload.stats)}`,
            `  raw samples ms: ${formatSamples(workload.samples)}`,
          ].join('\n'),
      ),
    ];
  }
  return [
    '',
    `${module.name}`,
    `Task: ${module.task}`,
    `Backend: ${module.backend}`,
    `Input shape: ${module.inputShape.join(' x ')}`,
    `Model bytes: ${formatBytes(module.modelBytes)}`,
    `Downloaded: ${formatBytes(module.bytesDownloaded)} in ${formatMs(module.downloadMs)}`,
    `Session creation: ${formatMs(module.sessionCreateMs)}`,
    `Latency: ${formatStatsLine(module.stats)}`,
    `FPS equivalent: p50 ${module.stats.fpsP50.toFixed(2)}, p95 ${module.stats.fpsP95.toFixed(2)}`,
    `Raw samples ms: ${formatSamples(module.samples)}`,
    ...(module.warnings?.length
      ? [`Warnings: ${module.warnings.join(' | ')}`]
      : []),
  ];
}

function formatStatsLine(stats) {
  return `p50 ${formatMs(stats.p50)}, p95 ${formatMs(stats.p95)}, mean ${formatMs(stats.mean)}, min ${formatMs(stats.min)}, max ${formatMs(stats.max)}, stddev ${formatMs(stats.stddev)}`;
}

function formatSamples(samples) {
  return samples.map((value) => value.toFixed(2)).join(', ');
}

function formatGpu(adapter) {
  if (!adapter) return 'not available';
  if (adapter.error) return `error: ${adapter.error}`;
  if (adapter.available === false) return 'not available';
  return [
    adapter.vendor,
    adapter.architecture,
    adapter.device,
    adapter.description,
    adapter.isFallback ? 'fallback adapter' : null,
  ]
    .filter(Boolean)
    .join(' / ');
}

function formatGpuLimits(adapter) {
  const limits = adapter?.limits;
  if (!limits) return 'not exposed';
  return [
    `maxBufferSize ${formatBytes(limits.maxBufferSize)}`,
    `maxStorageBufferBindingSize ${formatBytes(limits.maxStorageBufferBindingSize)}`,
    `maxComputeInvocationsPerWorkgroup ${limits.maxComputeInvocationsPerWorkgroup}`,
  ].join(', ');
}

function formatThroughputSuffix(throughput) {
  if (!throughput) return '';
  return `: ${throughput.value >= 100 ? throughput.value.toFixed(0) : throughput.value.toFixed(2)} ${throughput.unit}`;
}

function formatStorage(storage) {
  if (!storage?.quota) return 'not exposed';
  return `${formatBytes(storage.usage ?? 0)} used of ${formatBytes(storage.quota)} quota`;
}

function formatNetwork(network) {
  if (!network?.effectiveType) return 'not exposed';
  const parts = [network.effectiveType];
  if (network.downlinkMbps) parts.push(`${network.downlinkMbps} Mbps downlink`);
  if (network.rttMs) parts.push(`${network.rttMs} ms rtt`);
  if (network.saveData) parts.push('save-data on');
  return parts.join(', ');
}

function joinOrNa(parts) {
  return parts.filter(Boolean).join(' ') || 'not exposed';
}

function yesNo(value) {
  return value ? 'yes' : 'no';
}
