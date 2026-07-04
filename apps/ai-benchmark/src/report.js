import { formatBytes, formatMs, stabilityLabel } from './stats.js';

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
    `- Platform: ${device.platform ?? 'not exposed'}`,
    `- Browser user agent: ${device.userAgent}`,
    `- Logical CPU cores: ${device.logicalCpuCores ?? 'not exposed'}`,
    `- Device memory: ${
      device.deviceMemoryGb ? `${device.deviceMemoryGb} GB` : 'not exposed'
    }`,
    `- WebGPU: ${device.webGpuAvailable ? 'available' : 'not available'}`,
    `- GPU adapter: ${formatGpu(device.gpuAdapter)}`,
    `- Cross-origin isolated: ${device.crossOriginIsolated ? 'yes' : 'no'}`,
    `- Screen: ${device.screen.width} x ${device.screen.height} @ ${device.screen.devicePixelRatio}x`,
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
  const spiky = completed.filter(
    (module) => module.stats && stabilityLabel(module.stats) === 'spiky',
  );
  const backendText = modelResults.length
    ? `Model modules ran on ${[
        ...new Set(modelResults.map((module) => module.backend)),
      ].join(', ')}.`
    : 'Only the general hardware check completed.';
  if (failed.length || slow.length || spiky.length) {
    return `Usable but watch the slow or variable modules. ${backendText}`;
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
      ...module.workloads.map((workload) => {
        const stats = workload.stats;
        return [
          `- ${workload.name}`,
          `  p50 ${formatMs(stats.p50)}, p95 ${formatMs(stats.p95)}, mean ${formatMs(stats.mean)}, min ${formatMs(stats.min)}, max ${formatMs(stats.max)}, stddev ${formatMs(stats.stddev)}, stability ${stabilityLabel(stats)}`,
          `  raw samples ms: ${workload.samples.map((value) => value.toFixed(2)).join(', ')}`,
        ].join('\n');
      }),
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
    `Latency: p50 ${formatMs(module.stats.p50)}, p95 ${formatMs(module.stats.p95)}, mean ${formatMs(module.stats.mean)}, min ${formatMs(module.stats.min)}, max ${formatMs(module.stats.max)}, stddev ${formatMs(module.stats.stddev)}, stability ${stabilityLabel(module.stats)}`,
    `FPS equivalent: p50 ${module.stats.fpsP50.toFixed(2)}, p95 ${module.stats.fpsP95.toFixed(2)}`,
    `Raw samples ms: ${module.samples.map((value) => value.toFixed(2)).join(', ')}`,
    ...(module.warnings?.length
      ? [`Warnings: ${module.warnings.join(' | ')}`]
      : []),
  ];
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
  ]
    .filter(Boolean)
    .join(' / ');
}
