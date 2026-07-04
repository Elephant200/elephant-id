import './styles.css';
import { collectDeviceInfo } from './device.js';
import { formatBytes, formatMs, stabilityLabel, summarizeSamples } from './stats.js';

const DEFAULT_MANIFEST_URL = '/benchmark-assets.example.json';
const HARDWARE_MODULE = {
  id: 'general-hardware',
  name: 'General Hardware Check',
  task: 'hardware',
  bytes: 0,
};

const state = {
  manifestUrl: DEFAULT_MANIFEST_URL,
  manifest: null,
  selected: new Set(['general-hardware']),
  running: false,
  canceled: false,
  results: [],
  device: null,
  report: '',
  abortController: null,
  worker: null,
  statusText: 'Ready',
};

const app = document.querySelector('#app');

init();

async function init() {
  renderShell();
  await loadManifest();
  state.device = await collectDeviceInfo();
  render();
}

function renderShell() {
  app.innerHTML = `
    <div class="shell">
      <header class="topbar">
        <div>
          <div class="brand">AI Benchmark</div>
          <div class="brand-sub" id="env-pill">Checking browser...</div>
        </div>
        <div class="status-pill" id="status-pill">Ready</div>
      </header>
      <main class="main">
        <section class="panel intro">
          <div>
            <h1>Local CV benchmark</h1>
            <p>Select tests, run them, copy the report.</p>
          </div>
          <div class="actions">
            <button class="primary" id="start-btn" type="button">Start selected</button>
            <button class="danger hidden" id="cancel-btn" type="button">Cancel</button>
          </div>
        </section>

        <section class="panel">
          <div class="section-head">
            <h2>Benchmark Modules</h2>
            <div id="selected-total" class="mono"></div>
          </div>
          <div id="module-list" class="module-list"></div>
        </section>

        <section class="panel">
          <div class="section-head">
            <h2>Run Progress</h2>
            <div id="run-state" class="mono">Idle</div>
          </div>
          <div id="progress-log" class="progress-log"></div>
          <div id="results" class="results"></div>
        </section>

        <section class="panel report-panel">
          <div class="section-head">
            <h2>Report</h2>
            <button class="ghost" id="copy-report" type="button" disabled>Copy Report</button>
          </div>
          <div id="report" class="report-empty">Run selected modules to generate a report.</div>
        </section>
      </main>
    </div>
  `;

  document.querySelector('#start-btn').addEventListener('click', runSelected);
  document.querySelector('#cancel-btn').addEventListener('click', cancelRun);
  document.querySelector('#copy-report').addEventListener('click', copyReport);
}

async function loadManifest() {
  try {
    setStatus('Loading manifest');
    const response = await fetch(state.manifestUrl, { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.manifest = await response.json();
    state.statusText = 'Ready';
  } catch (error) {
    state.manifest = null;
    state.statusText = `Manifest failed: ${error.message}`;
  }
}

async function runSelected() {
  if (state.running) return;
  state.running = true;
  state.canceled = false;
  state.results = [];
  state.report = '';
  state.abortController = new AbortController();
  render();
  logProgress('Starting selected modules');

  const modules = getModules().filter((module) => state.selected.has(module.id));
  for (const module of modules) {
    if (state.abortController.signal.aborted) break;
    try {
      if (module.id === 'general-hardware') {
        const result = await runHardwareModule();
        const gpuResult = await runWebGpuProbe(state.abortController.signal);
        if (gpuResult) result.workloads.push(gpuResult);
        state.results.push(result);
      } else {
        const { runModelModule } = await import('./modelRunner.js');
        const result = await runModelModule({
          model: module,
          manifest: state.manifest,
          signal: state.abortController.signal,
          onProgress: logProgress,
        });
        state.results.push(result);
      }
    } catch (error) {
      const aborted = error.name === 'AbortError';
      state.results.push({
        id: module.id,
        name: module.name,
        task: module.task,
        status: aborted ? 'skipped' : 'failed',
        reason: aborted ? 'Run canceled before this module completed.' : undefined,
        error: aborted ? undefined : error.message,
      });
      if (aborted) break;
    }
    renderResults();
  }

  state.running = false;
  state.canceled = state.abortController.signal.aborted;
  const { buildEmailReport } = await import('./report.js');
  state.report = buildEmailReport({
    device: state.device,
    manifest: state.manifest ?? {},
    modules: state.results,
    status: state.canceled ? 'canceled' : 'complete',
  });
  setStatus(state.canceled ? 'Canceled' : 'Complete');
  render();
}

function runHardwareModule() {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./benchmarkWorker.js', import.meta.url), {
      type: 'module',
    });
    state.worker = worker;
    const abort = () => {
      worker.terminate();
      reject(new DOMException('Benchmark canceled', 'AbortError'));
    };
    state.abortController.signal.addEventListener('abort', abort, { once: true });
    worker.onmessage = (event) => {
      const data = event.data;
      if (data.type === 'progress') logProgress(data.message);
      if (data.type === 'partial-result') renderWorkloadResult(data.result);
      if (data.type === 'complete') {
        state.abortController.signal.removeEventListener('abort', abort);
        worker.terminate();
        state.worker = null;
        resolve(data.result);
      }
      if (data.type === 'error') {
        state.abortController.signal.removeEventListener('abort', abort);
        worker.terminate();
        state.worker = null;
        reject(new Error(data.error));
      }
    };
    worker.postMessage({ type: 'run-hardware' });
  });
}

async function runWebGpuProbe(signal) {
  if (!navigator.gpu) return null;
  const samples = [];
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return null;
  const device = await adapter.requestDevice();
  const count = 1 << 20;
  const buffer = device.createBuffer({
    size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const shader = device.createShaderModule({
    code: `
      @group(0) @binding(0) var<storage, read_write> data: array<f32>;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let i = id.x;
        if (i < ${count}u) {
          data[i] = f32((i * 17u) % 251u) * 0.001 + data[i] * 1.0001;
        }
      }
    `,
  });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: shader, entryPoint: 'main' },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer } }],
  });
  for (let i = 0; i < 2; i += 1) {
    throwIfAborted(signal);
    await dispatchGpu(device, pipeline, bindGroup, count);
  }
  for (let i = 0; i < 7; i += 1) {
    throwIfAborted(signal);
    const start = performance.now();
    await dispatchGpu(device, pipeline, bindGroup, count);
    samples.push(performance.now() - start);
  }
  device.destroy();
  return {
    id: 'webgpu-compute',
    name: 'WebGPU compute dispatch',
    unit: 'ms',
    samples,
    stats: summarizeSamples(samples),
    details: 'Compute shader over 1,048,576 float values.',
  };
}

function throwIfAborted(signal) {
  if (signal?.aborted) {
    throw new DOMException('Benchmark canceled', 'AbortError');
  }
}

async function dispatchGpu(device, pipeline, bindGroup, count) {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

function cancelRun() {
  if (!state.running) return;
  logProgress('Cancel requested');
  state.abortController?.abort();
  state.worker?.terminate();
  state.canceled = true;
  setStatus('Canceling');
  render();
}

function render() {
  document.querySelector('#status-pill').textContent = state.statusText;
  document.querySelector('#start-btn').disabled = state.running || !state.manifest;
  document.querySelector('#cancel-btn').classList.toggle('hidden', !state.running || state.canceled);
  document.querySelector('#copy-report').disabled = !state.report;
  document.querySelector('#report').innerHTML = state.report
    ? renderReportHtml(state.report)
    : 'Run selected modules to generate a report.';
  document.querySelector('#report').classList.toggle('report-empty', !state.report);
  document.querySelector('#run-state').textContent = state.running
    ? state.canceled
      ? 'Canceling'
      : 'Running'
    : state.canceled
      ? 'Canceled'
      : state.report
        ? 'Complete'
        : 'Idle';
  renderEnvironment();
  renderModules();
  renderResults();
}

function renderEnvironment() {
  const target = document.querySelector('#env-pill');
  if (!state.device) {
    target.textContent = 'Checking browser...';
    return;
  }
  target.textContent = `Platform: ${state.device.platform ?? 'hidden'} · CPU cores: ${state.device.logicalCpuCores ?? 'hidden'} · WebGPU: ${state.device.webGpuAvailable ? 'available' : 'not available'}`;
}

function renderModules() {
  const list = document.querySelector('#module-list');
  const modules = getModules();
  const selectedBytes = modules
    .filter((module) => state.selected.has(module.id))
    .reduce((sum, module) => sum + (module.bytes ?? 0), runtimeBytes());
  document.querySelector('#selected-total').textContent =
    state.selected.size > 1
      ? `Selected model/runtime download: about ${formatBytes(selectedBytes)}`
      : 'Select modules independently';
  list.innerHTML = modules.map(renderModuleCard).join('');
  list.querySelectorAll('input[type="checkbox"]').forEach((input) => {
    input.addEventListener('change', () => {
      if (input.checked) state.selected.add(input.value);
      else state.selected.delete(input.value);
      renderModules();
    });
  });
}

function renderModuleCard(module) {
  const checked = state.selected.has(module.id) ? 'checked' : '';
  const bytes = module.id === 'general-hardware'
    ? 'No model download'
    : `${formatBytes(module.bytes)} model`;
  const hash = module.sha256?.startsWith('replace-with')
    ? 'hash placeholder'
    : module.sha256
      ? 'hash pinned'
      : 'hash missing';
  return `
    <label class="module-card">
      <input type="checkbox" value="${module.id}" ${checked} ${state.running ? 'disabled' : ''} />
      <div>
        <div class="module-title">${escapeHtml(module.name)}</div>
        <div class="module-meta">${escapeHtml(module.task)} · ${bytes}${module.id === 'general-hardware' ? '' : ` · ${hash}`}</div>
      </div>
    </label>
  `;
}

function renderResults() {
  const target = document.querySelector('#results');
  if (!state.results.length) {
    target.innerHTML = '<div class="empty">No completed module results yet.</div>';
    return;
  }
  target.innerHTML = state.results.map(renderResultRow).join('');
}

function renderResultRow(result) {
  if (result.status !== 'complete') {
    return `<div class="result-row warn"><strong>${escapeHtml(result.name)}</strong><span>${escapeHtml(result.error ?? result.reason ?? result.status)}</span></div>`;
  }
  if (result.id === 'general-hardware') {
    return `<div class="result-row"><strong>${result.name}</strong><span>${result.workloads.length} workloads complete</span></div>`;
  }
  return `
    <div class="result-row">
      <strong>${escapeHtml(result.name)}</strong>
      <span>${result.backend} · p50 ${formatMs(result.stats.p50)} · p95 ${formatMs(result.stats.p95)} · ${stabilityLabel(result.stats)}</span>
    </div>
  `;
}

function renderWorkloadResult(result) {
  logProgress(
    `${result.name}: p50 ${formatMs(result.stats.p50)}, p95 ${formatMs(result.stats.p95)}`,
  );
}

function getModules() {
  return [HARDWARE_MODULE, ...(state.manifest?.models ?? [])];
}

function runtimeBytes() {
  if (!state.manifest) return 0;
  const wasm = state.manifest.runtime?.wasm?.bytes ?? 0;
  const webgpu = state.device?.webGpuAvailable
    ? state.manifest.runtime?.webgpu?.bytes ?? 0
    : 0;
  return wasm + webgpu;
}

function logProgress(message) {
  const log = document.querySelector('#progress-log');
  const line = document.createElement('div');
  line.textContent = `${new Date().toLocaleTimeString()} ${message}`;
  log.prepend(line);
  setStatus(message);
}

function setStatus(message) {
  state.statusText = message;
  const pill = document.querySelector('#status-pill');
  if (pill) pill.textContent = message;
}

async function copyReport() {
  await navigator.clipboard.writeText(state.report);
  setStatus('Report copied');
}

function renderReportHtml(report) {
  return report
    .split('\n')
    .map((line) => {
      const text = escapeHtml(line);
      if (!line.trim()) return '<div class="report-gap"></div>';
      if (!line.startsWith('-') && !line.startsWith(' ') && line.length < 40) {
        return `<h3>${text}</h3>`;
      }
      if (line.startsWith('-')) return `<div class="report-bullet">${text}</div>`;
      return `<div class="report-line">${text}</div>`;
    })
    .join('');
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}
