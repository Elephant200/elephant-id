import './styles.css';
import { collectDeviceInfo } from './device.js';
import { formatBytes, formatMs } from './stats.js';
import { TIER_LABELS } from './rating.js';

const DEFAULT_MANIFEST_URL = '/benchmark-assets.json';
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
  backend: 'auto',
  running: false,
  canceled: false,
  results: [],
  device: null,
  report: '',
  summary: null,
  abortController: null,
  worker: null,
  statusText: 'Ready',
};

const TASK_LABELS = {
  hardware: 'Device stress test',
  classification: 'Image classification',
  detection: 'Object detection',
  keypoints: 'Pose / keypoints',
  segmentation: 'Segmentation',
};

const app = document.querySelector('#app');

registerServiceWorker();
init();

function registerServiceWorker() {
  // Dev is served by Vite (HMR, transformed modules); a cache would fight it.
  if (!import.meta.env.PROD || !('serviceWorker' in navigator)) return;
  navigator.serviceWorker.register('/sw.js').catch(() => {});
}

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
          <div class="intro-copy">
            <h1>Can this device run local AI?</h1>
            <p>
              This runs real computer-vision models right here in your browser —
              nothing is uploaded and it works offline once loaded. Pick what to
              test and press Start.
            </p>
          </div>
          <div class="actions">
            <label class="field">
              <span>Run on</span>
              <select id="backend-select">
                <option value="auto">Auto (GPU, else CPU)</option>
                <option value="webgpu">GPU (WebGPU)</option>
                <option value="wasm">CPU (WASM)</option>
              </select>
            </label>
            <button class="primary" id="start-btn" type="button">Start</button>
            <button class="danger hidden" id="cancel-btn" type="button">Cancel</button>
          </div>
        </section>

        <section class="panel">
          <div class="section-head">
            <h2>What to test</h2>
            <div class="section-head-right">
              <button class="ghost" id="select-all-btn" type="button">Select all</button>
              <div id="selected-total" class="mono"></div>
            </div>
          </div>
          <div id="module-list" class="module-list"></div>
        </section>

        <section class="panel">
          <h2>Result</h2>
          <div id="summary" class="summary-empty">Press Start to benchmark this device.</div>
          <div id="results" class="results"></div>
        </section>

        <section class="panel">
          <details id="details-progress">
            <summary>Live activity <span id="run-state" class="mono">Idle</span></summary>
            <div id="progress-log" class="progress-log"></div>
          </details>
        </section>

        <section class="panel">
          <details id="details-report">
            <summary>
              Full technical report
              <button class="ghost" id="copy-report" type="button" disabled>Copy</button>
            </summary>
            <div id="report" class="report-empty">Run a benchmark to generate the report.</div>
          </details>
        </section>
      </main>
    </div>
  `;

  document.querySelector('#start-btn').addEventListener('click', runSelected);
  document.querySelector('#cancel-btn').addEventListener('click', cancelRun);
  document.querySelector('#copy-report').addEventListener('click', (event) => {
    event.preventDefault();
    event.stopPropagation();
    copyReport();
  });
  document.querySelector('#backend-select').addEventListener('change', (event) => {
    state.backend = event.target.value;
  });
  document.querySelector('#select-all-btn').addEventListener('click', toggleSelectAll);
}

function toggleSelectAll() {
  if (state.running) return;
  const modules = getModules();
  if (modules.every((module) => state.selected.has(module.id))) {
    state.selected.clear();
  } else {
    for (const module of modules) state.selected.add(module.id);
  }
  renderModules();
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
  state.summary = null;
  state.abortController = new AbortController();
  render();
  logProgress('Starting selected modules');

  const modules = getModules().filter((module) => state.selected.has(module.id));
  for (const module of modules) {
    if (state.abortController.signal.aborted) break;
    try {
      if (module.id === 'general-hardware') {
        const result = await runHardwareModule();
        state.results.push(result);
      } else {
        const { runModelModule } = await import('./modelRunner.js');
        const result = await runModelModule({
          model: module,
          manifest: state.manifest,
          signal: state.abortController.signal,
          onProgress: logProgress,
          backend: state.backend,
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
  const status = state.canceled ? 'canceled' : 'complete';
  const { buildEmailReport, buildSummary } = await import('./report.js');
  state.summary = buildSummary({
    device: state.device,
    modules: state.results,
    status,
  });
  state.report = buildEmailReport({
    device: state.device,
    manifest: state.manifest ?? {},
    modules: state.results,
    status,
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
    const finish = () => {
      state.abortController.signal.removeEventListener('abort', abort);
      worker.terminate();
      state.worker = null;
    };
    worker.onmessage = (event) => {
      const data = event.data;
      if (data.type === 'progress') logProgress(data.message);
      if (data.type === 'partial-result') renderWorkloadResult(data.result);
      if (data.type === 'complete') {
        finish();
        resolve(data.result);
      }
      if (data.type === 'error') {
        finish();
        reject(new Error(data.error));
      }
    };
    worker.postMessage({ type: 'run-hardware' });
  });
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
  document.querySelector('#backend-select').disabled = state.running;
  document.querySelector('#cancel-btn').classList.toggle('hidden', !state.running || state.canceled);
  document.querySelector('#copy-report').disabled = !state.report;
  document.querySelector('#report').innerHTML = state.report
    ? renderReportHtml(state.report)
    : 'Run a benchmark to generate the report.';
  document.querySelector('#report').classList.toggle('report-empty', !state.report);
  document.querySelector('#run-state').textContent = runStateLabel();
  document.querySelector('#details-progress').open = state.running;
  renderEnvironment();
  renderModules();
  renderSummary();
  renderResults();
}

function runStateLabel() {
  if (state.running) return state.canceled ? 'Canceling' : 'Running';
  if (state.canceled) return 'Canceled';
  return state.report ? 'Complete' : 'Idle';
}

function renderSummary() {
  const target = document.querySelector('#summary');
  if (!state.summary) {
    target.className = 'summary-empty';
    target.textContent = state.running
      ? 'Benchmarking… live activity is below.'
      : 'Press Start to benchmark this device.';
    return;
  }
  const { verdict, models, compute, system, legend } = state.summary;
  const tierClass = verdict.tier ?? TONE_TIER[verdict.tone] ?? 'good';
  target.className = 'summary';
  target.innerHTML = `
    <div class="verdict verdict-${tierClass}">
      <div class="verdict-label">${escapeHtml(verdict.label)}</div>
      <div class="verdict-detail">${escapeHtml(verdict.detail)}</div>
    </div>
    ${gradeSection('Model inference speed', models.map(modelGradeCard))}
    ${gradeSection('Compute breakdown', compute.map(computeGradeCard))}
    ${systemSection(system)}
    ${legendSection(legend)}
  `;
}

const TONE_TIER = { ok: 'great', warn: 'acceptable', bad: 'poor' };

function gradeSection(title, cards) {
  if (!cards.length) return '';
  return `
    <div class="grade-section">
      <h3>${escapeHtml(title)}</h3>
      <div class="grade-grid">${cards.join('')}</div>
    </div>
  `;
}

function modelGradeCard(model) {
  return `
    <div class="grade-card">
      <div class="grade-top">
        <span class="grade-name">${escapeHtml(model.name)}</span>
        ${tierBadge(model.tier)}
      </div>
      <div class="grade-metric">${Math.round(model.fps)} runs/sec</div>
      <div class="grade-meta">${escapeHtml(model.backendLabel)} · ${formatMs(model.latencyMs)}</div>
    </div>
  `;
}

function computeGradeCard(item) {
  return `
    <div class="grade-card">
      <div class="grade-top">
        <span class="grade-name">${escapeHtml(item.name)}</span>
        ${item.tier ? tierBadge(item.tier) : ''}
      </div>
      <div class="grade-metric">${escapeHtml(item.throughput)}</div>
    </div>
  `;
}

function systemSection(system) {
  if (!system.length) return '';
  return `
    <div class="grade-section">
      <h3>System</h3>
      <div class="stat-grid">
        ${system
          .map(
            (fact) => `
          <div class="stat-card">
            <div class="stat-label">${escapeHtml(fact.label)}</div>
            <div class="stat-value">${escapeHtml(fact.value)}</div>
          </div>`,
          )
          .join('')}
      </div>
    </div>
  `;
}

function legendSection(legend) {
  return `
    <details class="legend">
      <summary>How grades work</summary>
      <p class="legend-note">
        Model inference speed is the grade that matters for the workflow. The
        hardware rows are rough throughput heuristics. Cut-offs (higher is better):
      </p>
      ${legend
        .map(
          (scale) => `
        <div class="legend-row">
          <div class="legend-name">${escapeHtml(scale.name)}</div>
          <div class="legend-bands">
            ${scale.bands
              .map(
                (band) =>
                  `<span class="legend-band badge badge-${band.tier}">${TIER_LABELS[band.tier]} ${escapeHtml(band.text)}</span>`,
              )
              .join('')}
          </div>
        </div>`,
        )
        .join('')}
    </details>
  `;
}

function tierBadge(tier) {
  return `<span class="badge badge-${tier}">${TIER_LABELS[tier]}</span>`;
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
  const allSelected = modules.every((module) => state.selected.has(module.id));
  const selectAllBtn = document.querySelector('#select-all-btn');
  selectAllBtn.textContent = allSelected ? 'Clear all' : 'Select all';
  selectAllBtn.disabled = state.running;
  document.querySelector('#selected-total').textContent =
    state.selected.size > 1
      ? `About ${formatBytes(selectedBytes)} to download`
      : 'Choose one or more';
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
  const task = TASK_LABELS[module.task] ?? module.task;
  const size = module.id === 'general-hardware'
    ? 'no download'
    : `${formatBytes(module.bytes)} download`;
  return `
    <label class="module-card">
      <input type="checkbox" value="${module.id}" ${checked} ${state.running ? 'disabled' : ''} />
      <div>
        <div class="module-title">${escapeHtml(module.name)}</div>
        <div class="module-meta">${escapeHtml(task)} · ${escapeHtml(size)}</div>
      </div>
    </label>
  `;
}

function renderResults() {
  const target = document.querySelector('#results');
  // While running, show live per-module rows; the graded summary replaces them
  // once the run completes.
  if (state.summary || !state.results.length) {
    target.innerHTML = '';
    return;
  }
  target.innerHTML = state.results.map(renderResultRow).join('');
}

function renderResultRow(result) {
  if (result.status !== 'complete') {
    return `<div class="result-row warn"><strong>${escapeHtml(result.name)}</strong><span>${escapeHtml(result.error ?? result.reason ?? result.status)}</span></div>`;
  }
  if (result.id === 'general-hardware') {
    return `<div class="result-row"><strong>${escapeHtml(result.name)}</strong><span>${result.workloads.length} checks complete</span></div>`;
  }
  return `
    <div class="result-row">
      <strong>${escapeHtml(result.name)}</strong>
      <span>${escapeHtml(backendLabel(result.backend))} · ${Math.round(result.stats.fpsP50)} runs/sec</span>
    </div>
  `;
}

function backendLabel(backend) {
  if (backend === 'webgpu') return 'GPU';
  if (backend === 'wasm') return 'CPU';
  return backend;
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
