import { summarizeSamples } from './stats.js';

const SAMPLE_COUNT = 21;
const WARMUPS = 3;

self.onmessage = async (event) => {
  if (event.data?.type !== 'run-hardware') return;
  const startedAt = performance.now();
  const results = [];
  try {
    const workloads = [
      tensorMathWorkload(),
      memoryCopyWorkload(),
      preprocessWorkload(640),
      postprocessWorkload(),
      maskCompositeWorkload(),
    ];
    for (let index = 0; index < workloads.length; index += 1) {
      const workload = workloads[index];
      self.postMessage({
        type: 'progress',
        message: `General hardware: ${workload.name}`,
        completed: index,
        total: workloads.length,
      });
      const result = await measureWorkload(workload);
      results.push(result);
      self.postMessage({ type: 'partial-result', result });
    }
    self.postMessage({
      type: 'complete',
      result: {
        id: 'general-hardware',
        name: 'General Hardware Check',
        status: 'complete',
        backend: 'browser worker',
        startedAt,
        endedAt: performance.now(),
        workloads: results,
      },
    });
  } catch (error) {
    self.postMessage({
      type: 'error',
      error: error.message,
      result: {
        id: 'general-hardware',
        name: 'General Hardware Check',
        status: 'failed',
        backend: 'browser worker',
        startedAt,
        endedAt: performance.now(),
        workloads: results,
        error: error.message,
      },
    });
  }
};

async function measureWorkload(workload) {
  const samples = [];
  for (let i = 0; i < WARMUPS; i += 1) {
    workload.run();
    await tick();
  }
  for (let i = 0; i < SAMPLE_COUNT; i += 1) {
    const start = performance.now();
    workload.run();
    samples.push(performance.now() - start);
    await tick();
  }
  return {
    id: workload.id,
    name: workload.name,
    unit: 'ms',
    samples,
    stats: summarizeSamples(samples),
    details: workload.details,
  };
}

function tensorMathWorkload() {
  const length = 1 << 20;
  const a = new Float32Array(length);
  const b = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    a[i] = ((i % 251) - 125) / 127;
    b[i] = ((i % 197) - 98) / 97;
  }
  let sink = 0;
  return {
    id: 'cpu-float32-fma',
    name: 'CPU Float32 tensor math',
    details: 'Float32Array multiply-add over 1,048,576 elements, repeated.',
    run() {
      let acc = sink;
      for (let pass = 0; pass < 18; pass += 1) {
        for (let i = 0; i < length; i += 1) {
          acc += a[i] * b[i] + 0.000001;
        }
      }
      sink = acc;
    },
  };
}

function memoryCopyWorkload() {
  const length = 64 * 1024 * 1024;
  const a = new Uint8Array(length);
  const b = new Uint8Array(length);
  a.fill(37);
  return {
    id: 'memory-copy-64mb',
    name: 'Memory copy bandwidth',
    details: 'Copies 64 MiB between typed arrays.',
    run() {
      b.set(a);
      a[0] = (a[0] + b[length - 1]) & 255;
    },
  };
}

function preprocessWorkload(size) {
  const sourceSize = size * size * 4;
  const source = new Uint8Array(sourceSize);
  const output = new Float32Array(3 * size * size);
  for (let i = 0; i < source.length; i += 1) source[i] = (i * 13) & 255;
  return {
    id: `preprocess-${size}`,
    name: `${size}x${size} image preprocess`,
    details: 'RGBA uint8 to normalized NCHW float tensor.',
    run() {
      const plane = size * size;
      for (let y = 0; y < size; y += 1) {
        for (let x = 0; x < size; x += 1) {
          const src = (y * size + x) * 4;
          const dst = y * size + x;
          output[dst] = (source[src] / 255 - 0.485) / 0.229;
          output[plane + dst] = (source[src + 1] / 255 - 0.456) / 0.224;
          output[plane * 2 + dst] = (source[src + 2] / 255 - 0.406) / 0.225;
        }
      }
    },
  };
}

function postprocessWorkload() {
  const boxes = [];
  for (let i = 0; i < 1200; i += 1) {
    boxes.push({
      x1: (i * 17) % 640,
      y1: (i * 29) % 640,
      x2: ((i * 17) % 640) + 16 + (i % 80),
      y2: ((i * 29) % 640) + 16 + (i % 80),
      score: ((i * 37) % 1000) / 1000,
    });
  }
  return {
    id: 'postprocess-nms',
    name: 'Detection postprocess',
    details: 'Sorts 1,200 boxes and runs NMS-like overlap suppression.',
    run() {
      const selected = [];
      const candidates = [...boxes].sort((a, b) => b.score - a.score);
      for (const box of candidates) {
        let keep = true;
        for (const chosen of selected) {
          if (iou(box, chosen) > 0.45) {
            keep = false;
            break;
          }
        }
        if (keep) selected.push(box);
        if (selected.length >= 100) break;
      }
      return selected.length;
    },
  };
}

function maskCompositeWorkload() {
  const width = 160;
  const height = 160;
  const prototypes = 32;
  const masks = 24;
  const proto = new Float32Array(width * height * prototypes);
  const coeffs = new Float32Array(masks * prototypes);
  const output = new Uint8Array(masks * width * height);
  for (let i = 0; i < proto.length; i += 1) proto[i] = ((i % 127) - 63) / 63;
  for (let i = 0; i < coeffs.length; i += 1) coeffs[i] = ((i % 31) - 15) / 31;
  return {
    id: 'mask-composite',
    name: 'Segmentation mask composite',
    details: 'Combines 32 mask prototypes into 24 instance masks.',
    run() {
      const pixels = width * height;
      for (let mask = 0; mask < masks; mask += 1) {
        for (let pixel = 0; pixel < pixels; pixel += 1) {
          let value = 0;
          for (let p = 0; p < prototypes; p += 1) {
            value += proto[p * pixels + pixel] * coeffs[mask * prototypes + p];
          }
          output[mask * pixels + pixel] = value > 0 ? 255 : 0;
        }
      }
    },
  };
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  return intersection / Math.max(1, areaA + areaB - intersection);
}

function tick() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}
