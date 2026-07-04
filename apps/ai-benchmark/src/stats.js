export function percentile(values, pct) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil((pct / 100) * sorted.length) - 1),
  );
  return sorted[index];
}

export function summarizeSamples(samples) {
  const values = samples.filter((value) => Number.isFinite(value));
  if (!values.length) {
    return {
      count: 0,
      mean: 0,
      p50: 0,
      p95: 0,
      min: 0,
      max: 0,
      stddev: 0,
      cv: 0,
      fpsP50: 0,
      fpsP95: 0,
    };
  }
  const total = values.reduce((sum, value) => sum + value, 0);
  const mean = total / values.length;
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
    values.length;
  const p50 = percentile(values, 50);
  const p95 = percentile(values, 95);
  return {
    count: values.length,
    mean,
    p50,
    p95,
    min: Math.min(...values),
    max: Math.max(...values),
    stddev: Math.sqrt(variance),
    cv: mean > 0 ? Math.sqrt(variance) / mean : 0,
    fpsP50: p50 > 0 ? 1000 / p50 : 0,
    fpsP95: p95 > 0 ? 1000 / p95 : 0,
  };
}

export function formatMs(value) {
  if (!Number.isFinite(value)) return 'n/a';
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  return `${value.toFixed(value >= 10 ? 1 : 2)} ms`;
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return 'n/a';
  const units = ['bytes', 'KiB', 'MiB', 'GiB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 2)} ${units[unit]}`;
}
