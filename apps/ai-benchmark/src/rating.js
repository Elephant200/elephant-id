/**
 * Five-tier grading for benchmark results. Model inference speed (runs per
 * second) is the product-critical metric; the raw hardware throughputs are
 * rougher heuristics for field computer-vision suitability. Every threshold is
 * exported so the UI can show exactly how each grade is decided.
 */

export const TIERS = ['excellent', 'great', 'good', 'acceptable', 'poor'];

export const TIER_LABELS = {
  excellent: 'Excellent',
  great: 'Great',
  good: 'Good',
  acceptable: 'Acceptable',
  poor: 'Poor',
};

// Cut points are the minimum value for excellent / great / good / acceptable
// (higher is better); anything below the last cut is "poor".
//
// Model speed tiers are review-focused (this is offline photo review, not
// real-time video): Excellent >=25/s (<=40ms), Great >=10 (<=100ms), Good >=4
// (<=250ms), Acceptable >=1 (<=1s), Poor <1 (>1s). Anchored so a top consumer
// laptop GPU (M5 Max) reads Excellent, a mid laptop GPU (M3) reads Good, and
// CPU-only / tablet reads Acceptable.
export const MODEL_FPS_CUTS = [25, 10, 4, 1];

// Hardware throughput bars, calibrated against reference devices (M5 Max near
// the top of each, iPad / M3 lower). Rougher heuristics than model speed.
export const WORKLOAD_GRADES = {
  'cpu-float32-fma': { unit: 'GFLOP/s', cuts: [3.2, 2.7, 2.0, 1.0] },
  'memory-copy-64mb': { unit: 'GB/s', cuts: [50, 30, 18, 8] },
  'preprocess-640': { unit: 'img/s', cuts: [600, 350, 180, 60] },
  'postprocess-nms': { unit: 'runs/s', cuts: [2500, 1000, 300, 100] },
  'mask-composite': { unit: 'runs/s', cuts: [90, 65, 40, 15] },
};

/** Grade a higher-is-better value against four descending cut points. */
export function rateHigher(value, cuts) {
  if (!Number.isFinite(value)) return 'poor';
  const [excellent, great, good, acceptable] = cuts;
  if (value >= excellent) return 'excellent';
  if (value >= great) return 'great';
  if (value >= good) return 'good';
  if (value >= acceptable) return 'acceptable';
  return 'poor';
}

export function rateModelFps(fps) {
  return rateHigher(fps, MODEL_FPS_CUTS);
}

export function rateWorkload(id, value) {
  const grade = WORKLOAD_GRADES[id];
  return grade ? rateHigher(value, grade.cuts) : null;
}

/** The lowest (worst) tier among several — the bottleneck sets the overall grade. */
export function worstTier(tiers) {
  let worst = 0;
  for (const tier of tiers) {
    worst = Math.max(worst, TIERS.indexOf(tier));
  }
  return TIERS[worst];
}

/** Rows describing each grading scale, for the "How grades work" legend. */
export function gradingLegend() {
  const describe = (cuts, unit) => {
    const [excellent, great, good, acceptable] = cuts;
    return [
      { tier: 'excellent', text: `≥ ${excellent} ${unit}` },
      { tier: 'great', text: `≥ ${great} ${unit}` },
      { tier: 'good', text: `≥ ${good} ${unit}` },
      { tier: 'acceptable', text: `≥ ${acceptable} ${unit}` },
      { tier: 'poor', text: `< ${acceptable} ${unit}` },
    ];
  };
  return [
    { name: 'Model inference speed', bands: describe(MODEL_FPS_CUTS, 'runs/sec') },
    ...Object.entries(WORKLOAD_GRADES).map(([id, grade]) => ({
      name: WORKLOAD_NAMES[id] ?? id,
      bands: describe(grade.cuts, grade.unit),
    })),
  ];
}

const WORKLOAD_NAMES = {
  'cpu-float32-fma': 'CPU float32 math',
  'memory-copy-64mb': 'Memory bandwidth',
  'preprocess-640': 'Image preprocess',
  'postprocess-nms': 'Detection postprocess',
  'mask-composite': 'Segmentation masks',
};
