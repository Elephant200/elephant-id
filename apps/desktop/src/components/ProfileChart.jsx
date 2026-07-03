// SVG chart of tear-depth profiles: 0–180° along the outer ear margin,
// positive values are inward tears (depth / R). Supports overlaying a second
// series for query-vs-catalog comparison.

const WIDTH = 300;
const HEIGHT = 118;
const PAD_LEFT = 6;
const PAD_BOTTOM = 18;
const TRIM_DEGREES = 5;

function pathFor(values, yMax, close) {
  const n = values.length;
  const plotWidth = WIDTH - PAD_LEFT;
  const plotHeight = HEIGHT - PAD_BOTTOM;
  const x = (index) => PAD_LEFT + (index / (n - 1)) * plotWidth;
  const y = (value) => plotHeight - (Math.max(0, value) / yMax) * plotHeight;
  let d = `M ${x(0)} ${y(values[0])}`;
  for (let i = 1; i < n; i += 1) d += ` L ${x(i)} ${y(values[i])}`;
  if (close) d += ` L ${x(n - 1)} ${plotHeight} L ${x(0)} ${plotHeight} Z`;
  return d;
}

export default function ProfileChart({ series, height = HEIGHT }) {
  const drawn = series.filter((entry) => entry.values && entry.values.length > 1);
  if (drawn.length === 0) return null;
  // Low floor so small scallops — which carry identity signal — stay visible.
  const maxValue = Math.max(
    0.05,
    ...drawn.flatMap((entry) => entry.values.map((value) => value)),
  );
  const yMax = maxValue * 1.15;
  const plotWidth = WIDTH - PAD_LEFT;
  const plotHeight = HEIGHT - PAD_BOTTOM;
  const trimWidth = (TRIM_DEGREES / 180) * plotWidth;

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      style={{ width: '100%', height }}
      className="profile-chart"
      role="img"
      aria-label="tear depth profile"
    >
      <rect
        x={PAD_LEFT}
        y={0}
        width={plotWidth}
        height={plotHeight}
        className="chart-bg"
      />
      {/* anchor-adjacent trim bands are excluded from coding */}
      <rect x={PAD_LEFT} y={0} width={trimWidth} height={plotHeight} className="chart-trim" />
      <rect
        x={PAD_LEFT + plotWidth - trimWidth}
        y={0}
        width={trimWidth}
        height={plotHeight}
        className="chart-trim"
      />
      {[45, 90, 135].map((deg) => (
        <line
          key={deg}
          x1={PAD_LEFT + (deg / 180) * plotWidth}
          y1={0}
          x2={PAD_LEFT + (deg / 180) * plotWidth}
          y2={plotHeight}
          className="chart-grid"
        />
      ))}
      {drawn.map((entry) => (
        <g key={entry.label}>
          {entry.fill && (
            <path d={pathFor(entry.values, yMax, true)} fill={entry.color} opacity="0.18" />
          )}
          <path
            d={pathFor(entry.values, yMax, false)}
            fill="none"
            stroke={entry.color}
            strokeWidth="2"
          />
        </g>
      ))}
      {[0, 90, 180].map((deg) => (
        <text
          key={deg}
          x={PAD_LEFT + (deg / 180) * plotWidth}
          y={HEIGHT - 3}
          className="chart-tick"
          textAnchor={deg === 0 ? 'start' : deg === 180 ? 'end' : 'middle'}
        >
          {deg}°
        </text>
      ))}
    </svg>
  );
}
