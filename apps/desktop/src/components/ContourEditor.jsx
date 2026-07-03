// Full-screen, zoomable ear-contour editor (correction preview).
//
// The reviewer selects a region of the contour, which becomes a small set of
// coarse draggable handles; dragging re-fits that span through a Catmull-Rom
// curve while every point outside the region keeps full fidelity. Edits are
// local-only: nothing persists and no tear profile is regenerated yet.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { imageUrl } from '../api.js';
import './contour-editor.css';

const MIN_ZOOM = 0.4;
const MAX_ZOOM = 12;
const HANDLE_ARC_SPACING = 90; // image pixels of contour arc per coarse handle
const MAX_HANDLES = 12;

function distance(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function nearestContourIndex(points, target) {
  let best = 0;
  let bestDistance = Infinity;
  for (let i = 0; i < points.length; i += 1) {
    const d = distance(points[i], target);
    if (d < bestDistance) {
      bestDistance = d;
      best = i;
    }
  }
  return best;
}

// Sample a centripetal Catmull-Rom spline through `controls` (>=2 points),
// returning `count` evenly spaced samples from the first to the last control.
function sampleCatmullRom(controls, count) {
  if (controls.length < 2) return controls.slice();
  const pts = [controls[0], ...controls, controls[controls.length - 1]];
  const segments = controls.length - 1;
  const samples = [];
  for (let s = 0; s < count; s += 1) {
    const t = (s / (count - 1)) * segments;
    const seg = Math.min(Math.floor(t), segments - 1);
    const u = t - seg;
    const [p0, p1, p2, p3] = [pts[seg], pts[seg + 1], pts[seg + 2], pts[seg + 3]];
    const u2 = u * u;
    const u3 = u2 * u;
    samples.push([0, 1].map((axis) => {
      const a = p0[axis];
      const b = p1[axis];
      const c = p2[axis];
      const d = p3[axis];
      return (
        0.5 *
        (2 * b +
          (-a + c) * u +
          (2 * a - 5 * b + 4 * c - d) * u2 +
          (-a + 3 * b - 3 * c + d) * u3)
      );
    }));
  }
  return samples;
}

export default function ContourEditor({ side, candidate, onClose }) {
  const containerRef = useRef(null);
  const [view, setView] = useState({ k: 1, tx: 0, ty: 0 });
  const [points, setPoints] = useState(() =>
    (candidate.contour || []).map((point) => [point[0], point[1]]),
  );
  const [region, setRegion] = useState(null); // { start, end } contour indices
  const [handles, setHandles] = useState([]); // [{ index, x, y }]
  const [mode, setMode] = useState('region'); // 'region' | 'pan'
  const [edited, setEdited] = useState(false);
  const dragRef = useRef(null);

  const width = candidate.crop_width || 800;
  const height = candidate.crop_height || 600;
  const imageSrc = imageUrl(candidate.clean_crop_path || candidate.crop_path);

  // Fit the crop to the viewport on mount.
  useEffect(() => {
    const node = containerRef.current;
    if (!node) return;
    const fit = Math.min(
      (node.clientWidth - 80) / width,
      (node.clientHeight - 80) / height,
    );
    const k = Math.max(MIN_ZOOM, Math.min(fit, MAX_ZOOM));
    setView({
      k,
      tx: (node.clientWidth - width * k) / 2,
      ty: (node.clientHeight - height * k) / 2,
    });
  }, [width, height]);

  useEffect(() => {
    const onKey = (event) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const toImage = useCallback(
    (clientX, clientY) => {
      const rect = containerRef.current.getBoundingClientRect();
      return [
        (clientX - rect.left - view.tx) / view.k,
        (clientY - rect.top - view.ty) / view.k,
      ];
    },
    [view],
  );

  const onWheel = useCallback(
    (event) => {
      event.preventDefault();
      const rect = containerRef.current.getBoundingClientRect();
      const cx = event.clientX - rect.left;
      const cy = event.clientY - rect.top;
      setView((current) => {
        const k = Math.min(
          MAX_ZOOM,
          Math.max(MIN_ZOOM, current.k * Math.exp(-event.deltaY * 0.0015)),
        );
        const scale = k / current.k;
        return {
          k,
          tx: cx - (cx - current.tx) * scale,
          ty: cy - (cy - current.ty) * scale,
        };
      });
    },
    [],
  );

  // Wheel listeners added natively so preventDefault works (passive: false).
  useEffect(() => {
    const node = containerRef.current;
    if (!node) return undefined;
    node.addEventListener('wheel', onWheel, { passive: false });
    return () => node.removeEventListener('wheel', onWheel);
  }, [onWheel]);

  // Place handles evenly along the span's arc length (image pixels), so a
  // long ear-margin stretch gets a genuinely coarse, draggable set of points
  // regardless of how densely the underlying contour is sampled.
  const buildHandles = (start, end) => {
    const [lo, hi] = start < end ? [start, end] : [end, start];
    const arc = [0];
    for (let i = lo + 1; i <= hi; i += 1) {
      arc.push(arc[arc.length - 1] + distance(points[i - 1], points[i]));
    }
    const total = arc[arc.length - 1];
    const count = Math.min(
      MAX_HANDLES,
      Math.max(3, Math.round(total / HANDLE_ARC_SPACING) + 1),
    );
    const built = [];
    let cursor = 0;
    for (let i = 0; i < count; i += 1) {
      const targetArc = (total * i) / (count - 1);
      while (cursor < arc.length - 1 && arc[cursor] < targetArc) cursor += 1;
      const index = lo + cursor;
      built.push({ index, x: points[index][0], y: points[index][1] });
    }
    return built;
  };

  const startRegionDrag = (event) => {
    const target = toImage(event.clientX, event.clientY);
    const start = nearestContourIndex(points, target);
    dragRef.current = { type: 'region', start, end: start };
    setRegion({ start, end: start });
    setHandles([]);
  };

  const startPan = (event) => {
    dragRef.current = {
      type: 'pan',
      x: event.clientX,
      y: event.clientY,
      tx: view.tx,
      ty: view.ty,
    };
  };

  const onPointerDown = (event) => {
    if (event.button !== 0) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    if (mode === 'region') startRegionDrag(event);
    else startPan(event);
  };

  const onHandlePointerDown = (event, handleIndex) => {
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { type: 'handle', handleIndex };
  };

  const refitSpan = (nextHandles) => {
    const [lo, hi] = [nextHandles[0].index, nextHandles[nextHandles.length - 1].index];
    const controls = nextHandles.map((handle) => [handle.x, handle.y]);
    const resampled = sampleCatmullRom(controls, hi - lo + 1);
    setPoints((current) => {
      const next = current.slice();
      for (let i = lo; i <= hi; i += 1) next[i] = resampled[i - lo];
      return next;
    });
    setEdited(true);
  };

  const onPointerMove = (event) => {
    const drag = dragRef.current;
    if (!drag) return;
    if (drag.type === 'pan') {
      setView((current) => ({
        ...current,
        tx: drag.tx + event.clientX - drag.x,
        ty: drag.ty + event.clientY - drag.y,
      }));
      return;
    }
    const target = toImage(event.clientX, event.clientY);
    if (drag.type === 'region') {
      drag.end = nearestContourIndex(points, target);
      setRegion({ start: drag.start, end: drag.end });
      return;
    }
    if (drag.type === 'handle') {
      setHandles((current) => {
        const next = current.map((handle, i) =>
          i === drag.handleIndex ? { ...handle, x: target[0], y: target[1] } : handle,
        );
        refitSpan(next);
        return next;
      });
    }
  };

  const onPointerUp = () => {
    const drag = dragRef.current;
    dragRef.current = null;
    if (drag?.type === 'region' && region && region.start !== region.end) {
      setHandles(buildHandles(region.start, region.end));
    }
  };

  const reset = () => {
    setPoints((candidate.contour || []).map((point) => [point[0], point[1]]));
    setRegion(null);
    setHandles([]);
    setEdited(false);
  };

  const pathData = useMemo(() => {
    if (points.length === 0) return '';
    return `M ${points.map((point) => `${point[0]} ${point[1]}`).join(' L ')}`;
  }, [points]);

  const regionPath = useMemo(() => {
    if (!region) return '';
    const [lo, hi] =
      region.start < region.end
        ? [region.start, region.end]
        : [region.end, region.start];
    const span = points.slice(lo, hi + 1);
    if (span.length < 2) return '';
    return `M ${span.map((point) => `${point[0]} ${point[1]}`).join(' L ')}`;
  }, [region, points]);

  const handleRadius = Math.max(4, 9 / view.k);

  return (
    <div className="contour-editor" data-testid={`contour-editor-${side}`}>
      <header className="ce-header">
        <div>
          <div className="ce-title">
            {side === 'left' ? 'Left' : 'Right'} ear · contour correction
          </div>
          <div className="ce-sub">
            {candidate.file_name} — preview only: edits are not saved and the tear
            profile is not regenerated in this slice.
          </div>
        </div>
        <div className="ce-actions">
          <div className="ce-modes" role="group" aria-label="editor mode">
            <button
              type="button"
              className={`btn ${mode === 'region' ? 'ok' : 'ghost'}`}
              onClick={() => setMode('region')}
              title="Drag along the contour to choose the span to correct"
            >
              Select region
            </button>
            <button
              type="button"
              className={`btn ${mode === 'pan' ? 'ok' : 'ghost'}`}
              onClick={() => setMode('pan')}
              title="Drag to pan; scroll to zoom"
            >
              Pan
            </button>
          </div>
          <button type="button" className="btn ghost" onClick={reset} disabled={!edited && !region}>
            Reset
          </button>
          <button type="button" className="btn primary" data-testid="contour-editor-done" onClick={onClose}>
            Done
          </button>
        </div>
      </header>

      <div
        ref={containerRef}
        className={`ce-canvas ${mode === 'pan' ? 'panning' : ''}`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      >
        <div
          className="ce-stage"
          style={{
            transform: `translate(${view.tx}px, ${view.ty}px) scale(${view.k})`,
            width,
            height,
          }}
        >
          <img src={imageSrc} alt={`${side} ear crop`} width={width} height={height} draggable={false} />
          <svg
            className="ce-overlay"
            width={width}
            height={height}
            viewBox={`0 0 ${width} ${height}`}
          >
            <path d={pathData} className="ce-contour" style={{ strokeWidth: 2.5 / view.k }} />
            {regionPath && (
              <path
                d={regionPath}
                className="ce-region"
                style={{ strokeWidth: 4.5 / view.k }}
              />
            )}
            {handles.map((handle, index) => (
              <circle
                key={`${handle.index}`}
                cx={handle.x}
                cy={handle.y}
                r={handleRadius}
                className="ce-handle"
                style={{ strokeWidth: 2 / view.k }}
                onPointerDown={(event) => onHandlePointerDown(event, index)}
              />
            ))}
          </svg>
        </div>
      </div>

      <footer className="ce-footer">
        <span>
          {handles.length > 0
            ? `${handles.length} coarse handles on the selected span — drag them; the rest of the contour keeps full detail`
            : 'Drag along the ear margin to select the region to correct · scroll to zoom · Esc to close'}
        </span>
        {edited && <span className="ce-flag">edited (preview only)</span>}
      </footer>
    </div>
  );
}
