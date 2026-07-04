import { useState } from 'react';
import { imageUrl } from '../api.js';

/** Zoomable, pannable diagnostic image panel for development outputs. */
export default function ZoomPanel({ path, src, alt, className = '', children }) {
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [drag, setDrag] = useState(null);
  const resolvedSrc = src || imageUrl(path);

  const zoomBy = (delta) => {
    setScale((value) => Math.min(5, Math.max(0.5, Number((value + delta).toFixed(2)))));
  };

  const reset = () => {
    setScale(1);
    setOffset({ x: 0, y: 0 });
  };

  if (!resolvedSrc) return null;

  return (
    <div className={`zoom-panel ${className}`}>
      <div className="zoom-toolbar">
        <button type="button" className="zoom-btn" onClick={() => zoomBy(-0.25)}>
          -
        </button>
        <button type="button" className="zoom-btn" onClick={reset}>
          {Math.round(scale * 100)}%
        </button>
        <button type="button" className="zoom-btn" onClick={() => zoomBy(0.25)}>
          +
        </button>
        {children && <div className="zoom-meta">{children}</div>}
      </div>
      <div
        className="zoom-stage"
        onWheel={(event) => {
          event.preventDefault();
          zoomBy(event.deltaY > 0 ? -0.15 : 0.15);
        }}
        onMouseDown={(event) => {
          setDrag({
            x: event.clientX,
            y: event.clientY,
            startX: offset.x,
            startY: offset.y,
          });
        }}
        onMouseMove={(event) => {
          if (!drag) return;
          setOffset({
            x: drag.startX + event.clientX - drag.x,
            y: drag.startY + event.clientY - drag.y,
          });
        }}
        onMouseUp={() => setDrag(null)}
        onMouseLeave={() => setDrag(null)}
      >
        <img
          src={resolvedSrc}
          alt={alt}
          draggable="false"
          style={{
            transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
          }}
        />
      </div>
    </div>
  );
}
