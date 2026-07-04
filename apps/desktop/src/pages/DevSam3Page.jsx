import { useState } from 'react';
import { devSam3 } from '../api.js';
import DevUploadZone from '../components/DevUploadZone.jsx';
import ZoomPanel from '../components/ZoomPanel.jsx';

/** Development-only SAM3 overlay diagnostics. */
export default function DevSam3Page({ engineReady }) {
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const run = async (file) => {
    setError(null);
    setResult(null);
    setBusy(true);
    try {
      setResult(await devSam3(file));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <h1 className="screen-title">
        SAM3 overlays <span className="badge warn">DEV ONLY</span>
      </h1>
      <p className="screen-sub">
        Body and feature segmentation are run independently and rendered side by side.
      </p>

      {error && <div className="error-note">{error}</div>}

      {!result ? (
        <div className="panel dev-panel">
          <div className="panel-title">SAM3 IMAGE</div>
          <DevUploadZone
            busy={busy}
            disabled={!engineReady}
            title={busy ? 'Running SAM3...' : 'Run SAM3'}
            action="Choose image"
            onFile={run}
          />
        </div>
      ) : (
        <div className="dev-result">
          <div className="dev-result-head">
            <div>
              <div className="panel-title">RESULT · {result.identifier}</div>
              <div className="mono-dim">{result.file_name}</div>
            </div>
            <button type="button" className="btn ghost" onClick={() => setResult(null)}>
              New image
            </button>
          </div>
          <div className="sam3-compare">
            {result.overlays.map((overlay) => (
              <ZoomPanel
                key={overlay.preset}
                path={overlay.overlay_path}
                alt={`${overlay.preset} SAM3 overlay`}
                className="sam3-panel"
              >
                <span className="badge">{overlay.preset}</span>
                <span>{overlay.detection_count} detections</span>
                <span>{overlay.cache_status}</span>
              </ZoomPanel>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
