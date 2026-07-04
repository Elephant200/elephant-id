import { useState } from 'react';
import { devPhotoAnalysis, imageUrl } from '../api.js';
import DevUploadZone from '../components/DevUploadZone.jsx';

/** Development-only full photo analyzer diagnostics. */
export default function DevPhotoAnalysisPage({ engineReady }) {
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const run = async (file) => {
    setError(null);
    setResult(null);
    setBusy(true);
    try {
      setResult(await devPhotoAnalysis(file));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className={result ? 'analysis-dev-screen has-result' : 'analysis-dev-screen'}>
      <h1 className="screen-title">
        Photo analysis <span className="badge warn">DEV ONLY</span>
      </h1>
      {!result && (
        <p className="screen-sub">
          Runs the full analyzer and renders the same diagnostic dashboard used by analyzer scripts.
        </p>
      )}

      {error && <div className="error-note">{error}</div>}

      {!result && (
        <div className="panel dev-panel">
          <div className="panel-title">PHOTO</div>
          <DevUploadZone
            busy={busy}
            disabled={!engineReady}
            title={busy ? 'Analyzing photo...' : 'Analyze photo'}
            action="Choose image"
            onFile={run}
          />
        </div>
      )}

      {result && (
        <div className="analysis-full">
          <div className="analysis-full-head">
            <div>
              <div className="panel-title">ANALYSIS · {result.identifier}</div>
              <div className="mono-dim">
                {result.photo.status} · {result.photo.detail}
              </div>
            </div>
            <button type="button" className="btn ghost" onClick={() => setResult(null)}>
              New image
            </button>
          </div>
          <div className="dev-natural-frame">
            <img
              src={imageUrl(result.diagnostic_path)}
              alt={`${result.identifier} analyzer diagnostic`}
              className="dev-natural-image analysis-natural-image"
            />
          </div>
        </div>
      )}
    </div>
  );
}
