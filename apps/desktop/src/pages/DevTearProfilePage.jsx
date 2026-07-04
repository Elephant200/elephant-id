import { useState } from 'react';
import { devTearProfile, imageUrl } from '../api.js';
import DevUploadZone from '../components/DevUploadZone.jsx';

/** Development-only tear-profile extraction diagnostics. */
export default function DevTearProfilePage({ engineReady }) {
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const run = async (file) => {
    setError(null);
    setResult(null);
    setBusy(true);
    try {
      setResult(await devTearProfile(file));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <h1 className="screen-title">
        Tear profile <span className="badge warn">DEV ONLY</span>
      </h1>
      <p className="screen-sub">
        Runs the ear anchoring path and renders the extracted tear-depth profile.
      </p>

      {error && <div className="error-note">{error}</div>}

      {!result ? (
        <div className="panel dev-panel">
          <div className="panel-title">EAR IMAGE</div>
          <DevUploadZone
            busy={busy}
            disabled={!engineReady}
            title={busy ? 'Extracting profile...' : 'Extract tear profile'}
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
          <div className="tear-grid">
            {result.ears.map((ear) => (
              <div className="tear-card" key={ear.side}>
                <div className="tear-card-head">
                  <span className={`badge ${ear.side}`}>{ear.side}</span>
                  <span className="mono-dim">mass {ear.mass.toFixed(1)}</span>
                </div>
                <img
                  src={imageUrl(ear.diagnostic_path)}
                  alt={`${ear.side} tear profile diagnostic`}
                  className="dev-natural-image"
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
