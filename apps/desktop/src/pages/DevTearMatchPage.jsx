import { useRef, useState } from 'react';
import { devTearMatch, imageUrl } from '../api.js';

function PairUploadBox({ label, file, disabled, onFile }) {
  const inputRef = useRef(null);

  const choose = (nextFile) => {
    if (nextFile && !disabled) onFile(nextFile);
  };

  return (
    <div
      className={`dropzone dev-upload pair-upload-box ${file ? 'has-file' : ''}`}
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => {
        event.preventDefault();
        choose(event.dataTransfer.files?.[0]);
      }}
    >
      <div className="pair-upload-label">{label}</div>
      <h3>{file ? file.name : 'Choose image'}</h3>
      <p>Drop a JPG or PNG, or choose one from disk.</p>
      <input
        ref={inputRef}
        type="file"
        accept=".jpg,.jpeg,.png"
        style={{ display: 'none' }}
        onChange={(event) => choose(event.target.files?.[0])}
      />
      <button
        type="button"
        className="btn primary"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
      >
        {file ? 'Change image' : 'Choose image'}
      </button>
    </div>
  );
}

/** Development-only two-image tear-profile matching diagnostics. */
export default function DevTearMatchPage({ engineReady }) {
  const [fileA, setFileA] = useState(null);
  const [fileB, setFileB] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const reset = () => {
    setFileA(null);
    setFileB(null);
    setResult(null);
    setError(null);
  };

  const run = async () => {
    if (!fileA || !fileB || busy || !engineReady) return;
    setError(null);
    setResult(null);
    setBusy(true);
    try {
      setResult(await devTearMatch(fileA, fileB));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className={`analysis-dev-screen ${result ? 'has-result' : ''}`}>
      <h1 className="screen-title">
        Tear match <span className="badge warn">DEV ONLY</span>
      </h1>
      <p className="screen-sub">
        Upload two images, extract tear profiles, and match only shared ear sides.
      </p>

      {error && <div className="error-note">{error}</div>}

      {!result ? (
        <div className="panel dev-panel">
          <div className="panel-title">IMAGE PAIR</div>
          <div className="pair-upload-grid">
            <PairUploadBox
              label="Image A · query"
              file={fileA}
              disabled={busy || !engineReady}
              onFile={setFileA}
            />
            <PairUploadBox
              label="Image B · candidate"
              file={fileB}
              disabled={busy || !engineReady}
              onFile={setFileB}
            />
          </div>
          <div className="pair-action-row">
            <button
              type="button"
              className="btn primary"
              disabled={busy || !engineReady || !fileA || !fileB}
              onClick={run}
            >
              {busy ? 'Matching...' : 'Match tear profiles'}
            </button>
          </div>
        </div>
      ) : (
        <div className="analysis-full">
          <div className="analysis-full-head">
            <div>
              <div className="panel-title">RESULT · {result.shared_sides.join(' + ')}</div>
              <div className="pair-result-summary">
                <div>
                  <span className="mono-dim">A</span> {result.image_a.identifier}
                  <span className="mono-dim"> · {result.image_a.file_name}</span>
                </div>
                <div>
                  <span className="mono-dim">B</span> {result.image_b.identifier}
                  <span className="mono-dim"> · {result.image_b.file_name}</span>
                </div>
              </div>
            </div>
            <button type="button" className="btn ghost" onClick={reset}>
              New images
            </button>
          </div>

          <div className="match-score-grid">
            {result.matches.map((match) => (
              <div className="match-score-card" key={match.side}>
                <span className={`badge ${match.side}`}>{match.side}</span>
                <strong>{match.score.toFixed(3)}</strong>
                <span className="mono-dim">IoU {match.overlap_score.toFixed(3)}</span>
                <span className="mono-dim">shift {match.shift_degrees.toFixed(1)}deg</span>
                <span className="mono-dim">stretch x{match.stretch.toFixed(2)}</span>
              </div>
            ))}
          </div>

          <img
            src={imageUrl(result.aligned_graph_path)}
            alt="Aligned same-side tear profile matches"
            className="dev-natural-image match-natural-image"
          />
        </div>
      )}
    </div>
  );
}
