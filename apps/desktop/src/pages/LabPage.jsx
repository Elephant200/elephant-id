import { useRef, useState } from 'react';
import { devAnalyze } from '../api.js';
import PhotoEvidenceCard from '../components/PhotoEvidenceCard.jsx';

/** Development-only: run the full analyzer on one uploaded image. */
export default function LabPage({ engineReady }) {
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [fileName, setFileName] = useState(null);
  const inputRef = useRef(null);

  const analyze = async (file) => {
    if (!file) return;
    setError(null);
    setResult(null);
    setFileName(file.name);
    setBusy(true);
    try {
      setResult(await devAnalyze(file));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <h1 className="screen-title">
        Analysis lab <span className="badge warn">DEV ONLY</span>
      </h1>
      <p className="screen-sub">
        Drop in a single photo and run the full analysis pipeline on it — body
        and feature segmentation, view estimate, age and gender suggestions,
        tusk sides, anchored ear contours, and tear profiles. This is a
        development/demo page, separate from the reviewer queue. Photos named
        like dataset photos use the local cache; anything else needs the live
        model services.
      </p>

      {error && (
        <div className="error-note" data-testid="error-note">
          {error}
        </div>
      )}

      <div className="panel">
        <div className="panel-title">SINGLE PHOTO</div>
        <div
          className="dropzone"
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault();
            analyze(event.dataTransfer.files?.[0]);
          }}
        >
          <h3>{busy ? `Analyzing ${fileName}…` : 'Analyze one photo'}</h3>
          <p>Drag an image here or choose a file. Nothing is added to the known-elephant catalog.</p>
          <input
            ref={inputRef}
            type="file"
            accept=".jpg,.jpeg,.png"
            style={{ display: 'none' }}
            data-testid="lab-file"
            onChange={(event) => analyze(event.target.files?.[0])}
          />
          <button
            type="button"
            className="btn primary"
            disabled={busy || !engineReady}
            data-testid="lab-choose"
            onClick={() => inputRef.current?.click()}
          >
            {busy ? 'Running analysis…' : 'Choose photo'}
          </button>
        </div>
      </div>

      {result && (
        <>
          <div className="panel-title" style={{ marginBottom: 12 }}>
            ANALYSIS RESULT
          </div>
          <PhotoEvidenceCard photo={result} />
        </>
      )}
    </div>
  );
}
