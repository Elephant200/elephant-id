import { useState } from 'react';
import { createSighting, hasNativeFolderPicker, selectFolder } from '../api.js';

export default function ImportPage({ setActiveSightingId, refreshSightings, setRoute }) {
  const [path, setPath] = useState('');
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const native = hasNativeFolderPicker();

  const startIngest = async (folder) => {
    setError(null);
    setBusy(true);
    try {
      const record = await createSighting(folder);
      setActiveSightingId(record.sighting_id);
      refreshSightings();
      setRoute('analyze');
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const pickFolder = async () => {
    const folder = await selectFolder();
    if (folder) startIngest(folder);
  };

  return (
    <div>
      <h1 className="screen-title">Import a sighting</h1>
      <p className="screen-sub">
        Select one folder of photos from one elephant sighting. Every photo in
        the folder is treated as evidence about the same individual.
      </p>

      {error && (
        <div className="error-note" data-testid="error-note">
          {error}
        </div>
      )}

      <div className="panel">
        <div className="panel-title">SIGHTING FOLDER</div>
        <div className="dropzone">
          <h3>One folder. One elephant.</h3>
          <p>
            The folder is indexed in place — nothing is moved or modified. All
            analysis runs locally; no internet needed.
          </p>
          {native ? (
            <button
              type="button"
              className="btn primary"
              disabled={busy}
              onClick={pickFolder}
              data-testid="pick-folder"
            >
              {busy ? 'Starting…' : 'Choose sighting folder'}
            </button>
          ) : (
            <form
              onSubmit={(event) => {
                event.preventDefault();
                if (path.trim()) startIngest(path.trim());
              }}
            >
              <input
                className="path-input"
                placeholder="/path/to/sighting-folder"
                value={path}
                onChange={(event) => setPath(event.target.value)}
                data-testid="folder-input"
              />
              <div>
                <button
                  type="submit"
                  className="btn primary"
                  disabled={busy || !path.trim()}
                  data-testid="ingest-submit"
                >
                  {busy ? 'Starting…' : 'Import folder'}
                </button>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}
