import { useState } from 'react';
import { decideSighting, imageUrl } from '../api.js';
import { ZoomImage } from '../components/Lightbox.jsx';

export default function FilePage({
  sighting,
  setSighting,
  refreshSightings,
  selectedCandidate,
  startNewSighting,
  setRoute,
}) {
  const [enrollName, setEnrollName] = useState('');
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  if (!sighting || (!sighting.match && !sighting.decision)) {
    return (
      <div>
        <h1 className="screen-title">File the decision</h1>
        <div className="empty-note">RUN MATCHING FIRST</div>
      </div>
    );
  }

  const decide = async (action, elephantName) => {
    setError(null);
    setBusy(true);
    try {
      setSighting(await decideSighting(sighting.sighting_id, action, elephantName));
      refreshSightings();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  if (sighting.decision) {
    const { decision } = sighting;
    const label =
      decision.action === 'confirm'
        ? `Confirmed as ${decision.elephant_name}`
        : decision.action === 'enroll'
          ? `Enrolled as new individual ${decision.elephant_name}`
          : 'Left unresolved';
    const filedCrops = sighting.photos
      .flatMap((photo) =>
        photo.ears.map((ear) => ({
          crop_path: ear.crop_path,
          side: ear.side,
          photo_id: photo.photo_id,
        })),
      )
      .filter((ear) => ear.crop_path)
      .slice(0, 6);
    return (
      <div>
        <h1 className="screen-title">File the decision</h1>
        <div className="decided-banner" data-testid="decided">
          <div className="big">✓ {label}</div>
          <div style={{ flex: 1 }} />
          <button type="button" className="btn" onClick={startNewSighting}>
            New sighting
          </button>
        </div>

        {decision.action !== 'unresolved' && (
          <div className="panel">
            <div className="panel-title">FILED TO CATALOG</div>
            <p className="screen-sub" style={{ marginBottom: 14 }}>
              {sighting.profile_count} ear profile
              {sighting.profile_count === 1 ? '' : 's'} from{' '}
              {sighting.folder_name} now belong to{' '}
              <strong>{decision.elephant_name}</strong>. Future sightings will
              match against them.
            </p>
            <div className="repr-row">
              {filedCrops.map((ear, index) => (
                <figure className="repr-ear" key={`${ear.photo_id}-${ear.side}-${index}`}>
                  <ZoomImage
                    src={imageUrl(ear.crop_path)}
                    alt={`${ear.side} ear`}
                    caption={`${ear.photo_id} — ${ear.side} ear`}
                  />
                  <figcaption>
                    {ear.photo_id} · {ear.side}
                  </figcaption>
                </figure>
              ))}
            </div>
            <div className="page-actions">
              <button
                type="button"
                className="btn"
                data-testid="view-catalog"
                onClick={() => setRoute('catalog')}
              >
                View in catalog
              </button>
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div>
      <h1 className="screen-title">File the decision</h1>
      <p className="screen-sub">
        This is the final human call. Confirming or enrolling files this
        sighting&apos;s ear profiles into the catalog; unresolved keeps the
        sighting on record without an identity.
      </p>

      {error && (
        <div className="error-note" data-testid="error-note">
          {error}
        </div>
      )}

      <div className="panel">
        <div className="panel-title">CONFIRM AN EXISTING ELEPHANT</div>
        {selectedCandidate ? (
          <div className="decision-bar">
            <span className="mono-dim">SELECTED ON MATCH PAGE:</span>
            <span className="candidate-name">{selectedCandidate}</span>
            <span style={{ flex: 1 }} />
            <button
              type="button"
              className="btn ok"
              data-testid="confirm-selected"
              disabled={busy}
              onClick={() => decide('confirm', selectedCandidate)}
            >
              Confirm {selectedCandidate}
            </button>
          </div>
        ) : (
          <div className="mono-dim">
            NO CANDIDATE SELECTED — GO BACK TO MATCH AND CLICK THE CORRECT CANDIDATE
          </div>
        )}
      </div>

      <div className="panel">
        <div className="panel-title">OR ENROLL A NEW INDIVIDUAL</div>
        <div className="decision-bar">
          <input
            className="enroll-input"
            placeholder="New elephant name"
            value={enrollName}
            onChange={(event) => setEnrollName(event.target.value)}
            data-testid="enroll-name"
          />
          <button
            type="button"
            className="btn primary"
            disabled={busy || !enrollName.trim()}
            data-testid="enroll-submit"
            onClick={() => decide('enroll', enrollName.trim())}
          >
            Enroll new individual
          </button>
          <button
            type="button"
            className="btn ghost"
            disabled={busy}
            onClick={() => decide('unresolved')}
          >
            Leave unresolved
          </button>
        </div>
      </div>
    </div>
  );
}
