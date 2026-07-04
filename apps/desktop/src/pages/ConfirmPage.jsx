import { useEffect, useState } from 'react';
import { decideSighting, getElephant, imageUrl } from '../api.js';
import { ZoomImage } from '../components/Lightbox.jsx';

const DECISION_LABELS = {
  existing_known_elephant: 'Existing known elephant',
  new_known_elephant: 'New known elephant',
  unresolved: 'Unresolved',
  confirm: 'Existing known elephant',
  enroll: 'New known elephant',
};

export default function ConfirmPage({
  sighting,
  setSighting,
  refreshSightings,
  pendingDecision,
  startNewSighting,
  setStage,
}) {
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [catalogDetail, setCatalogDetail] = useState(null);

  const decided = sighting?.decision;
  const decision = decided || pendingDecision;
  const action = decision?.action;
  const label = DECISION_LABELS[action] || action;
  const elephantName = decision?.elephantName || decision?.elephant_name || null;

  useEffect(() => {
    let cancelled = false;
    setCatalogDetail(null);
    if (action !== 'existing_known_elephant' || !elephantName) return undefined;
    getElephant(elephantName)
      .then((detail) => {
        if (!cancelled) setCatalogDetail(detail);
      })
      .catch(() => {
        if (!cancelled) setCatalogDetail(null);
      });
    return () => {
      cancelled = true;
    };
  }, [action, elephantName]);

  if (!sighting) {
    return (
      <div>
        <h1 className="screen-title">Confirm decision</h1>
        <div className="empty-note">OPEN A SIGHTING FROM THE QUEUE FIRST</div>
      </div>
    );
  }

  if (!decision) {
    return (
      <div>
        <h1 className="screen-title">Confirm decision</h1>
        <div className="empty-note">CHOOSE AN IDENTITY DECISION FIRST</div>
      </div>
    );
  }

  const approvedEvidence = ['left', 'right']
    .map((side) => sighting.approved_evidence?.[side])
    .filter(Boolean);
  const selectedCandidate =
    decision.candidate ||
    sighting.match?.candidates?.find((candidate) => candidate.identity === elephantName);
  const referenceEvidence =
    action === 'existing_known_elephant'
      ? (selectedCandidate?.evidence || []).filter(
          (item) => item.gallery_display_crop_path || item.gallery_crop_path,
        )
      : [];
  const queryPhotos = uniqueImages(
    (sighting.photos || [])
      .filter((photo) => photo.photo_path)
      .map((photo) => ({
        key: photo.photo_id || photo.file_name,
        src: photo.photo_path,
        label: photo.file_name || photo.photo_id,
      })),
  );
  const catalogPhotos = uniqueImages(
    (catalogDetail?.photos || [])
      .map((photo) => ({
        key: photo.photo_path || photo.photo_id,
        src: photo.photo_path || photo.display_crop_path || photo.crop_path,
        label: `${photo.photo_id} · ${photo.date}`,
      }))
      .filter((photo) => photo.src),
  );

  const confirm = async () => {
    if (decided) return;
    setError(null);
    setBusy(true);
    try {
      const updated = await decideSighting(
        sighting.sighting_id,
        action,
        decision.elephantName,
      );
      setSighting(updated);
      refreshSightings();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <h1 className="screen-title">Confirm decision</h1>
      <p className="screen-sub">
        This read-only screen is the final review before the identity decision
        is saved. Existing known elephant decisions show the selected reference
        evidence; existing known-elephant decisions show side-by-side query and
        catalog context before and after saving.
      </p>

      {error && (
        <div className="error-note" data-testid="error-note">
          {error}
        </div>
      )}

      {decided && (
        <div className="decided-banner" data-testid="decided">
          <div className="big">
            {label}
            {decided.elephant_name ? `: ${decided.elephant_name}` : ''}
          </div>
          <div style={{ flex: 1 }} />
          <button type="button" className="btn" onClick={startNewSighting}>
            New sighting
          </button>
        </div>
      )}

      <div className="panel">
        <div className="panel-title">Identity decision</div>
        <div className="decision-summary">
          <div>
            <div className="mono-dim">STATE</div>
            <div className="candidate-name">{label}</div>
          </div>
          <div>
            <div className="mono-dim">KNOWN ELEPHANT</div>
            <div className="candidate-name">
              {decision.elephantName || decision.elephant_name || 'None'}
            </div>
          </div>
          <div>
            <div className="mono-dim">SIGHTING</div>
            <div className="candidate-name">{sighting.folder_name}</div>
          </div>
        </div>
      </div>

      <div className="confirm-compare">
        <EvidenceColumn
          title="Query sighting"
          subtitle={sighting.folder_name}
          ears={approvedEvidence.map((evidence) => ({
            key: evidence.candidate_id,
            side: evidence.side,
            src: evidence.display_crop_path || evidence.crop_path,
            label: `${evidence.file_name} · ${evidence.side}`,
            caption: `${evidence.file_name} - approved ${evidence.side} ear`,
          }))}
          photos={queryPhotos}
          empty="NO APPROVED LEFT/RIGHT EVIDENCE. THIS SIGHTING WILL BE SAVED AS UNRESOLVED."
        />
        <EvidenceColumn
          title="Known-elephant catalog"
          subtitle={elephantName || 'No catalog elephant selected'}
          ears={referenceEvidence.map((evidence) => ({
            key: `${evidence.gallery_photo_id}-${evidence.side}`,
            side: evidence.side,
            src: evidence.gallery_display_crop_path || evidence.gallery_crop_path,
            label: `${evidence.gallery_photo_id} · ${evidence.side}`,
            caption: `${elephantName} - ${evidence.gallery_photo_id} (${evidence.side})`,
          }))}
          photos={catalogPhotos}
          empty={
            action === 'existing_known_elephant'
              ? 'NO CATALOG REFERENCE IMAGES AVAILABLE'
              : 'NEW AND UNRESOLVED DECISIONS DO NOT ADD CATALOG REFERENCE IMAGES'
          }
        />
      </div>

      {!decided && (
        <div className="action-bar" data-testid="confirm-action-bar">
          <div className="action-bar-status">
            <div className="action-bar-line">
              <span className="side-tag">{label.toUpperCase()}</span>
              {decision.elephantName || decision.elephant_name
                ? `: ${decision.elephantName || decision.elephant_name}`
                : ''}
            </div>
            <div className="action-bar-hint">
              Check the summary above, then save the decision
            </div>
          </div>
          <button
            type="button"
            className="btn ghost"
            onClick={() => setStage(sighting.approved_evidence ? 'match' : 'review')}
          >
            Back
          </button>
          <button
            type="button"
            className="btn primary"
            data-testid="confirm-decision"
            disabled={busy}
            onClick={confirm}
          >
            {busy ? 'Saving decision...' : 'Confirm identity decision'}
          </button>
        </div>
      )}
    </div>
  );
}

function EvidenceColumn({ title, subtitle, ears, photos, empty }) {
  return (
    <section className="compare-column">
      <div className="compare-column-head">
        <div>
          <div className="panel-title">{title}</div>
          <div className="candidate-name">{subtitle}</div>
        </div>
      </div>
      {ears.length === 0 ? (
        <div className="empty-note">{empty}</div>
      ) : (
        <div className="confirm-ear-grid">
          {ears.map((ear) => (
            <figure className="confirm-ear" key={ear.key}>
              <ZoomImage
                src={imageUrl(ear.src)}
                alt={`${ear.side} ear`}
                caption={ear.caption}
              />
              <figcaption>
                <span className={`badge ${ear.side}`}>{ear.side}</span>
                {ear.label}
              </figcaption>
            </figure>
          ))}
        </div>
      )}
      <div className="confirm-photo-list">
        {photos.length === 0 ? (
          <div className="empty-note">NO FULL PHOTOS AVAILABLE</div>
        ) : (
          photos.map((photo) => (
            <figure className="confirm-photo" key={photo.key}>
              <ZoomImage
                src={imageUrl(photo.src)}
                alt={photo.label}
                caption={photo.label}
              />
              <figcaption>{photo.label}</figcaption>
            </figure>
          ))
        )}
      </div>
    </section>
  );
}

function uniqueImages(images) {
  const seen = new Set();
  return images.filter((image) => {
    if (!image.src || seen.has(image.src)) return false;
    seen.add(image.src);
    return true;
  });
}
