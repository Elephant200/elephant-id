import { useState } from 'react';
import { decideSighting, imageUrl } from '../api.js';
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

  if (!sighting) {
    return (
      <div>
        <h1 className="screen-title">Confirm decision</h1>
        <div className="empty-note">OPEN A SIGHTING FROM THE QUEUE FIRST</div>
      </div>
    );
  }

  const decided = sighting.decision;
  const decision = decided || pendingDecision;
  if (!decision) {
    return (
      <div>
        <h1 className="screen-title">Confirm decision</h1>
        <div className="empty-note">CHOOSE AN IDENTITY DECISION FIRST</div>
      </div>
    );
  }

  const action = decision.action;
  const label = DECISION_LABELS[action] || action;
  const approvedEvidence = ['left', 'right']
    .map((side) => sighting.approved_evidence?.[side])
    .filter(Boolean);
  const referenceEvidence =
    action === 'existing_known_elephant'
      ? (decision.candidate?.evidence || []).filter((item) => item.gallery_crop_path)
      : [];

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
        evidence; new and unresolved decisions do not add reference images.
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

      <div className="panel">
        <div className="panel-title">Approved sighting evidence</div>
        {approvedEvidence.length === 0 ? (
          <div className="empty-note">
            NO APPROVED LEFT/RIGHT EVIDENCE. THIS SIGHTING WILL BE SAVED AS UNRESOLVED.
          </div>
        ) : (
          <div className="repr-row">
            {approvedEvidence.map((evidence) => (
              <figure className="repr-ear" key={evidence.candidate_id}>
                <ZoomImage
                  src={imageUrl(evidence.crop_path)}
                  alt={`${evidence.side} approved evidence`}
                  caption={`${evidence.file_name} - approved ${evidence.side} ear`}
                />
                <figcaption>
                  {evidence.file_name} · {evidence.side}
                </figcaption>
              </figure>
            ))}
          </div>
        )}
      </div>

      {referenceEvidence.length > 0 && (
        <div className="panel">
          <div className="panel-title">Selected known-elephant reference</div>
          <div className="repr-row">
            {referenceEvidence.map((evidence) => (
              <figure className="repr-ear" key={`${evidence.gallery_photo_id}-${evidence.side}`}>
                <ZoomImage
                  src={imageUrl(evidence.gallery_crop_path)}
                  alt={`${evidence.side} reference evidence`}
                  caption={`${decision.elephantName} - ${evidence.gallery_photo_id} (${evidence.side})`}
                />
                <figcaption>
                  {evidence.gallery_photo_id} · {evidence.side}
                </figcaption>
              </figure>
            ))}
          </div>
        </div>
      )}

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
