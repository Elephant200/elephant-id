import { useEffect, useState } from 'react';
import { approveEvidence, getAnalysis, imageUrl } from '../api.js';
import ContourEditor from '../components/ContourEditor.jsx';
import { ZoomImage } from '../components/Lightbox.jsx';
import PhotoEvidenceCard from '../components/PhotoEvidenceCard.jsx';

const SIDES = ['left', 'right'];

export default function ReviewPage({
  sighting,
  setSighting,
  refreshSightings,
  setStage,
  setPendingDecision,
  onEditContour,
}) {
  const [analysis, setAnalysis] = useState(null);
  const [selected, setSelected] = useState({ left: null, right: null });
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [editingContour, setEditingContour] = useState(null); // { side, candidate }

  const openContourEditor = (side, candidate) => {
    if (onEditContour) onEditContour(side, candidate);
    else setEditingContour({ side, candidate });
  };

  useEffect(() => {
    let cancelled = false;
    if (!sighting || sighting.status !== 'ready') {
      setAnalysis(null);
      return undefined;
    }
    getAnalysis(sighting.sighting_id)
      .then((payload) => {
        if (cancelled) return;
        setAnalysis(payload);
        setSelected({
          left: payload.approved_evidence?.left?.candidate_id || null,
          right: payload.approved_evidence?.right?.candidate_id || null,
        });
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [sighting]);

  if (!sighting || sighting.status !== 'ready') {
    return (
      <div>
        <h1 className="screen-title">Evidence review</h1>
        <div className="empty-note">ANALYZE A SIGHTING FIRST</div>
      </div>
    );
  }

  const candidates = analysis?.ear_candidates || { left: [], right: [] };
  const canApprove = SIDES.every((side) => selected[side]);
  const missingSides = SIDES.filter((side) => candidates[side].length === 0);

  const selectedCandidateFor = (side) =>
    candidates[side].find((item) => item.candidate_id === selected[side]) || null;

  const approve = async () => {
    setError(null);
    setBusy(true);
    try {
      const updated = await approveEvidence(sighting.sighting_id, selected.left, selected.right);
      setSighting(updated);
      refreshSightings();
      setStage('match');
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const leaveUnresolved = () => {
    setPendingDecision({ action: 'unresolved', elephantName: null });
    setStage('confirm');
  };

  return (
    <div>
      <h1 className="screen-title">Evidence review</h1>
      <p className="screen-sub">
        Select one left ear and one right ear for the analysis package, then
        approve the evidence to continue to matching.
      </p>

      {error && (
        <div className="error-note" data-testid="error-note">
          {error}
        </div>
      )}

      {missingSides.length > 0 && (
        <div className="error-note" data-testid="missing-side-blocker">
          V1 matching requires approved evidence for both sides. No valid{' '}
          {missingSides.join(' or ')} ear candidate passed the preview filter, so
          this sighting can only be saved as unresolved.
        </div>
      )}

      <div className="side-columns review-candidate-grid">
        {SIDES.map((side) => {
          const contour = selectedCandidateFor(side);
          return (
            <section className="panel" key={side}>
              <div className="panel-title">
                {side} ear
                <span className={`badge ${side}`}>
                  {candidates[side].length} candidate{candidates[side].length === 1 ? '' : 's'}
                </span>
              </div>
              {candidates[side].length === 0 ? (
                <div className="empty-note">NO VALID {side.toUpperCase()} EAR CANDIDATES</div>
              ) : (
                <>
                  <div className="panel-footnote">
                    Ranking is a temporary preview heuristic based on crop aspect
                    ratio and pixel area.
                  </div>
                  <div className="candidate-tiles">
                    {candidates[side].map((candidate) => (
                      <button
                        type="button"
                        key={candidate.candidate_id}
                        className={`ear-candidate ${
                          selected[side] === candidate.candidate_id ? 'selected' : ''
                        }`}
                        data-testid={`candidate-${side}-${candidate.profile_row_index}`}
                        onClick={() =>
                          setSelected((current) => ({
                            ...current,
                            [side]: candidate.candidate_id,
                          }))
                        }
                      >
                        <span className="candidate-crop">
                          <ZoomImage
                            src={imageUrl(candidate.crop_path)}
                            alt={`${side} ear candidate`}
                            caption={`${candidate.file_name} - ${side} ear candidate`}
                          />
                        </span>
                        <span className="candidate-detail">
                          <span className="candidate-name">{candidate.file_name}</span>
                          <span className="candidate-sub">
                            aspect {candidate.aspect_ratio.toFixed(2)} ·{' '}
                            {candidate.pixel_area.toLocaleString()} px
                          </span>
                        </span>
                      </button>
                    ))}
                  </div>
                  {contour && contour.contour && (
                    <div className="contour-hook">
                      <ZoomImage
                        src={imageUrl(contour.crop_path)}
                        alt={`${side} selected crop`}
                        caption={`${contour.file_name} - selected ${side} ear`}
                      />
                      <div className="contour-hook-detail">
                        <span className="candidate-name">{contour.file_name}</span>
                        <span className="mono-dim">
                          Adjust the approved crop and ear outline before matching.
                        </span>
                      </div>
                      <button
                        type="button"
                        className="btn"
                        data-testid={`edit-contour-${side}`}
                        onClick={() => openContourEditor(side, contour)}
                      >
                        Open contour editor
                      </button>
                    </div>
                  )}
                </>
              )}
            </section>
          );
        })}
      </div>

      <details className="disclosure">
        <summary>Source photos &amp; full analysis ({sighting.photos.length} photos)</summary>
        <div className="disclosure-body">
          {sighting.photos.map((photo) => (
            <PhotoEvidenceCard key={photo.file_name} photo={photo} />
          ))}
        </div>
      </details>

      <div className="action-bar" data-testid="review-action-bar">
        <div className="action-bar-status">
          <div className="action-bar-line">
            {SIDES.map((side, index) => {
              const candidate = selectedCandidateFor(side);
              return (
                <span key={side}>
                  {index > 0 && <span className="side-tag"> · </span>}
                  <span className="side-tag">{side.toUpperCase()} </span>
                  {candidate ? (
                    <>
                      <span className="ok-tick">✓</span> {candidate.file_name}
                    </>
                  ) : (
                    <span className="missing">— not selected</span>
                  )}
                </span>
              );
            })}
          </div>
          <div className="action-bar-hint">
            {canApprove
              ? 'Both ears selected — ready to approve'
              : 'Select one left and one right ear'}
          </div>
        </div>
        {missingSides.length > 0 && (
          <button type="button" className="btn ghost" onClick={leaveUnresolved}>
            Leave unresolved
          </button>
        )}
        <button
          type="button"
          className="btn primary"
          data-testid="approve-evidence"
          disabled={busy || !canApprove}
          onClick={approve}
        >
          {busy ? 'Approving evidence...' : 'Approve evidence'}
        </button>
      </div>

      {editingContour && (
        <ContourEditor
          side={editingContour.side}
          candidate={editingContour.candidate}
          onClose={() => setEditingContour(null)}
        />
      )}
    </div>
  );
}
