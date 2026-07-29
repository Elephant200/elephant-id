import { useState } from 'react';
import { imageUrl, matchSighting } from '../api.js';
import { ZoomImage } from '../components/Lightbox.jsx';

// Strength labels are computed by the engine from its impostor-score
// distribution; the fallback covers match results stored before that existed.
function earStrength(evidence) {
  if (evidence.strength) return evidence.strength;
  if (evidence.score >= 2) return 'strong';
  if (evidence.score >= 0.5) return 'moderate';
  return 'weak';
}

function rationale(evidence) {
  return evidence
    .map((entry) => `${earStrength(entry)} ${entry.side}-ear agreement`)
    .join(' · ');
}

export default function MatchPage({
  engineReady,
  sighting,
  setSighting,
  setStage,
  pendingDecision,
  selectedCandidate,
  setSelectedCandidate,
  setPendingDecision,
}) {
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [newName, setNewName] = useState('');
  const [decisionMode, setDecisionMode] = useState(pendingDecision?.action ?? null);

  if (!sighting || sighting.status !== 'ready' || !sighting.approved_evidence) {
    return (
      <div>
        <h1 className="screen-title">Match against known-elephant catalog</h1>
        <div className="empty-note">APPROVE LEFT AND RIGHT EAR EVIDENCE FIRST</div>
      </div>
    );
  }

  const runMatch = async () => {
    setError(null);
    setBusy(true);
    try {
      setSighting(await matchSighting(sighting.sighting_id));
      setSelectedCandidate(null);
      setPendingDecision(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const queryEvidence = new Map(
    ['left', 'right'].map((side) => {
      const evidence = sighting.approved_evidence?.[side];
      return [`${evidence?.photo_id}:${side}`, evidence];
    }),
  );
  // Approved query evidence carries a human-readable file name; prefer it over
  // the opaque generated photo id in query-side captions.
  const queryLabels = new Map(
    ['left', 'right'].map((side) => {
      const evidence = sighting.approved_evidence?.[side];
      return [side, evidence?.file_name || evidence?.photo_id || ''];
    }),
  );
  const candidates = sighting.match?.candidates ?? null;
  const selectedMatch = candidates?.find(
    (candidate) => candidate.identity === selectedCandidate,
  );
  const currentDecision =
    decisionMode === 'existing_known_elephant' && selectedMatch
      ? {
          action: 'existing_known_elephant',
          elephantName: selectedMatch.identity,
          candidate: selectedMatch,
          label: `Existing known elephant: ${selectedMatch.identity}`,
        }
      : decisionMode === 'new_known_elephant' && newName.trim()
        ? {
            action: 'new_known_elephant',
            elephantName: newName.trim(),
            label: `New known elephant: ${newName.trim()}`,
          }
        : decisionMode === 'unresolved'
          ? { action: 'unresolved', elephantName: null, label: 'Unresolved' }
          : null;

  const proceed = () => {
    if (!currentDecision) return;
    const { label, ...decision } = currentDecision;
    setPendingDecision(decision);
    setStage('confirm');
  };

  const selectExisting = (candidate, selected) => {
    setSelectedCandidate(selected ? null : candidate.identity);
    setDecisionMode(selected ? null : 'existing_known_elephant');
  };

  return (
    <div>
      <h1 className="screen-title">Match against known-elephant catalog</h1>
      <p className="screen-sub">
        The engine compares this sighting&apos;s ear tear patterns against every
        known elephant and ranks the known-elephant catalog. Select an existing
        known elephant, create a new known elephant, or leave the sighting
        unresolved.
      </p>

      {error && (
        <div className="error-note" data-testid="error-note">
          {error}
        </div>
      )}

      {!candidates && (
        <div className="panel">
          <div className="panel-title">RUN MATCHING</div>
          <p className="screen-sub" style={{ marginBottom: 14 }}>
            {sighting.profile_count} ear profile
            {sighting.profile_count === 1 ? '' : 's'}
            {' '}were extracted. Only the approved left and right tear profiles
            will be ranked against the known-elephant catalog.
          </p>
          <button
            type="button"
            className="btn primary"
            data-testid="run-match"
            disabled={busy || !engineReady}
            onClick={runMatch}
          >
            {busy ? 'Matching…' : engineReady ? 'Rank known-elephant matches' : 'Engine warming up…'}
          </button>
        </div>
      )}

      {candidates && (
        <>
          <div className="panel-title" style={{ marginBottom: 12 }}>
            RANKED CANDIDATES
            <button type="button" className="btn ghost" disabled={busy} onClick={runMatch}>
              Re-run
            </button>
          </div>
          {candidates.length === 0 && (
            <div className="empty-note">NO CANDIDATES — CATALOG HAS NO SAME-SIDE EVIDENCE</div>
          )}
          <div className="candidate-list" data-testid="candidates">
            {candidates.map((candidate, index) => {
              const selected = selectedCandidate === candidate.identity;
              return (
                <div
                  key={candidate.identity}
                  className={`candidate ${selected ? 'selected' : ''}`}
                  data-testid={`candidate-${candidate.identity}`}
                  onClick={() => selectExisting(candidate, selected)}
                >
                  <div className="candidate-head">
                    <div className="rank">{String(index + 1).padStart(2, '0')}</div>
                    <div className="candidate-headline">
                      <div className="candidate-name">{candidate.identity}</div>
                      <div className="candidate-sub">
                        MATCH STRENGTH {(candidate.confidence * 100).toFixed(1)}%
                        <span className="rationale"> — {rationale(candidate.evidence)}</span>
                      </div>
                      <div
                        className="meter"
                        title={`combined calibrated score ${candidate.score.toFixed(2)}`}
                      >
                        <div className="meter-track">
                          <div
                            className="meter-fill"
                            style={{ width: `${Math.round(candidate.confidence * 100)}%` }}
                          />
                        </div>
                      </div>
                    </div>
                    <div className="candidate-select">
                      <button
                        type="button"
                        className={`btn ${selected ? 'ok' : ''}`}
                        data-testid={`select-${candidate.identity}`}
                        onClick={(event) => {
                          event.stopPropagation();
                          selectExisting(candidate, selected);
                        }}
                      >
                        {selected ? '✓ Selected' : 'Select'}
                      </button>
                    </div>
                  </div>
                  <div className="evidence-row">
                      {candidate.evidence.map((evidence) => {
                        const approved = queryEvidence.get(
                          `${evidence.query_photo_id}:${evidence.side}`,
                        );
                        const queryLabel =
                          queryLabels.get(evidence.side) || evidence.query_photo_id;
                        const queryCrop = approved?.display_crop_path || approved?.crop_path;
                        const galleryCrop =
                          evidence.gallery_display_crop_path || evidence.gallery_crop_path;
                        return (
                          <div className="evidence" key={evidence.side}>
                            <div className="evidence-side">
                              <span className={`badge ${evidence.side}`}>{evidence.side}</span>
                              <strong>{earStrength(evidence)}</strong>
                              <span>{evidence.score.toFixed(2)}</span>
                            </div>
                            <div className="evidence-pair">
                              <figure>
                                {queryCrop ? (
                                  <ZoomImage
                                    src={imageUrl(queryCrop)}
                                    alt={`query ${evidence.side}`}
                                    caption={`This sighting — ${queryLabel} (${evidence.side} ear)`}
                                  />
                                ) : (
                                  <div className="noimg">NO CROP</div>
                                )}
                                <figcaption>THIS SIGHTING · {queryLabel}</figcaption>
                              </figure>
                              <figure>
                                {galleryCrop ? (
                                  <ZoomImage
                                    src={imageUrl(galleryCrop)}
                                    alt={`catalog ${evidence.side}`}
                                    caption={`Known-elephant catalog - ${evidence.gallery_photo_id} (${evidence.side} ear, ${evidence.gallery_date})`}
                                  />
                                ) : (
                                  <div className="noimg">NO CROP</div>
                                )}
                                <figcaption>KNOWN-ELEPHANT CATALOG · {evidence.gallery_photo_id}</figcaption>
                              </figure>
                            </div>
                            {evidence.profile_plot_path && (
                              <div className="evidence-chart">
                                <img
                                  src={imageUrl(evidence.profile_plot_path)}
                                  alt={`Aligned ${evidence.side} ear tear profiles for ${candidate.identity}`}
                                  className="profile-plot-image match-profile-plot"
                                />
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
              );
            })}
          </div>

          <div className="panel new-elephant-panel">
            <div className="panel-title">Register new known elephant</div>
            <p className="screen-sub" style={{ marginBottom: 12 }}>
              Use this only after reviewing all ranked candidates and deciding none
              of them are the sighted elephant.
            </p>
            <div className="new-elephant-row">
              <input
                className="enroll-input"
                placeholder="New known elephant name"
                value={newName}
                onChange={(event) => setNewName(event.target.value)}
                data-testid="new-elephant-name"
              />
              <button
                type="button"
                className="btn primary"
                disabled={!newName.trim()}
                data-testid="choose-new"
                onClick={() => {
                  setSelectedCandidate(null);
                  setDecisionMode('new_known_elephant');
                }}
              >
                Use new elephant
              </button>
            </div>
          </div>

          <div className="action-bar" data-testid="match-action-bar">
            <div className="action-bar-status">
              <div className="action-bar-line">
                {currentDecision ? (
                  <>
                    <span className="side-tag">CURRENT DECISION </span>
                    <span className="ok-tick">✓</span> {currentDecision.label}
                  </>
                ) : (
                  <span className="missing">No identity decision selected</span>
                )}
              </div>
              <div className="action-bar-hint">
                Select a candidate, choose unresolved, or register a new elephant below the list
              </div>
            </div>
            <button
              type="button"
              className="btn ghost"
              data-testid="choose-unresolved"
              onClick={() => {
                setSelectedCandidate(null);
                setDecisionMode('unresolved');
              }}
            >
              Unresolved
            </button>
            <button
              type="button"
              className="btn primary"
              data-testid="choose-existing"
              disabled={!currentDecision}
              onClick={proceed}
            >
              Continue to confirmation
            </button>
          </div>
        </>
      )}
    </div>
  );
}
