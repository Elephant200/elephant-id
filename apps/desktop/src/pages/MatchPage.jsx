import { useState } from 'react';
import { imageUrl, matchSighting } from '../api.js';
import { ZoomImage } from '../components/Lightbox.jsx';
import ProfileChart from '../components/ProfileChart.jsx';

const QUERY_COLOR = '#3e5a41';
const CATALOG_COLOR = '#b07c26';

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
  setRoute,
  selectedCandidate,
  setSelectedCandidate,
}) {
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  if (!sighting || sighting.status !== 'ready' || sighting.profile_count === 0) {
    return (
      <div>
        <h1 className="screen-title">Match against catalog</h1>
        <div className="empty-note">REVIEW A SIGHTING WITH EAR PROFILES FIRST</div>
      </div>
    );
  }

  const runMatch = async () => {
    setError(null);
    setBusy(true);
    try {
      setSighting(await matchSighting(sighting.sighting_id));
      setSelectedCandidate(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const queryCrops = new Map();
  sighting.photos.forEach((photo) =>
    photo.ears.forEach((ear) => queryCrops.set(`${photo.photo_id}:${ear.side}`, ear.crop_path)),
  );
  const candidates = sighting.match?.candidates ?? null;

  return (
    <div>
      <h1 className="screen-title">Match against catalog</h1>
      <p className="screen-sub">
        The engine compares this sighting&apos;s ear tear patterns against every
        known elephant and ranks the catalog. Select the candidate you judge to
        be the match — or continue to filing to enroll a new individual.
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
            {sighting.sides.length > 0 && ` (${sighting.sides.join(' + ')})`} will be ranked
            against the catalog.
          </p>
          <button
            type="button"
            className="btn primary"
            data-testid="run-match"
            disabled={busy || !engineReady}
            onClick={runMatch}
          >
            {busy ? 'Matching…' : engineReady ? 'Rank catalog matches' : 'Engine warming up…'}
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
          <div data-testid="candidates">
            {candidates.map((candidate, index) => {
              const selected = selectedCandidate === candidate.identity;
              return (
                <div
                  key={candidate.identity}
                  className={`candidate ${selected ? 'selected' : ''}`}
                  data-testid={`candidate-${candidate.identity}`}
                  onClick={() =>
                    setSelectedCandidate(selected ? null : candidate.identity)
                  }
                >
                  <div className="rank">{String(index + 1).padStart(2, '0')}</div>
                  <div>
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
                    <div className="evidence-row">
                      {candidate.evidence.map((evidence) => {
                        const queryCrop = queryCrops.get(
                          `${evidence.query_photo_id}:${evidence.side}`,
                        );
                        return (
                          <div className="evidence" key={evidence.side}>
                            <div className="evidence-pair">
                              <figure>
                                {queryCrop ? (
                                  <ZoomImage
                                    src={imageUrl(queryCrop)}
                                    alt={`query ${evidence.side}`}
                                    caption={`This sighting — ${evidence.query_photo_id} (${evidence.side} ear)`}
                                  />
                                ) : (
                                  <div className="noimg">NO CROP</div>
                                )}
                                <figcaption>THIS SIGHTING · {evidence.query_photo_id}</figcaption>
                              </figure>
                              <figure>
                                {evidence.gallery_crop_path ? (
                                  <ZoomImage
                                    src={imageUrl(evidence.gallery_crop_path)}
                                    alt={`catalog ${evidence.side}`}
                                    caption={`Catalog — ${evidence.gallery_photo_id} (${evidence.side} ear, ${evidence.gallery_date})`}
                                  />
                                ) : (
                                  <div className="noimg">NO CROP</div>
                                )}
                                <figcaption>CATALOG · {evidence.gallery_photo_id}</figcaption>
                              </figure>
                            </div>
                            {evidence.query_profile?.length > 0 && (
                              <div className="evidence-chart">
                                <ProfileChart
                                  series={[
                                    {
                                      label: 'sighting',
                                      values: evidence.query_profile,
                                      color: QUERY_COLOR,
                                      fill: true,
                                    },
                                    {
                                      label: 'catalog',
                                      values: evidence.gallery_profile,
                                      color: CATALOG_COLOR,
                                    },
                                  ]}
                                  height={72}
                                />
                                <div className="chart-legend">
                                  <span
                                    className="legend-swatch"
                                    style={{ background: QUERY_COLOR }}
                                  />
                                  this sighting
                                  <span
                                    className="legend-swatch"
                                    style={{ background: CATALOG_COLOR }}
                                  />
                                  catalog
                                </div>
                              </div>
                            )}
                            <div className="evidence-meta">
                              <span
                                className={`badge ${evidence.side}`}
                                title={`calibrated score ${evidence.score.toFixed(2)}`}
                              >
                                {evidence.side} ear · {earStrength(evidence)}
                              </span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                  <div className="candidate-actions">
                    <button
                      type="button"
                      className={`btn ${selected ? 'ok' : ''}`}
                      data-testid={`select-${candidate.identity}`}
                      onClick={(event) => {
                        event.stopPropagation();
                        setSelectedCandidate(selected ? null : candidate.identity);
                      }}
                    >
                      {selected ? '✓ Selected' : 'Select'}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="panel" style={{ marginTop: 8 }}>
            <div className="decision-bar">
              <span className="mono-dim">
                {selectedCandidate
                  ? `SELECTED: ${selectedCandidate} — CONFIRM IT ON THE FILING PAGE`
                  : 'NO CANDIDATE SELECTED — YOU CAN ENROLL A NEW INDIVIDUAL OR LEAVE THE SIGHTING UNRESOLVED AT FILING'}
              </span>
              <span style={{ flex: 1 }} />
              <button
                type="button"
                className="btn primary"
                data-testid="to-file"
                onClick={() => setRoute('file')}
              >
                Continue to filing
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
