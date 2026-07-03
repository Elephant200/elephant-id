import { useEffect, useState } from 'react';
import AnalyzePage from './AnalyzePage.jsx';
import ConfirmPage from './ConfirmPage.jsx';
import MatchPage from './MatchPage.jsx';
import ReviewPage from './ReviewPage.jsx';

const PROCESS_STEPS = [
  { id: 'analyze', label: 'Analyze' },
  { id: 'review', label: 'Evidence review' },
  { id: 'match', label: 'Match' },
  { id: 'confirm', label: 'Confirm decision' },
];

// One-line, state-aware hint describing what the current stage needs next.
function stageHelper(stage, sighting) {
  switch (stage) {
    case 'analyze':
      return sighting?.status === 'ready'
        ? 'Analysis complete — continue to evidence review'
        : 'Segmenting bodies and ears and tracing tear profiles';
    case 'review':
      return 'Select one left and one right ear, then approve';
    case 'match':
      return sighting?.match
        ? 'Choose an existing match, a new elephant, or unresolved'
        : 'Run ranking, then choose a decision';
    case 'confirm':
      return sighting?.decision
        ? 'Decision saved — start a new sighting when ready'
        : 'Check the summary and save the decision';
    default:
      return '';
  }
}

function stageForSighting(sighting) {
  if (!sighting) return 'analyze';
  if (sighting.status === 'analyzing' || sighting.status === 'failed') return 'analyze';
  if (sighting.decision) return 'confirm';
  if (sighting.match) return 'match';
  if (sighting.approved_evidence) return 'match';
  return 'review';
}

function stageEnabled(stage, sighting) {
  if (!sighting) return false;
  if (stage === 'analyze') return true;
  if (stage === 'review') return sighting.status === 'ready' || Boolean(sighting.decision);
  if (stage === 'match') {
    return sighting.status === 'ready' && Boolean(sighting.approved_evidence);
  }
  if (stage === 'confirm') {
    return Boolean(sighting.decision || sighting.match || sighting.approved_evidence);
  }
  return false;
}

const STAT_STAGES = [
  { key: 'analyzing', label: 'Analyzing' },
  { key: 'review', label: 'Needs review' },
  { key: 'match', label: 'Ready to match' },
  { key: 'decided', label: 'Decided' },
];

function statusOf(record) {
  return record.workflow_status || record.status || '';
}

// Bucket a sighting into one queue stat using the same precedence the sidecar
// applies in workflow_status(): decision > analysis state > evidence > match.
function stageKey(record) {
  if (record.decision) return 'decided';
  if (record.status === 'analyzing' || record.status === 'failed') return 'analyzing';
  if (!record.approved_evidence) return 'review';
  return 'match';
}

function queueStats(sightings) {
  const counts = { analyzing: 0, review: 0, match: 0, decided: 0 };
  sightings.forEach((record) => {
    counts[stageKey(record)] += 1;
  });
  return counts;
}

export default function QueuePage(props) {
  const {
    sightings,
    sighting,
    activeSightingId,
    setActiveSightingId,
    setSighting,
    selectedCandidate,
    setSelectedCandidate,
    pendingDecision,
    setPendingDecision,
    openSighting,
    setRoute,
  } = props;
  const [stage, setStage] = useState(stageForSighting(sighting));

  useEffect(() => {
    setStage(stageForSighting(sighting));
  }, [activeSightingId, sighting?.status, sighting?.approved_evidence, sighting?.match, sighting?.decision]);

  const backToQueue = () => {
    setActiveSightingId(null);
    setSighting(null);
    setSelectedCandidate(null);
    setPendingDecision(null);
  };

  if (!activeSightingId || !sighting) {
    const stats = queueStats(sightings);
    return (
      <div>
        <div className="page-header">
          <div>
            <h1 className="screen-title">Task queue</h1>
            <p className="screen-sub" style={{ marginBottom: 0 }}>
              Sightings waiting for evidence review, matching, or an identity
              decision. Open a job to continue the workflow.
            </p>
          </div>
          <button
            type="button"
            className="btn primary btn-lg"
            data-testid="queue-import"
            onClick={() => setRoute && setRoute('import')}
          >
            + Import sighting folder
          </button>
        </div>

        <div className="stats-strip" data-testid="queue-stats">
          {STAT_STAGES.map((entry) => (
            <div className="stats-chip" key={entry.key} data-testid={`stat-${entry.key}`}>
              <div className="stats-chip-value">{stats[entry.key]}</div>
              <div className="stats-chip-label">{entry.label}</div>
            </div>
          ))}
        </div>

        {sightings.length === 0 ? (
          <div className="empty-note">
            No sightings yet — import a grouped sighting folder to begin.
          </div>
        ) : (
          <div className="queue-list" data-testid="queue-list">
            {sightings.map((record) => {
              const analyzing = record.status === 'analyzing';
              const progress = record.progress || {};
              const percent =
                progress.total > 0
                  ? Math.round((progress.processed / progress.total) * 100)
                  : 0;
              const photoCount = record.photos?.length ?? progress.total ?? 0;
              return (
                <button
                  type="button"
                  key={record.sighting_id}
                  className="queue-row"
                  onClick={() => openSighting(record)}
                  data-testid={`queue-${record.sighting_id}`}
                >
                  <span className="queue-main">
                    <span className="queue-title">{record.folder_name}</span>
                    <span className="queue-sub">
                      {new Date(record.created_at).toLocaleDateString()} ·{' '}
                      {photoCount} photo{photoCount === 1 ? '' : 's'} ·{' '}
                      {record.profile_count} tear profile
                      {record.profile_count === 1 ? '' : 's'}
                    </span>
                    {analyzing && (
                      <span className="queue-progress">
                        <span className="progress-track">
                          <span
                            className="progress-fill"
                            style={{ width: `${percent}%` }}
                          />
                        </span>
                        <span className="queue-progress-caption">
                          {progress.processed || 0} / {progress.total || 0} photos
                        </span>
                      </span>
                    )}
                  </span>
                  <span className="queue-status">
                    <span className={`badge ${record.decision ? 'ok' : ''}`}>
                      {statusOf(record)}
                    </span>
                    {record.next_action && (
                      <span className="mono-dim">{record.next_action}</span>
                    )}
                  </span>
                </button>
              );
            })}
          </div>
        )}
      </div>
    );
  }

  const detailProps = {
    ...props,
    setStage,
    pendingDecision,
    setPendingDecision,
  };

  return (
    <div>
      <div className="process-head">
        <button type="button" className="btn ghost" onClick={backToQueue}>
          Back to queue
        </button>
        <div>
          <h1 className="screen-title">{sighting.folder_name}</h1>
          <div className="mono-dim">{sighting.workflow_status || sighting.status}</div>
        </div>
      </div>

      <div className="stage-rail">
        {PROCESS_STEPS.map((step, index) => {
          const enabled =
            stageEnabled(step.id, sighting) ||
            (step.id === 'confirm' && Boolean(pendingDecision));
          return (
            <button
              type="button"
              key={step.id}
              className={`stage-pill ${stage === step.id ? 'active' : ''}`}
              disabled={!enabled}
              onClick={() => setStage(step.id)}
            >
              <span>{String(index + 1).padStart(2, '0')}</span>
              {step.label}
            </button>
          );
        })}
      </div>

      <div className="stage-helper" data-testid="stage-helper">
        <span className="stage-helper-label">Next</span>
        {stageHelper(stage, sighting)}
      </div>

      {stage === 'analyze' && <AnalyzePage {...detailProps} />}
      {stage === 'review' && <ReviewPage {...detailProps} />}
      {stage === 'match' && <MatchPage {...detailProps} />}
      {stage === 'confirm' && <ConfirmPage {...detailProps} />}
    </div>
  );
}
