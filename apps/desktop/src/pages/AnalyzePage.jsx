export default function AnalyzePage({ sighting, setStage }) {
  if (!sighting) {
    return <EmptyState message="IMPORT A SIGHTING FOLDER FIRST" />;
  }

  const { processed, total } = sighting.progress;
  const percent = total > 0 ? Math.round((processed / total) * 100) : 0;
  const counts = { analyzed: 0, precomputed: 0, skipped: 0 };
  sighting.photos.forEach((photo) => {
    counts[photo.status] = (counts[photo.status] || 0) + 1;
  });

  return (
    <div>
      <h1 className="screen-title">Analyze photos</h1>
      <p className="screen-sub">
        Each source photo is analyzed into an analysis package with suggested
        ear evidence and tear profiles for reviewer approval.
      </p>

      {sighting.status === 'failed' && (
        <div className="error-note">Analysis failed: {sighting.error || 'unknown error'}</div>
      )}

      <div className="panel" data-testid="analyzing">
        <div className="panel-title">
          {sighting.status === 'analyzing' ? 'ANALYZING' : 'ANALYSIS COMPLETE'} ·{' '}
          {sighting.folder_name}
        </div>
        <div className="progress-wrap">
          <div className="progress-track">
            <div
              className={`progress-fill ${sighting.status === 'ready' ? 'done' : ''}`}
              style={{ width: `${percent}%` }}
            />
          </div>
          <div className="progress-caption">
            {processed} / {total} PHOTOS
            {sighting.status === 'analyzing'
              ? ' · SEGMENTING BODIES AND EARS · TRACING TEAR PROFILES'
              : ` · ${sighting.profile_count} EAR PROFILES EXTRACTED`}
          </div>
        </div>

        {sighting.status === 'ready' && (
          <>
            <div className="stat-row">
              <div className="stat">
                <div className="stat-value">{counts.analyzed || 0}</div>
                <div className="stat-label">analyzed</div>
              </div>
              <div className="stat">
                <div className="stat-value">{counts.precomputed || 0}</div>
                <div className="stat-label">reused</div>
              </div>
              <div className="stat">
                <div className="stat-value">{counts.skipped || 0}</div>
                <div className="stat-label">skipped</div>
              </div>
              <div className="stat">
                <div className="stat-value">{sighting.profile_count}</div>
                <div className="stat-label">ear profiles</div>
              </div>
            </div>
            <button
              type="button"
              className="btn primary"
              data-testid="to-review"
              onClick={() => setStage('review')}
            >
              Continue to review
            </button>
          </>
        )}
      </div>
    </div>
  );
}

function EmptyState({ message }) {
  return (
    <div>
      <h1 className="screen-title">Analyze photos</h1>
      <div className="empty-note">{message}</div>
    </div>
  );
}
