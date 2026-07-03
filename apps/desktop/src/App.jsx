import { useCallback, useEffect, useState } from 'react';
import { getHealth, getSighting, listSightings } from './api.js';
import { LightboxProvider } from './components/Lightbox.jsx';
import AnalyzePage from './pages/AnalyzePage.jsx';
import CatalogPage from './pages/CatalogPage.jsx';
import FilePage from './pages/FilePage.jsx';
import ImportPage from './pages/ImportPage.jsx';
import LabPage from './pages/LabPage.jsx';
import MatchPage from './pages/MatchPage.jsx';
import ReviewPage from './pages/ReviewPage.jsx';

// Sidebar pages follow the user-facing pipeline steps in docs/pipeline.md:
// import a folder, analyze photos, review evidence, match, file the decision.
const WORKFLOW_STEPS = [
  { id: 'import', label: 'Import' },
  { id: 'analyze', label: 'Analyze' },
  { id: 'review', label: 'Review' },
  { id: 'match', label: 'Match' },
  { id: 'file', label: 'File' },
];

function stepEnabled(stepId, sighting) {
  switch (stepId) {
    case 'import':
      return true;
    case 'analyze':
      return Boolean(sighting);
    case 'review':
      return Boolean(sighting && sighting.status === 'ready');
    case 'match':
      return Boolean(sighting && sighting.status === 'ready' && sighting.profile_count > 0);
    case 'file':
      return Boolean(sighting && (sighting.match || sighting.decision));
    default:
      return false;
  }
}

function stepForSighting(sighting) {
  if (!sighting) return 'import';
  if (sighting.status === 'analyzing' || sighting.status === 'failed') return 'analyze';
  if (sighting.decision) return 'file';
  if (sighting.match) return 'match';
  return 'review';
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [route, setRoute] = useState('import');
  const [sightings, setSightings] = useState([]);
  const [activeSightingId, setActiveSightingId] = useState(null);
  const [sighting, setSighting] = useState(null);
  const [selectedCandidate, setSelectedCandidate] = useState(null);

  const refreshSightings = useCallback(async () => {
    try {
      setSightings(await listSightings());
    } catch {
      /* sidecar not up yet */
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const status = await getHealth();
        if (!cancelled) setHealth(status);
      } catch {
        if (!cancelled) setHealth(null);
      }
    };
    poll();
    const timer = setInterval(poll, 3000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    refreshSightings();
  }, [refreshSightings]);

  // Track the active sighting, polling while analysis runs.
  useEffect(() => {
    if (!activeSightingId) {
      setSighting(null);
      return undefined;
    }
    let cancelled = false;
    let timer = null;
    const poll = async () => {
      try {
        const record = await getSighting(activeSightingId);
        if (cancelled) return;
        setSighting(record);
        if (record.status === 'analyzing') {
          timer = setTimeout(poll, 1200);
        } else {
          refreshSightings();
        }
      } catch {
        /* transient */
      }
    };
    poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [activeSightingId, refreshSightings]);

  const openSighting = (record) => {
    setActiveSightingId(record.sighting_id);
    setSelectedCandidate(null);
    setRoute(stepForSighting(record));
  };

  const startNewSighting = () => {
    setActiveSightingId(null);
    setSighting(null);
    setSelectedCandidate(null);
    setRoute('import');
  };

  if (!health) {
    return (
      <div className="loading-splash">
        <div className="brand">
          ALPHA<em>PHANT</em>
        </div>
        <div className="mono-dim">CONNECTING TO LOCAL ANALYSIS ENGINE…</div>
      </div>
    );
  }

  const engineReady = health.engine_ready;
  const pageProps = {
    engineReady,
    sighting,
    setSighting,
    setActiveSightingId,
    refreshSightings,
    setRoute,
    selectedCandidate,
    setSelectedCandidate,
    startNewSighting,
  };

  return (
    <LightboxProvider>
    <div className="shell">
      <header className="header">
        <div className="brand">
          ALPHA<em>PHANT</em>
        </div>
        <div className="brand-sub">Elephants Alive · Field ID Console</div>
        <div className="header-spacer" />
        <div className="engine-status" data-testid="engine-status">
          <span className={`led ${engineReady ? 'on' : ''}`} />
          {engineReady
            ? `ENGINE READY · ${health.elephants} ELEPHANTS · ${health.profiles} EAR PROFILES`
            : 'ENGINE WARMING UP…'}
        </div>
      </header>

      <aside className="sidebar">
        <div className="side-section">
          <div className="side-label">Workflow</div>
          {WORKFLOW_STEPS.map((step, index) => {
            const enabled = stepEnabled(step.id, sighting);
            return (
              <button
                key={step.id}
                type="button"
                className={`nav-btn ${route === step.id ? 'active' : ''}`}
                disabled={!enabled}
                data-testid={`nav-${step.id}`}
                onClick={() => setRoute(step.id)}
              >
                <span className="num">{String(index + 1).padStart(2, '0')}</span>
                {step.label}
              </button>
            );
          })}
        </div>

        <div className="side-section">
          <div className="side-label">Reference</div>
          <button
            type="button"
            className={`nav-btn ${route === 'catalog' ? 'active' : ''}`}
            data-testid="nav-catalog"
            onClick={() => setRoute('catalog')}
          >
            <span className="num">◆</span> Catalog
          </button>
          <button
            type="button"
            className={`nav-btn ${route === 'lab' ? 'active' : ''}`}
            data-testid="nav-lab"
            onClick={() => setRoute('lab')}
          >
            <span className="num">⚗</span> Lab
          </button>
        </div>

        <div className="side-section">
          <div className="side-label">Sightings</div>
          {sightings.length === 0 && (
            <div className="mono-dim">none yet — import a folder to begin</div>
          )}
          {sightings.map((record) => (
            <button
              key={record.sighting_id}
              type="button"
              className={`sighting-item ${
                record.sighting_id === activeSightingId ? 'active' : ''
              }`}
              title={record.folder}
              onClick={() => openSighting(record)}
            >
              <span className="row1">
                <span>{record.folder_name}</span>
                <span className={`badge ${record.decision ? 'ok' : ''}`}>
                  {record.decision
                    ? record.decision.elephant_name || 'unresolved'
                    : record.status}
                </span>
              </span>
              <span className="row2">
                {new Date(record.created_at).toLocaleDateString()} ·{' '}
                {record.profile_count} profiles
              </span>
            </button>
          ))}
        </div>
      </aside>

      <main className="main">
        {route === 'import' && <ImportPage {...pageProps} />}
        {route === 'analyze' && <AnalyzePage {...pageProps} />}
        {route === 'review' && <ReviewPage {...pageProps} />}
        {route === 'match' && <MatchPage {...pageProps} />}
        {route === 'file' && <FilePage {...pageProps} />}
        {route === 'catalog' && <CatalogPage />}
        {route === 'lab' && <LabPage engineReady={engineReady} />}
      </main>
    </div>
    </LightboxProvider>
  );
}
