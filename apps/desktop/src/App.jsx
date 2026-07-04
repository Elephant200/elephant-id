import { useCallback, useEffect, useState } from 'react';
import {
  Activity,
  FolderPlus,
  GalleryHorizontalEnd,
  Image,
  ListChecks,
  Scale,
  ScanSearch,
  Settings,
} from 'lucide-react';
import elephantsAliveLogo from '../src/assets/elephants-alive.png';
import { getHealth, getSighting, listSightings } from './api.js';
import { LightboxProvider } from './components/Lightbox.jsx';
import CatalogPage from './pages/CatalogPage.jsx';
import DevPhotoAnalysisPage from './pages/DevPhotoAnalysisPage.jsx';
import DevSam3Page from './pages/DevSam3Page.jsx';
import DevTearMatchPage from './pages/DevTearMatchPage.jsx';
import DevTearProfilePage from './pages/DevTearProfilePage.jsx';
import ImportPage from './pages/ImportPage.jsx';
import QueuePage from './pages/QueuePage.jsx';
import SettingsPage from './pages/SettingsPage.jsx';

const CORE_NAV_ITEMS = [
  { id: 'import', label: 'Import', icon: FolderPlus },
  { id: 'queue', label: 'Queue', icon: ListChecks },
  { id: 'catalog', label: 'Catalog', icon: GalleryHorizontalEnd },
];

const DEV_NAV_ITEMS = [
  { id: 'dev-sam3', label: 'SAM3', icon: ScanSearch },
  { id: 'dev-tear-profile', label: 'Tear Profile', icon: Activity },
  { id: 'dev-tear-match', label: 'Tear Match', icon: Scale },
  { id: 'dev-photo-analysis', label: 'Photo Analysis', icon: Image },
];

export default function App() {
  const [health, setHealth] = useState(null);
  const [route, setRoute] = useState('import');
  const [sightings, setSightings] = useState([]);
  const [activeSightingId, setActiveSightingId] = useState(null);
  const [sighting, setSighting] = useState(null);
  const [selectedCandidate, setSelectedCandidate] = useState(null);
  const [pendingDecision, setPendingDecision] = useState(null);

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
    setPendingDecision(null);
    setRoute('queue');
  };

  const startNewSighting = () => {
    setActiveSightingId(null);
    setSighting(null);
    setSelectedCandidate(null);
    setPendingDecision(null);
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
    health,
    sightings,
    sighting,
    setSighting,
    activeSightingId,
    setActiveSightingId,
    refreshSightings,
    setRoute,
    selectedCandidate,
    setSelectedCandidate,
    pendingDecision,
    setPendingDecision,
    startNewSighting,
  };

  const clearActive = () => {
    setActiveSightingId(null);
    setSighting(null);
    setSelectedCandidate(null);
    setPendingDecision(null);
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
          <div className="side-label">Core</div>
          {CORE_NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`nav-btn ${route === item.id ? 'active' : ''}`}
              data-testid={`nav-${item.id}`}
              onClick={() => {
                if (item.id === 'import') {
                  startNewSighting();
                  return;
                }
                if (item.id === 'queue') {
                  clearActive();
                }
                setRoute(item.id);
              }}
            >
              <item.icon className="nav-icon" aria-hidden="true" />
              {item.label}
            </button>
          ))}
        </div>

        <div className="side-section dev-section">
          <div className="side-label">Development</div>
          {DEV_NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`nav-btn dev-nav ${route === item.id ? 'active' : ''}`}
              data-testid={`nav-${item.id}`}
              onClick={() => setRoute(item.id)}
            >
              <item.icon className="nav-icon" aria-hidden="true" />
              {item.label}
            </button>
          ))}
        </div>

        <div className="side-section settings-section">
          <button
            type="button"
            className={`nav-btn settings-nav ${route === 'settings' ? 'active' : ''}`}
            data-testid="nav-settings"
            onClick={() => setRoute('settings')}
          >
            <Settings className="nav-icon" aria-hidden="true" />
            Settings
          </button>
        </div>

        <div className="sidebar-brand">
          <img
            className="sidebar-brand-logo"
            src={elephantsAliveLogo}
            alt="Elephants Alive"
          />
          <div className="sidebar-brand-caption">Field ID Console · v0.1.0</div>
        </div>
      </aside>

      <main className="main">
        {route === 'import' && <ImportPage {...pageProps} />}
        {route === 'queue' && <QueuePage {...pageProps} openSighting={openSighting} />}
        {route === 'catalog' && <CatalogPage />}
        {route === 'dev-sam3' && <DevSam3Page engineReady={engineReady} />}
        {route === 'dev-tear-profile' && <DevTearProfilePage engineReady={engineReady} />}
        {route === 'dev-tear-match' && <DevTearMatchPage engineReady={engineReady} />}
        {route === 'dev-photo-analysis' && <DevPhotoAnalysisPage engineReady={engineReady} />}
        {route === 'settings' && <SettingsPage health={health} />}
      </main>
    </div>
    </LightboxProvider>
  );
}
