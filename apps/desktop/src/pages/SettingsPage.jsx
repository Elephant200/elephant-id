import elephantsAliveLogo from '../assets/elephants-alive.png';
import { API_BASE } from '../api.js';

/** Read-only console showing engine, storage, and product information. */
export default function SettingsPage({ health }) {
  const engineReady = Boolean(health?.engine_ready);

  return (
    <div>
      <h1 className="screen-title">Settings</h1>
      <p className="screen-sub">
        Read-only status for the local analysis engine, the App Library on this
        machine, and the Alphaphant build. Nothing here is editable in the V1
        preview.
      </p>

      <div className="panel" data-testid="settings-engine">
        <div className="panel-title">Engine</div>
        <div className="settings-grid">
          <div className="settings-item">
            <div className="settings-key">Status</div>
            <div className="settings-value">
              <span className={`led ${engineReady ? 'on' : ''}`} />
              {engineReady ? 'Ready' : 'Warming up'}
            </div>
          </div>
          <div className="settings-item">
            <div className="settings-key">Known elephants</div>
            <div className="settings-value">
              {health?.elephants ?? '—'}
            </div>
          </div>
          <div className="settings-item">
            <div className="settings-key">Ear profiles</div>
            <div className="settings-value">
              {health?.profiles ?? '—'}
            </div>
          </div>
        </div>
        {health?.engine_error && (
          <div className="error-note" data-testid="engine-error">
            {health.engine_error}
          </div>
        )}
      </div>

      <div className="panel" data-testid="settings-storage">
        <div className="panel-title">Storage</div>
        <div className="settings-list">
          <div className="settings-item">
            <div className="settings-key">App Library path</div>
            <div className="settings-value mono-value">
              {health?.data_dir || '—'}
            </div>
          </div>
          <div className="settings-item">
            <div className="settings-key">Sidecar API</div>
            <div className="settings-value mono-value">{API_BASE}</div>
          </div>
        </div>
      </div>

      <div className="panel" data-testid="settings-about">
        <div className="panel-title">About</div>
        <div className="about-row">
          <img
            className="about-logo"
            src={elephantsAliveLogo}
            alt="Elephants Alive"
          />
          <div>
            <div className="about-name">Alphaphant</div>
            <div className="about-desc">
              Offline desktop workflow for identifying individual African
              elephants from grouped sighting folders — built for Elephants
              Alive.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
