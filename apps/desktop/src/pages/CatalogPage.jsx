import { useEffect, useState } from 'react';
import { getCatalog, getElephant, imageUrl } from '../api.js';
import { ZoomImage } from '../components/Lightbox.jsx';

export default function Catalog() {
  const [elephants, setElephants] = useState(null);
  const [error, setError] = useState(null);
  const [detail, setDetail] = useState(null);

  useEffect(() => {
    getCatalog().then(setElephants).catch((err) => setError(err.message));
  }, []);

  const openDetail = async (name) => {
    try {
      setDetail(await getElephant(name));
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div>
      <h1 className="screen-title">Known-elephant catalog</h1>
      <p className="screen-sub">
        The reference set of known elephants and their approved ear evidence.
        New known elephants appear here after an identity decision is confirmed.
      </p>

      {error && <div className="error-note">{error}</div>}
      {!elephants && !error && <div className="mono-dim">loading catalog…</div>}

      {elephants && (
        <div className="catalog-grid" data-testid="catalog-grid">
          {elephants.map((elephant) => (
            <button
              key={elephant.name}
              type="button"
              className="elephant-card"
              onClick={() => openDetail(elephant.name)}
            >
              {elephant.thumbnail ? (
                <img src={imageUrl(elephant.thumbnail)} alt={elephant.name} loading="lazy" />
              ) : (
                <div className="noimg">NO CROP</div>
              )}
              <div className="card-body">
                <div className="name">{elephant.name}</div>
                <div className="stats">
                  {elephant.photo_count} EARS · L{elephant.side_counts.left} R
                  {elephant.side_counts.right} · {elephant.sighting_dates.length} SIGHTINGS
                </div>
              </div>
            </button>
          ))}
        </div>
      )}

      {detail && (
        <div className="detail-overlay" onClick={() => setDetail(null)}>
          <div className="detail-card" onClick={(event) => event.stopPropagation()}>
            <div className="panel-title">KNOWN-ELEPHANT RECORD</div>
            <h2 className="screen-title">{detail.name}</h2>
            <p className="screen-sub">
              {detail.photos.length} ear record{detail.photos.length === 1 ? '' : 's'} on
              file — each crop shows the ear region used for matching
            </p>
            <div className="photo-grid">
              {detail.photos.map((photo) => (
                <div className="photo-cell" key={`${photo.photo_id}-${photo.side}`}>
                  {photo.crop_path ? (
                    <ZoomImage
                      src={imageUrl(photo.crop_path)}
                      alt={photo.photo_id}
                      caption={`${detail.name} — ${photo.photo_id} (${photo.side} ear, ${photo.date})`}
                    />
                  ) : (
                    <div className="noimg">NO CROP</div>
                  )}
                  <div className="photo-meta">
                    <span className="fname">{photo.date}</span>
                    <span className={`badge ${photo.side}`}>{photo.side[0]}</span>
                  </div>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 16 }}>
              <button type="button" className="btn ghost" onClick={() => setDetail(null)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
