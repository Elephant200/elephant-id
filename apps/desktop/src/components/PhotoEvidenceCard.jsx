import { imageUrl } from '../api.js';
import { useLightbox, ZoomImage } from './Lightbox.jsx';
import ProfileChart from './ProfileChart.jsx';

export const SIDE_COLORS = { left: '#3d6379', right: '#8a5a2c' };

const SIDE_ORDER = { left: 0, right: 1 };

/** Full review evidence for one analyzed photo (docs/pipeline.md §4.2). */
export default function PhotoEvidenceCard({ photo }) {
  const skipped = photo.status === 'skipped';
  const mainImage = photo.overlay_path || photo.photo_path;
  return (
    <div className={`photo-card ${skipped ? 'skipped' : ''}`} data-testid="photo-card">
      <div className="photo-card-head">
        <span className="fname">{photo.file_name}</span>
        <span className={`badge ${skipped ? 'warn' : 'ok'}`}>{photo.status}</span>
        {photo.view && <span className="badge">view {photo.view}</span>}
        {[...(photo.tusks || [])]
          .sort((a, b) => (SIDE_ORDER[a.side] ?? 2) - (SIDE_ORDER[b.side] ?? 2))
          .map((tusk, index) => (
            <span key={`tusk-${index}`} className="badge">
              {tusk.side} tusk · {Math.round(tusk.confidence * 100)}%
            </span>
          ))}
        <span className="head-spacer" />
        <span className="mono-dim">{photo.detail}</span>
      </div>

      <div className="photo-card-body">
        {mainImage ? (
          <div className="overlay-block">
            <ZoomImage
              src={imageUrl(mainImage)}
              fullSrc={imageUrl(mainImage)}
              alt={photo.file_name}
              caption={`${photo.file_name} — model detections`}
              className="overlay-thumb"
            />
            <div className="img-caption">
              {photo.overlay_path ? 'MODEL DETECTIONS' : 'ORIGINAL PHOTO'}
              {photo.overlay_path && photo.photo_path && (
                <ZoomLink path={photo.photo_path} label="VIEW ORIGINAL" caption={photo.file_name} />
              )}
            </div>
          </div>
        ) : (
          <div className="overlay-block">
            <div className="noimg overlay-thumb">NO IMAGE</div>
          </div>
        )}

        <div className="ear-evidence">
          {photo.ears.length === 0 && !skipped && (
            <div className="mono-dim">NO USABLE EARS</div>
          )}
          {photo.ears.map((ear, index) => (
            <div className="ear-block" key={`${ear.side}-${index}`}>
              <figure className="ear-crop-figure">
                {ear.crop_path ? (
                  <ZoomImage
                    src={imageUrl(ear.crop_path)}
                    alt={`${ear.side} ear`}
                    caption={`${photo.photo_id} — ${ear.side} ear (anchored contour)`}
                    className="ear-crop"
                  />
                ) : (
                  <div className="noimg ear-crop">NO CROP</div>
                )}
                <figcaption className="img-caption">ANCHORED EAR CROP</figcaption>
              </figure>
              <div className="ear-chart">
                <ProfileChart
                  series={[
                    {
                      label: ear.side,
                      values: ear.profile,
                      color: SIDE_COLORS[ear.side] || '#7c7f7e',
                      fill: true,
                    },
                  ]}
                />
                <div className="ear-chart-caption">
                  <span className={`badge ${ear.side}`}>{ear.side} ear</span>
                  <span className="mono-dim" title="tear-depth profile along the outer ear margin">
                    TEAR PROFILE · 0–180° · TOTAL DEPTH{' '}
                    {ear.mass ? ear.mass.toFixed(1) : '0.0'}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function ZoomLink({ path, label, caption }) {
  const open = useLightbox();
  return (
    <button
      type="button"
      className="link-btn"
      onClick={() => open(imageUrl(path), caption)}
    >
      {label}
    </button>
  );
}
