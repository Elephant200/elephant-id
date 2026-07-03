import { imageUrl } from '../api.js';
import { ZoomImage } from '../components/Lightbox.jsx';
import PhotoEvidenceCard from '../components/PhotoEvidenceCard.jsx';

export default function ReviewPage({ sighting, setRoute }) {
  if (!sighting || sighting.status !== 'ready') {
    return (
      <div>
        <h1 className="screen-title">Review evidence</h1>
        <div className="empty-note">ANALYZE A SIGHTING FIRST</div>
      </div>
    );
  }

  const bySide = { left: [], right: [] };
  sighting.photos.forEach((photo) => {
    photo.ears.forEach((ear) => {
      if (ear.crop_path && bySide[ear.side]) {
        bySide[ear.side].push({ photo_id: photo.photo_id, crop_path: ear.crop_path });
      }
    });
  });

  return (
    <div>
      <h1 className="screen-title">Review evidence</h1>
      <p className="screen-sub">
        Check what the models saw before matching. Each photo shows its
        detections (body, ears, tusks, anchors), the anchored ear crops, and
        the tear-depth embedding along the outer ear margin — the signal the
        matcher compares. Click any image to view it full size.
      </p>

      <div className="panel">
        <div className="panel-title">
          BEST EAR EVIDENCE
          <span className="badge ok">{sighting.profile_count} PROFILES</span>
        </div>
        <div className="side-columns">
          {['left', 'right'].map((side) => (
            <div key={side} className="side-column">
              <div className={`badge ${side}`} style={{ marginBottom: 8 }}>
                {side} ear · {bySide[side].length} photo
                {bySide[side].length === 1 ? '' : 's'}
              </div>
              {bySide[side].length === 0 ? (
                <div className="empty-note">NO {side.toUpperCase()} EAR FOUND</div>
              ) : (
                <div className="repr-row">
                  {bySide[side].map((ear, index) => (
                    <figure className="repr-ear" key={`${ear.photo_id}-${index}`}>
                      <ZoomImage
                        src={imageUrl(ear.crop_path)}
                        alt={`${side} ear ${ear.photo_id}`}
                        caption={`${ear.photo_id} — ${side} ear`}
                      />
                      <figcaption>{ear.photo_id}</figcaption>
                    </figure>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="panel-title" style={{ marginBottom: 12 }}>
        PHOTO DRILL-DOWN · {sighting.photos.length} PHOTOS
      </div>
      {sighting.photos.map((photo) => (
        <PhotoEvidenceCard key={photo.file_name} photo={photo} />
      ))}

      <div className="page-actions">
        <button
          type="button"
          className="btn primary"
          data-testid="to-match"
          disabled={sighting.profile_count === 0}
          onClick={() => setRoute('match')}
        >
          Continue to matching
        </button>
      </div>
    </div>
  );
}
