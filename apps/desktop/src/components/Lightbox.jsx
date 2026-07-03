import { createContext, useCallback, useContext, useEffect, useState } from 'react';

const LightboxContext = createContext(() => {});

/** Wraps the app so any image can be opened full-size with a click. */
export function LightboxProvider({ children }) {
  const [item, setItem] = useState(null);

  const open = useCallback((src, caption) => setItem({ src, caption }), []);

  useEffect(() => {
    if (!item) return undefined;
    const onKey = (event) => {
      if (event.key === 'Escape') setItem(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [item]);

  return (
    <LightboxContext.Provider value={open}>
      {children}
      {item && (
        <div className="lightbox" onClick={() => setItem(null)} data-testid="lightbox">
          <img src={item.src} alt={item.caption || 'full size'} />
          <div className="lightbox-caption">
            {item.caption}
            <span className="lightbox-hint">CLICK OR PRESS ESC TO CLOSE</span>
          </div>
        </div>
      )}
    </LightboxContext.Provider>
  );
}

/** Returns open(src, caption) for showing an image full-size. */
export function useLightbox() {
  return useContext(LightboxContext);
}

/** An image that opens itself full-size when clicked. */
export function ZoomImage({ src, alt, caption, className, fullSrc }) {
  const open = useLightbox();
  if (!src) return null;
  return (
    <img
      src={src}
      alt={alt}
      loading="lazy"
      className={className}
      style={{ cursor: 'zoom-in' }}
      onClick={(event) => {
        event.stopPropagation();
        open(fullSrc || src, caption ?? alt);
      }}
    />
  );
}
