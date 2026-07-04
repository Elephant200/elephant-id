import { useRef } from 'react';

/** Shared upload target for development diagnostics. */
export default function DevUploadZone({ busy, disabled, title, action, onFile }) {
  const inputRef = useRef(null);

  const choose = (file) => {
    if (file && !busy && !disabled) onFile(file);
  };

  return (
    <div
      className="dropzone dev-upload"
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => {
        event.preventDefault();
        choose(event.dataTransfer.files?.[0]);
      }}
    >
      <h3>{title}</h3>
      <p>Drop a JPG or PNG, or choose one from disk.</p>
      <input
        ref={inputRef}
        type="file"
        accept=".jpg,.jpeg,.png"
        style={{ display: 'none' }}
        onChange={(event) => choose(event.target.files?.[0])}
      />
      <button
        type="button"
        className="btn primary"
        disabled={busy || disabled}
        onClick={() => inputRef.current?.click()}
      >
        {busy ? 'Running...' : action}
      </button>
    </div>
  );
}
