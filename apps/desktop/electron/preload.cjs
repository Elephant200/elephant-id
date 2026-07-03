// Exposes the minimal desktop surface to the renderer.
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('alphaphant', {
  apiBase: `http://127.0.0.1:${Number(process.env.ALPHAPHANT_API_PORT || 8756)}`,
  selectFolder: () => ipcRenderer.invoke('alphaphant:select-folder'),
});
