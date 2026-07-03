// Alphaphant Electron main process: spawns the FastAPI sidecar, waits for
// /health, then opens the renderer. Fully local; no internet required.
const { app, BrowserWindow, dialog, ipcMain } = require('electron');
const { spawn } = require('node:child_process');
const http = require('node:http');
const path = require('node:path');

const API_PORT = Number(process.env.ALPHAPHANT_API_PORT || 8756);
const API_BASE = `http://127.0.0.1:${API_PORT}`;
// apps/desktop/electron -> repo root
const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');

let sidecar = null;

function startSidecar() {
  // ALPHAPHANT_EXTERNAL_API=1 lets developers run the sidecar themselves.
  if (process.env.ALPHAPHANT_EXTERNAL_API === '1') return;
  // Resolve path-valued env vars against the app dir so npm scripts can use
  // repo-relative paths; the sidecar itself runs with cwd = repo root.
  const env = { ...process.env };
  for (const key of ['ALPHAPHANT_GALLERY_PROFILES', 'ALPHAPHANT_DATA_DIR']) {
    if (env[key]) env[key] = path.resolve(path.join(__dirname, '..'), env[key]);
  }
  sidecar = spawn(
    'uv',
    ['run', 'python', '-m', 'elephant_id.api', '--port', String(API_PORT)],
    { cwd: REPO_ROOT, env, stdio: ['ignore', 'inherit', 'inherit'] },
  );
  sidecar.on('exit', (code) => {
    console.log(`[alphaphant] sidecar exited with code ${code}`);
    sidecar = null;
  });
}

function checkHealth() {
  return new Promise((resolve) => {
    const request = http.get(`${API_BASE}/health`, { timeout: 1500 }, (response) => {
      response.resume();
      resolve(response.statusCode === 200);
    });
    request.on('error', () => resolve(false));
    request.on('timeout', () => {
      request.destroy();
      resolve(false);
    });
  });
}

async function waitForSidecar(timeoutMs = 90000) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    if (await checkHealth()) return true;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  return false;
}

async function createWindow() {
  const window = new BrowserWindow({
    width: 1480,
    height: 940,
    minWidth: 1080,
    minHeight: 700,
    backgroundColor: '#ece5d2',
    title: 'Alphaphant',
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  const healthy = await waitForSidecar();
  if (!healthy) {
    dialog.showErrorBox(
      'Alphaphant sidecar did not start',
      `No response from ${API_BASE}/health. ` +
        'Check that `uv run python -m elephant_id.api` works from the repo root.',
    );
  } else {
    console.log(`[alphaphant] connected to sidecar at ${API_BASE}`);
  }

  if (process.env.VITE_DEV_SERVER_URL) {
    await window.loadURL(process.env.VITE_DEV_SERVER_URL);
  } else {
    await window.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
  }

  // Dev/CI hook: capture the window and exit, proving the app booted.
  if (process.env.ALPHAPHANT_SCREENSHOT) {
    setTimeout(async () => {
      const image = await window.webContents.capturePage();
      require('node:fs').writeFileSync(process.env.ALPHAPHANT_SCREENSHOT, image.toPNG());
      console.log(`[alphaphant] screenshot written to ${process.env.ALPHAPHANT_SCREENSHOT}`);
      app.quit();
    }, Number(process.env.ALPHAPHANT_SCREENSHOT_DELAY_MS || 6000));
  }
}

ipcMain.handle('alphaphant:select-folder', async () => {
  const result = await dialog.showOpenDialog({
    title: 'Select a sighting folder',
    properties: ['openDirectory'],
  });
  return result.canceled ? null : result.filePaths[0];
});

app.whenReady().then(() => {
  startSidecar();
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  app.quit();
});

app.on('before-quit', () => {
  if (sidecar) sidecar.kill('SIGTERM');
});
