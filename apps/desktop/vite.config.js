import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// base './' so the built bundle loads over file:// inside Electron.
export default defineConfig({
  base: './',
  plugins: [react()],
  server: { port: 5183 },
});
