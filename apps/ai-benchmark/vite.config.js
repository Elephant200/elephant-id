import { defineConfig } from 'vite';

export default defineConfig({
  resolve: {
    conditions: ['onnxruntime-web-use-extern-wasm'],
  },
  build: {
    sourcemap: false,
    target: 'es2022',
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
