import { defineConfig } from 'vite';

export default defineConfig({
  resolve: {
    conditions: ['onnxruntime-web-use-extern-wasm'],
  },
  build: {
    sourcemap: false,
    target: 'es2022',
  },
  // Both the dev server and `vite preview` need cross-origin isolation so the
  // wasm backend can use threads (matches public/_headers in production).
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
