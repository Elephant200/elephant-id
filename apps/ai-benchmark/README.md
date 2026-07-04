# AI Benchmark

Static browser benchmark for local AI and computer-vision suitability.

Live at [benchmark.elephant-id.org](https://benchmark.elephant-id.org).

## Offline-first

The app is meant to test whether a device can run local AI in the field, so it
holds to one rule: **nothing large downloads unless the user asks for it.**

- Page load and the general-hardware check run **locally** — only the tiny app
  shell and manifest touch the network.
- The ONNX Runtime engine (~13–24MB) and model weights (9–216MB) download **only
  when the user selects a model and clicks Start**.
- A service worker (`public/sw.js`) caches the app shell and the runtime on first
  use, and `modelRunner.js` caches model weights, so re-runs work with no internet.

## Deploy Shape

- **App shell**: Cloudflare Pages (`ai-benchmark` project), build `npm run build`,
  output `dist`. `public/_headers` sets `COOP: same-origin` / `COEP: require-corp`
  so the wasm backend can use threads.
- **Runtime + weights**: Cloudflare R2 bucket `ai-benchmark-assets`, custom domain
  `weights.benchmark.elephant-id.org`, referenced by `public/benchmark-assets.json`.
  The runtime `wasmPaths` is an **absolute cross-origin URL** — this is deliberate:
  it loads under COEP via CORS, and (unlike a same-origin `/public` path) Vite's dev
  server doesn't try to transform it. R2 CORS is locked to `GET`/`HEAD` from
  `https://benchmark.elephant-id.org` only.

### Runtime files on R2

The wasm/glue files under `runtime/` on R2 must match the pinned `onnxruntime-web`
version. The wasm backend uses the threaded build, WebGPU uses the asyncify build:

```bash
D=node_modules/onnxruntime-web/dist
for f in ort-wasm-simd-threaded.mjs ort-wasm-simd-threaded.wasm \
         ort-wasm-simd-threaded.asyncify.mjs ort-wasm-simd-threaded.asyncify.wasm; do
  ct=$([[ $f == *.wasm ]] && echo application/wasm || echo text/javascript)
  npx wrangler r2 object put "ai-benchmark-assets/runtime/$f" --file "$D/$f" \
    --content-type "$ct" --remote   # --remote is required, or it writes to the local sim
done
```

## Local Use

```bash
npm install
npm run dev       # http://localhost:5173 — Vite dev (no service worker)
npm run build && npm run preview -- --port 5173 --strictPort   # test the SW/offline path
npm run deploy    # build + wrangler pages deploy dist --project-name ai-benchmark
```

R2 CORS is production-only (`https://benchmark.elephant-id.org`). To run models
locally, temporarily add `http://localhost:5173` to the bucket's CORS origins
(`wrangler r2 bucket cors set ai-benchmark-assets --file ...`) and remove it after.
Without that, local model runs fail on CORS, but the shell and hardware check still work.
