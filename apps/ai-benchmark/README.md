# AI Benchmark

Static browser benchmark for local AI and computer-vision suitability.

## Deploy Shape

- App shell: Cloudflare Pages, build command `npm run build`, output `dist`.
- Heavy assets: Cloudflare R2 public bucket, referenced by a manifest URL.
- R2 assets need CORS for `GET` and `HEAD`, plus cross-origin resource policy compatible with the Pages `_headers` file.

## Local Use

```bash
npm install
npm run dev
```

The bundled `public/benchmark-assets.example.json` uses public source URLs and placeholder model hashes. Replace it with an R2-hosted manifest before production use.
