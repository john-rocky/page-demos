import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { defineConfig } from 'vite';

// Dev-only endpoint so automation can save rendered audio straight to disk,
// bypassing the browser's download flow (which blocks programmatic,
// non-gesture downloads). POST /__save?name=foo.wav with the blob body.
// COOP/COEP make the page cross-origin isolated so LiteRT can use the
// threaded+SIMD WASM build (SharedArrayBuffer); without them the decoder
// falls back to a single-thread build and is ~50x slower. 'require-corp'
// matches what coi-serviceworker injects on the deployed page (and, unlike
// 'credentialless', isolates Safari too); the HF model fetches are CORS-mode
// against ACAO:*, which require-corp accepts.
const coi = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  base: './',
  server: { headers: coi },
  preview: { headers: coi },
  plugins: [
    {
      name: 'save-recording',
      configureServer(server) {
        server.middlewares.use('/__save', (req, res) => {
          if (req.method !== 'POST') {
            res.statusCode = 405;
            res.end('POST only');
            return;
          }
          const name = new URL(req.url, 'http://localhost').searchParams.get('name');
          // allow one optional subdir level, no '..'
          if (!name || !/^(?:[\w-]+\/)?[\w.-]+$/.test(name)) {
            res.statusCode = 400;
            res.end('bad name');
            return;
          }
          const chunks = [];
          req.on('data', (c) => chunks.push(c));
          req.on('end', async () => {
            await mkdir(dirname(name), { recursive: true }).catch(() => {});
            await writeFile(name, Buffer.concat(chunks));
            res.statusCode = 200;
            res.end('ok');
          });
        });
      },
    },
  ],
});
