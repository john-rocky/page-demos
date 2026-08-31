import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { defineConfig } from 'vite';

// Dev-only endpoint so the showcase can save its recorded video straight to
// disk, bypassing the browser's download flow (which blocks programmatic,
// non-gesture downloads). POST /__save?name=foo.webm with the blob body.
export default defineConfig({
  base: './',
  build: {
    rollupOptions: {
      input: {
        index: 'index.html',
        moge: 'moge/index.html',
      },
    },
  },
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
