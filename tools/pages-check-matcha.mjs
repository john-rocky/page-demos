// Deploy-shape / live e2e checks for the demos. Needs playwright
// (not a project dependency): npm i --no-save playwright
// Deploy-shaped check for the Matcha page: serve dist/ under /page-demos/
// with NO COOP/COEP headers (= GitHub Pages), then verify coi-serviceworker
// turns on cross-origin isolation, the threaded wasm rung loads, the HF
// model fetches pass COEP require-corp, and a full synthesis completes.
// Usage: node matcha-pages-check.mjs [--no-sw-expect]
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { join, normalize, extname } from 'node:path';
const pw = await import('playwright');
const engine = pw[process.argv[2] === '--webkit' ? 'webkit' : 'chromium'];

const DIST = new URL('../dist', import.meta.url).pathname;
const PREFIX = '/page-demos';
const PORT = 8931;
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
  '.css': 'text/css',
  '.png': 'image/png',
  '.gif': 'image/gif',
};

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost');
    if (!url.pathname.startsWith(PREFIX + '/')) {
      res.statusCode = 404;
      return res.end('outside prefix');
    }
    let rel = normalize(url.pathname.slice(PREFIX.length + 1));
    if (rel.includes('..')) { res.statusCode = 400; return res.end(); }
    let file = join(DIST, rel);
    try {
      if ((await stat(file)).isDirectory()) file = join(file, 'index.html');
    } catch { /* fall through to readFile error */ }
    const body = await readFile(file);
    // deliberately NO COOP/COEP — that is GitHub Pages
    res.setHeader('Content-Type', MIME[extname(file)] ?? 'application/octet-stream');
    res.setHeader('Content-Length', body.length);
    res.end(body);
  } catch {
    res.statusCode = 404;
    res.end('not found');
  }
});
await new Promise((r) => server.listen(PORT, r));

const browser = await engine.launch();
const context = await browser.newContext();
const page = await context.newPage();

let stats = null;
const consoleLines = [];
page.on('console', (msg) => {
  const t = msg.text();
  consoleLines.push(t);
  if (t.startsWith('MATCHA_STATS ')) stats = JSON.parse(t.slice('MATCHA_STATS '.length));
});
page.on('pageerror', (e) => consoleLines.push('PAGEERROR ' + e.message));
let navs = 0;
page.on('framenavigated', (f) => { if (f === page.mainFrame()) navs++; });

const url = `http://localhost:${PORT}${PREFIX}/matcha-tts/?nosound=1&text=${encodeURIComponent(
  'The quick brown fox jumps over the lazy dog.'
)}`;
console.log('open', url);
await page.goto(url);

// wait for the SW reload + full pipeline (first visit downloads ~92 MB from HF)
const deadline = Date.now() + 420_000;
let lastStatus = '';
while (Date.now() < deadline && !stats) {
  await new Promise((r) => setTimeout(r, 3000));
  const s = await page.locator('#status').textContent().catch(() => '(gone)');
  if (s !== lastStatus) {
    lastStatus = s;
    console.log(`[${Math.round((Date.now() % 1e7) / 1000)}s] status: ${s}`);
  }
  if (s.startsWith('Failed')) break;
}

const iso = await page.evaluate(() => ({
  crossOriginIsolated: window.crossOriginIsolated,
  swController: !!navigator.serviceWorker?.controller,
  sab: typeof SharedArrayBuffer !== 'undefined',
}));
console.log('navigations (1 = no reload, 2 = coi reload):', navs);
console.log('isolation:', JSON.stringify(iso));
console.log('latency line:', await page.locator('#latency').innerText().catch(() => '(none)'));
if (stats) {
  console.log('STATS', JSON.stringify({
    backends: stats.backends,
    wasmThreads: stats.wasmThreads,
    seconds: stats.seconds,
    rms: stats.rms,
    peak: stats.peak,
    nonFinite: stats.nonFinite,
    timings: stats.timings,
    rtf: +((stats.timings.g2p + stats.timings.textenc + stats.timings.decoder + stats.timings.vocoder) / 1000 / stats.seconds).toFixed(3),
  }, null, 1));
} else {
  console.log('NO STATS — console tail:');
  for (const line of consoleLines.slice(-15)) console.log('  |', line);
}
await browser.close();
server.close();
process.exit(stats && iso.crossOriginIsolated && stats.wasmThreads ? 0 : 1);
