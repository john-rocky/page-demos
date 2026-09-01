// Deploy-shape / live e2e checks for the demos. Needs playwright
// (not a project dependency): npm i --no-save playwright
// Deploy-shaped check for the MoGe page with the site-root coi-serviceworker:
// no COOP/COEP from the server, SW-injected isolation, threaded wasm fallback
// (headless chromium has no GPU adapter → webgpu → wasm fallback path).
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { join, normalize, extname } from 'node:path';
const pw = await import('playwright');

const DIST = new URL('../dist', import.meta.url).pathname;
const PREFIX = '/page-demos';
const PORT = 8932;
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
  '.jpg': 'image/jpeg',
};

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost');
    let rel = normalize(url.pathname.slice(PREFIX.length + 1));
    if (rel.includes('..')) { res.statusCode = 400; return res.end(); }
    let file = join(DIST, rel);
    try {
      if ((await stat(file)).isDirectory()) file = join(file, 'index.html');
    } catch { /* readFile below 404s */ }
    const body = await readFile(file);
    res.setHeader('Content-Type', MIME[extname(file)] ?? 'application/octet-stream');
    res.setHeader('Content-Length', body.length);
    res.end(body);
  } catch {
    res.statusCode = 404;
    res.end('not found');
  }
});
await new Promise((r) => server.listen(PORT, r));

const browser = await pw[process.argv[2] === '--webkit' ? 'webkit' : 'chromium'].launch();
const page = await (await browser.newContext()).newPage();
const consoleLines = [];
page.on('console', (m) => consoleLines.push(m.text()));
page.on('pageerror', (e) => consoleLines.push('PAGEERROR ' + e.message));
let navs = 0;
page.on('framenavigated', (f) => { if (f === page.mainFrame()) navs++; });

const url = `http://localhost:${PORT}${PREFIX}/moge/?img=${encodeURIComponent(`${PREFIX}/test-photo.jpg`)}`;
console.log('open', url);
await page.goto(url);

const deadline = Date.now() + 420_000;
let lastStatus = '';
let done = false;
while (Date.now() < deadline && !done) {
  await new Promise((r) => setTimeout(r, 3000));
  const s = await page.locator('#status').textContent().catch(() => '(gone)');
  if (s !== lastStatus) { lastStatus = s; console.log('status:', s); }
  if (s.startsWith('Failed')) break;
  done = await page.locator('#latency').isVisible().catch(() => false);
}

const iso = await page.evaluate(() => ({
  crossOriginIsolated: window.crossOriginIsolated,
  swController: !!navigator.serviceWorker?.controller,
}));
console.log('navigations (2 = coi reload):', navs);
console.log('isolation:', JSON.stringify(iso));
console.log('backend label:', await page.locator('#backend').textContent().catch(() => '(none)'));
console.log('env line:', await page.locator('#env').textContent().catch(() => '(none)'));
console.log('latency:', (await page.locator('#latency').innerText().catch(() => '(none)')).replace(/\n/g, ' | '));
if (!done) for (const l of consoleLines.slice(-8)) console.log('  |', l);
await browser.close();
server.close();
const label = await Promise.resolve();
process.exit(done && iso.crossOriginIsolated ? 0 : 1);
