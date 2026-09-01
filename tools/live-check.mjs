// Deploy-shape / live e2e checks for the demos. Needs playwright
// (not a project dependency): npm i --no-save playwright
// Live-origin check of the deployed pages (fresh profile): coi SW install,
// isolation, threaded wasm, full synthesis / inference. Headless = no GPU,
// so webgpu legs fall back — that fallback path is itself under test.
const pw = await import('playwright');
const which = process.argv[2] ?? 'matcha';
const browser = await pw[process.argv[3] === 'webkit' ? 'webkit' : 'chromium'].launch();
const page = await (await browser.newContext()).newPage();
let stats = null;
const tail = [];
page.on('console', (m) => {
  const t = m.text();
  tail.push(t);
  if (t.startsWith('MATCHA_STATS ')) stats = JSON.parse(t.slice(13));
});
let navs = 0;
page.on('framenavigated', (f) => { if (f === page.mainFrame()) navs++; });

const url = which === 'matcha'
  ? 'https://john-rocky.github.io/page-demos/matcha-tts/?nosound=1&text=Live+check+after+the+root+worker+deploy.'
  : 'https://john-rocky.github.io/page-demos/moge/?img=' +
    encodeURIComponent('https://images.pexels.com/photos/1170986/pexels-photo-1170986.jpeg?w=640');
await page.goto(url);
const deadline = Date.now() + 420_000;
let lastStatus = '';
let done = false;
while (Date.now() < deadline && !done) {
  await new Promise((r) => setTimeout(r, 4000));
  const s = await page.locator('#status').textContent().catch(() => '');
  if (s !== lastStatus) { lastStatus = s; console.log('status:', s); }
  if (s.startsWith('Failed')) break;
  done = which === 'matcha' ? !!stats : await page.locator('#latency').isVisible().catch(() => false);
}
const state = await page.evaluate(() => ({
  iso: window.crossOriginIsolated,
  controller: navigator.serviceWorker?.controller?.scriptURL ?? null,
}));
console.log('navs:', navs, 'state:', JSON.stringify(state));
if (which === 'matcha') {
  console.log('threads:', stats?.wasmThreads, 'timings:', JSON.stringify(stats?.timings));
} else {
  console.log('backend:', await page.locator('#backend').textContent().catch(() => '(none)'));
  console.log('env:', await page.locator('#env').textContent().catch(() => '(none)'));
  console.log('latency:', (await page.locator('#latency').innerText().catch(() => '')).replace(/\n/g, ' | '));
}
if (!done) for (const l of tail.slice(-6)) console.log('  |', l);
await browser.close();
process.exit(done && state.iso ? 0 : 1);
