// Boot the MoGe demo page in a real WebKit and report every status
// transition, the env label, and the console — repro harness for mobile
// WebKit boot failures.
//
// Needs playwright + its WebKit build (not a project dependency):
//   npm i -D playwright && npx playwright install webkit
// Usage:
//   node tools/webkit-boot-check.mjs "<page url with ?img=...>" [--break-gpu | --break-pipeline]
//   --break-gpu       adapter request rejects → LiteRT.js itself detects it and
//                     the page runs on WASM (verified: full pass)
//   --break-pipeline  detection passes, pipeline creation throws — NOTE: a sync
//                     throw here wedges loadLiteRt (hangs, no rejection)
//
// For iOS-WebKit specifics, the iOS Simulator runs the real engine:
//   xcrun simctl boot <udid> && xcrun simctl openurl booted "<url>"
//   xcrun simctl io <udid> screenshot shot.png
// (it does not model real-device memory ceilings or GPU limits).
import { webkit } from 'playwright';

const url = process.argv[2];
const breakGpu = process.argv[3] === '--break-gpu';
const breakPipeline = process.argv[3] === '--break-pipeline';
const browser = await webkit.launch();
const page = await browser.newPage();
if (breakGpu) {
  // navigator.gpu stays present (so the demo still *chooses* webgpu) but the
  // adapter request fails — models the iPhone's "exists on paper" case.
  // Patch the prototype: navigator.gpu may hand out a fresh instance per
  // access, so an own-property shadow on one instance never connects.
  await page.addInitScript(() => {
    const proto = Object.getPrototypeOf(navigator.gpu);
    Object.defineProperty(proto, 'requestAdapter', {
      value: () => Promise.reject(new Error('simulated adapter failure')),
      configurable: true,
    });
  });
}
const logs = [];
page.on('console', (m) => logs.push(`[console.${m.type()}] ${m.text()}`));
page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));

if (breakPipeline) {
  // Detection passes (adapter + device fine) but real use fails — the
  // "exists on paper" failure mode: every compute pipeline creation dies.
  await page.addInitScript(() => {
    Object.defineProperty(GPUDevice.prototype, 'createComputePipeline', {
      value: function () { throw new Error('simulated pipeline failure'); },
      configurable: true,
    });
    Object.defineProperty(GPUDevice.prototype, 'createComputePipelineAsync', {
      value: function () { return Promise.reject(new Error('simulated pipeline failure')); },
      configurable: true,
    });
  });
}

await page.goto(url, { waitUntil: 'domcontentloaded' });
console.log(`[gpu] navigator.gpu present: ${await page.evaluate(() => !!navigator.gpu)}`);
if (breakGpu) {
  console.log(`[sabotage] ${await page.evaluate(() =>
    navigator.gpu.requestAdapter().then((a) => `NOT CONNECTED (resolved: ${!!a})`, (e) => `connected (rejects: ${e.message})`))}`);
}
console.log(`[coi] crossOriginIsolated: ${await page.evaluate(() => window.crossOriginIsolated)}`);

const deadline = Date.now() + 300000;
let last = '';
let final = '';
while (Date.now() < deadline) {
  const s = await page.textContent('#status').catch(() => null);
  if (s && s.trim() !== last) {
    last = s.trim();
    console.log(`[status] ${last}`);
  }
  if (/Failed|Done\./.test(last)) {
    final = last;
    break;
  }
  await page.waitForTimeout(500);
}
if (!final) console.log('[timeout] no terminal status within 300 s');
console.log(`[env] ${(await page.textContent('#env').catch(() => '')).trim()}`);
console.log(`[latency] ${(await page.textContent('#latency').catch(() => '')).trim()}`);
for (const l of logs.slice(-40)) console.log(l);
await browser.close();
