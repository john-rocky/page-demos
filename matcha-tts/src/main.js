/**
 * Matcha-TTS in the browser, fully client-side.
 *
 * Pipeline (per chunk ≤127 phonemes):
 *   text → G2P (dict + DeepPhonemizer on WASM) → symbol ids
 *   → text encoder (WebGPU) → durations → length-regulate
 *   → flow-matching decoder ×10 Euler steps (WASM — GPU mis-fuses this graph)
 *   → HiFi-GAN vocoder (WebGPU) → 22.05 kHz waveform.
 *
 * Model + glue files come from litert-community/Matcha-TTS on Hugging Face
 * and are cached with the Cache API after first download (~92 MB).
 *
 * Debug URL params: ?text=…&seed=0&steps=10&voc=wasm|webgpu&enc=wasm|webgpu
 *   &nosound=1&save=<name> (dev server writes out/<name>.wav + <name>.json)
 *   &ab=1 (re-vocode the same mels on the other vocoder backend, report the
 *   numeric diff, save out/<name>_ab.wav — isolates GPU-vocoder correctness)
 */
import { isWebGPUSupported, loadAndCompile, loadLiteRt } from '@litertjs/core';
import { G2P, phonemize } from './g2p.js';
import { Synthesizer, encodeWav } from './synth.js';
import { Viz } from './viz.js';

const HF = 'https://huggingface.co/litert-community/Matcha-TTS/resolve/main/';
// Resolve the wasm dir against this module's own URL, which works in dev
// (/src/main.js) and in the build (/assets/*.js) alike — a page-relative dir
// breaks as soon as the page URL shape changes.
const WASM_DIR = new URL('../litert-wasm/', import.meta.url).href;
// 4 Euler steps: ear-approved quality at ~1/3 the latency of the full 10
// (decoder is 341 ms/call on WASM — see the model card). Override: ?steps=
const DEFAULT_STEPS = 4;
const FILES = {
  textenc: 'matcha_textenc_fp16.tflite',
  decoder: 'matcha_decoder_fp16.tflite',
  vocoder: 'matcha_vocoder_fp16.tflite',
  g2p: 'dp_g2p_matcha_fp16.tflite',
  emb: 'emb.bin',
  dict: 'g2p_dict.txt.gz',
  config: 'config.json',
  g2pMeta: 'g2p_meta.json',
};
const CACHE_NAME = 'matcha-tts-demo-v1';

const params = new URLSearchParams(location.search);
const statusEl = document.getElementById('status');
const latencyEl = document.getElementById('latency');
const ipaEl = document.getElementById('ipa');
const textEl = document.getElementById('text');
const speakBtn = document.getElementById('speak');
const downloadBtn = document.getElementById('download');

let g2p = null;
let synth = null;
let vocoderBytes = null; // kept for ?ab=1 re-compilation
let viz = null;
let symToId = null;
let cfg = null;
let backends = null;
let wasmOpts = null; // which loadLiteRt rung succeeded (threads or plain)
let audioCtx = null;
let liveSources = [];
let lastWav = null;
let busy = false;

function status(text, pct = null) {
  statusEl.textContent = text;
  if (pct !== null) {
    const bar = document.createElement('span');
    bar.className = 'bar';
    const fill = document.createElement('i');
    fill.style.width = `${Math.round(pct * 100)}%`;
    bar.appendChild(fill);
    statusEl.appendChild(bar);
  }
}

// --- asset loading ---------------------------------------------------------

async function fetchCached(name, onProgress) {
  const url = HF + name;
  const cache = 'caches' in window ? await caches.open(CACHE_NAME) : null;
  if (cache) {
    const hit = await cache.match(url);
    if (hit) {
      const bytes = new Uint8Array(await hit.arrayBuffer());
      onProgress?.(bytes.length, bytes.length);
      return bytes;
    }
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
  const total = Number(response.headers.get('Content-Length')) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress?.(received, total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  if (cache) await cache.put(url, new Response(bytes.slice().buffer));
  return bytes;
}

async function gunzip(bytes) {
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes; // already plain
  const ds = new DecompressionStream('gzip');
  const stream = new Blob([bytes]).stream().pipeThrough(ds);
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function parseDict(bytes) {
  const text = new TextDecoder().decode(bytes);
  const dict = new Map();
  let start = 0;
  while (start < text.length) {
    let end = text.indexOf('\n', start);
    if (end === -1) end = text.length;
    const tab = text.indexOf('\t', start);
    if (tab > start && tab < end) {
      dict.set(text.slice(start, tab), text.slice(tab + 1, end));
    }
    start = end + 1;
  }
  return dict;
}

async function boot() {
  try {
    status('Loading runtime…');
    // `threads` and `jspi` are mutually exclusive in LiteRT.js, and threads
    // only work on a cross-origin-isolated page — ask for what can succeed,
    // then fall back to plain. The threaded build is what makes the CPU
    // decoder usable; single-thread is ~5-50x slower. ?threads=0 forces the
    // single-thread build for comparison.
    const wantThreads = params.get('threads') !== '0' && window.crossOriginIsolated;
    const rungs = wantThreads
      ? [{ threads: true }, { threads: false }]
      : [{ threads: false }];
    for (const [index, opts] of rungs.entries()) {
      try {
        await loadLiteRt(WASM_DIR, opts);
        wasmOpts = opts;
        break;
      } catch (err) {
        if (index === rungs.length - 1) throw err;
      }
    }
    const gpu = isWebGPUSupported() ? 'webgpu' : 'wasm';
    backends = {
      g2p: 'wasm',
      textenc: params.get('enc') === 'wasm' ? 'wasm' : gpu,
      decoder: 'wasm',
      vocoder: params.get('voc') === 'wasm' ? 'wasm' : gpu,
    };

    // ~92 MB total on first visit; Cache API afterwards
    const progress = {};
    const report = () => {
      let done = 0;
      for (const k in progress) done += progress[k][0];
      status(`Downloading models (one-time)… ${(done / 1048576).toFixed(0)} MB`, Math.min(done / 92e6, 1));
    };
    const grab = (name) =>
      fetchCached(FILES[name], (rec, tot) => {
        progress[name] = [rec, tot];
        report();
      });
    const [teB, deB, voB, g2pB, embB, dictB, cfgB, metaB] = await Promise.all([
      grab('textenc'), grab('decoder'), grab('vocoder'), grab('g2p'),
      grab('emb'), grab('dict'), grab('config'), grab('g2pMeta'),
    ]);

    cfg = JSON.parse(new TextDecoder().decode(cfgB));
    const meta = JSON.parse(new TextDecoder().decode(metaB));
    symToId = new Map();
    cfg.symbols.forEach((s, i) => {
      if (s.length === 1) symToId.set(s, i);
    });
    const dict = parseDict(await gunzip(dictB));
    const emb = new Float32Array(embB.buffer, embB.byteOffset, embB.byteLength / 4);

    status(`Compiling (${backends.textenc}/${backends.vocoder})…`);
    const numThreads = Math.min(8, navigator.hardwareConcurrency || 4);
    const wasmCompile = { accelerator: 'wasm', cpuOptions: { numThreads } };
    const models = {};
    for (const [key, bytes] of [['textenc', teB], ['decoder', deB], ['vocoder', voB], ['g2p', g2pB]]) {
      if (backends[key] === 'wasm') {
        models[key] = await loadAndCompile(bytes, wasmCompile);
        continue;
      }
      try {
        models[key] = await loadAndCompile(bytes, { accelerator: backends[key] });
      } catch {
        // WebGPU exists on paper in more browsers than it works in (mobile
        // WebKit in particular) — fall back to WASM instead of dying. Only
        // models not compiled yet are moved; an already-compiled one keeps
        // its backend (and its label stays truthful).
        for (const k of Object.keys(backends)) {
          if (backends[k] === 'webgpu' && !models[k]) backends[k] = 'wasm';
        }
        status(`WebGPU failed — compiling on WASM…`);
        models[key] = await loadAndCompile(bytes, wasmCompile);
      }
    }

    if (params.get('ab') === '1') vocoderBytes = voB;
    g2p = new G2P(dict, meta, models.g2p);
    synth = new Synthesizer(models, emb, cfg);
    viz = new Viz(document.getElementById('viz'), {
      nFeats: cfg.n_feats,
      melMean: cfg.mel_mean,
      melStd: cfg.mel_std,
      sampleRate: cfg.sample_rate,
    });

    // console debugging hook (dev only; harmless in prod)
    window.__matcha = { synth, g2p, cfg, models, backends, loadAndCompile, decoderBytes: deB };

    speakBtn.disabled = false;
    status('Ready — type something and press Speak.');

    if (params.get('text')) textEl.value = params.get('text');
    if (params.get('frames') === '1') {
      // dev-only frames-dump showcase → frames/ via the /__save endpoint
      const { runFramesShowcase } = await import('./showcase.js');
      status('Rendering showcase frames…');
      const n = await runFramesShowcase({ synth, g2p, symToId, cfg }, textEl.value, {
        seed: Number(params.get('seed') ?? 0),
        steps: Number(params.get('steps') ?? DEFAULT_STEPS),
      });
      status(`Showcase frame dump complete (${n} frames).`);
    } else if (params.get('text') || params.get('demo') === '1') {
      // ?demo=1: auto-speak the default text (clean URL for recordings)
      await speak();
    }
  } catch (err) {
    status(`Failed to start: ${err instanceof Error ? err.message : err}`);
    throw err;
  }
}

// --- synthesis + playback --------------------------------------------------

function stopPlayback() {
  for (const src of liveSources) {
    try { src.stop(); } catch { /* already ended */ }
  }
  liveSources = [];
}

function fmtLatency(t, audioSeconds, chunkCount, steps) {
  const total = t.g2p + t.textenc + t.decoder + t.vocoder;
  const rtf = total / 1000 / audioSeconds;
  // Show which wasm actually loaded — a page that silently lost threads
  // (and with them the advertised RTF) must be visible at a glance.
  const wasmLabel = wasmOpts?.threads ? 'wasm' : 'wasm·1-thread';
  const env = (b) => `<span class="env">(${b === 'wasm' ? wasmLabel : b})</span>`;
  return (
    `g2p <b>${t.g2p.toFixed(0)} ms</b> ${env(backends.g2p)} · ` +
    `textenc <b>${t.textenc.toFixed(0)} ms</b> ${env(backends.textenc)} · ` +
    `decoder×${steps} <b>${t.decoder.toFixed(0)} ms</b> ${env(backends.decoder)} · ` +
    `vocoder <b>${t.vocoder.toFixed(0)} ms</b> ${env(backends.vocoder)}<br>` +
    `${chunkCount > 1 ? `${chunkCount} chunks · ` : ''}` +
    `total <b>${total.toFixed(0)} ms</b> for ${audioSeconds.toFixed(1)} s of audio · RTF <b>${rtf.toFixed(2)}</b>`
  );
}

async function speak() {
  if (busy || !synth) return;
  const text = textEl.value.trim();
  if (!text) return;
  busy = true;
  speakBtn.disabled = true;
  const nosound = params.get('nosound') === '1';
  const seed = Number(params.get('seed') ?? 0);
  const steps = Number(params.get('steps') ?? DEFAULT_STEPS);
  try {
    stopPlayback();
    viz.reset();
    status('Phonemizing…');
    const tG2p0 = performance.now();
    const chunks = await phonemize(g2p, symToId, text);
    const tG2p = performance.now() - tG2p0;
    if (!chunks.length) {
      status('Nothing pronounceable in that text.');
      return;
    }
    ipaEl.style.display = 'block';
    ipaEl.textContent = 'IPA: ' + chunks.map((c) => c.ipa).join(' ‖ ');
    ipaEl.title = ipaEl.textContent;

    if (!nosound) {
      audioCtx ??= new AudioContext({ sampleRate: cfg.sample_rate });
      if (audioCtx.state === 'suspended') await audioCtx.resume();
    }
    const timings = { g2p: tG2p, textenc: 0, decoder: 0, vocoder: 0 };
    const results = [];
    let playCursor = null;
    let audioSeconds = 0;
    for (let i = 0; i < chunks.length; i++) {
      status(`Synthesizing… ${i + 1}/${chunks.length}`);
      const r = await synth.run(chunks[i].ids, { steps, seed });
      results.push(r);
      for (const k of ['textenc', 'decoder', 'vocoder']) timings[k] += r.timings[k];
      viz.append(r.mel, r.ylen, cfg.MAX_MEL, r.wav);
      audioSeconds += r.wav.length / cfg.sample_rate;
      if (!nosound) {
        const buf = audioCtx.createBuffer(1, r.wav.length, cfg.sample_rate);
        buf.copyToChannel(r.wav, 0);
        const src = audioCtx.createBufferSource();
        src.buffer = buf;
        src.connect(audioCtx.destination);
        const at = Math.max(playCursor ?? 0, audioCtx.currentTime + 0.05);
        if (playCursor === null) viz.playFrom(audioCtx, at, 0);
        src.start(at);
        liveSources.push(src);
        playCursor = at + buf.duration;
      }
      latencyEl.style.display = 'block';
      latencyEl.innerHTML = fmtLatency(timings, audioSeconds, chunks.length, steps);
    }
    if (!nosound && playCursor !== null) {
      viz.playDur = audioSeconds; // cursor spans the full utterance
    }
    status('Done.');

    // assemble the full waveform for download / save
    const total = results.reduce((n, r) => n + r.wav.length, 0);
    const full = new Float32Array(total);
    let off = 0;
    for (const r of results) {
      full.set(r.wav, off);
      off += r.wav.length;
    }
    lastWav = full;
    downloadBtn.style.display = 'inline-block';

    // machine-readable stats for automation
    let peak = 0;
    let sumSq = 0;
    let nonFinite = 0;
    for (let i = 0; i < full.length; i++) {
      const v = full[i];
      if (!Number.isFinite(v)) { nonFinite++; continue; }
      peak = Math.max(peak, Math.abs(v));
      sumSq += v * v;
    }
    const stats = {
      text,
      backends,
      wasmThreads: !!wasmOpts?.threads,
      seed,
      steps,
      chunks: chunks.map((c, i) => ({ ipa: c.ipa, pids: c.ids.length, ylen: results[i].ylen })),
      samples: full.length,
      seconds: +(full.length / cfg.sample_rate).toFixed(3),
      rms: +Math.sqrt(sumSq / full.length).toFixed(5),
      peak: +peak.toFixed(5),
      nonFinite,
      timings: Object.fromEntries(Object.entries(timings).map(([k, v]) => [k, +v.toFixed(1)])),
    };
    // A/B: same mels through the other vocoder backend → numeric diff
    if (vocoderBytes) {
      const altBackend = backends.vocoder === 'webgpu' ? 'wasm' : 'webgpu';
      status(`A/B: vocoding on ${altBackend}…`);
      const altModel = await loadAndCompile(vocoderBytes, { accelerator: altBackend });
      const altFull = new Float32Array(total);
      let aOff = 0;
      const t0 = performance.now();
      for (const r of results) {
        altFull.set(await synth.vocode(r.mel, r.ylen, altModel), aOff);
        aOff += r.wav.length;
      }
      const abMs = performance.now() - t0;
      let maxDiff = 0;
      let dotAB = 0;
      let dotA = 0;
      let dotB = 0;
      for (let i = 0; i < total; i++) {
        const d = Math.abs(full[i] - altFull[i]);
        if (d > maxDiff) maxDiff = d;
        dotAB += full[i] * altFull[i];
        dotA += full[i] * full[i];
        dotB += altFull[i] * altFull[i];
      }
      stats.ab = {
        backend: altBackend,
        ms: +abMs.toFixed(1),
        maxAbsDiff: +maxDiff.toFixed(6),
        corr: +(dotAB / Math.sqrt(dotA * dotB)).toFixed(6),
      };
      if (params.get('save')) {
        await fetch(`/__save?name=out/${params.get('save')}_ab.wav`, {
          method: 'POST', body: encodeWav(altFull, cfg.sample_rate),
        });
      }
    }
    console.log('MATCHA_STATS ' + JSON.stringify(stats));
    const save = params.get('save');
    if (save) {
      await fetch(`/__save?name=out/${save}.wav`, { method: 'POST', body: encodeWav(full, cfg.sample_rate) });
      await fetch(`/__save?name=out/${save}.json`, { method: 'POST', body: JSON.stringify(stats, null, 2) });
      if (params.get('dumpmel') === '1') {
        // raw mel tensors for offline vocoder cross-checks (80×MAX_MEL f32 each)
        for (let i = 0; i < results.length; i++) {
          await fetch(`/__save?name=out/${save}_mel_${i}.bin`, { method: 'POST', body: results[i].mel });
        }
      }
      status(`Done — saved out/${save}.wav`);
    }
  } catch (err) {
    status(`Failed: ${err instanceof Error ? err.message : err}`);
    throw err;
  } finally {
    busy = false;
    speakBtn.disabled = false;
  }
}

speakBtn.addEventListener('click', speak);
textEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
    e.preventDefault();
    speak();
  }
});
downloadBtn.addEventListener('click', () => {
  if (!lastWav) return;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(encodeWav(lastWav, cfg.sample_rate));
  a.download = 'matcha-tts.wav';
  a.click();
  URL.revokeObjectURL(a.href);
});

boot();
