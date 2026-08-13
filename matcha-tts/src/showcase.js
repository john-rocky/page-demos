/**
 * Frames-dump showcase (dev only, ?frames=1): renders the demo story as
 * deterministic 1280×720 frames — typewriter text → "generating" beat →
 * spectrogram + waveform with a playhead sweep synced to the audio — and
 * POSTs each JPEG plus the wav and a meta.json to the dev server's /__save.
 * ffmpeg then muxes frames + audio (see meta.json for the audio offset).
 */
import { phonemize } from './g2p.js';
import { encodeWav } from './synth.js';

const W = 1280;
const H = 720;
const FPS = 30;
const MARGIN = 90;

function wrapText(ctx, text, maxWidth) {
  const words = text.split(' ');
  const lines = [];
  let line = '';
  for (const w of words) {
    const probe = line ? line + ' ' + w : w;
    if (ctx.measureText(probe).width > maxWidth && line) {
      lines.push(line);
      line = w;
    } else {
      line = probe;
    }
  }
  if (line) lines.push(line);
  return lines;
}

const MAGMA = [
  [0.0, [0, 0, 4]], [0.25, [80, 18, 123]], [0.5, [182, 54, 121]],
  [0.75, [251, 136, 97]], [1.0, [252, 253, 191]],
];

function magma(t) {
  const v = Math.max(0, Math.min(1, t));
  for (let i = 1; i < MAGMA.length; i++) {
    if (v <= MAGMA[i][0]) {
      const [t0, c0] = MAGMA[i - 1];
      const [t1, c1] = MAGMA[i];
      const f = (v - t0) / (t1 - t0);
      return [c0[0] + (c1[0] - c0[0]) * f, c0[1] + (c1[1] - c0[1]) * f, c0[2] + (c1[2] - c0[2]) * f];
    }
  }
  return MAGMA[MAGMA.length - 1][1];
}

export async function runFramesShowcase({ synth, g2p, symToId, cfg }, text, { seed = 0, steps = 4 } = {}) {
  // 1) synthesize up front, measuring the real wall time shown in the video
  const chunks = await phonemize(g2p, symToId, text);
  const t0 = performance.now();
  const results = [];
  for (const c of chunks) results.push(await synth.run(c.ids, { steps, seed }));
  const synthMs = performance.now() - t0;

  const totalSamples = results.reduce((n, r) => n + r.wav.length, 0);
  const wav = new Float32Array(totalSamples);
  let off = 0;
  for (const r of results) {
    wav.set(r.wav, off);
    off += r.wav.length;
  }
  const audioSec = totalSamples / cfg.sample_rate;

  // mel strip: 1px per frame × n_feats, low bins at the bottom
  const F = cfg.n_feats;
  const totalFrames = results.reduce((n, r) => n + r.ylen, 0);
  const mel = document.createElement('canvas');
  mel.width = totalFrames;
  mel.height = F;
  const mctx = mel.getContext('2d');
  const lo = cfg.mel_mean - 2.5 * cfg.mel_std;
  const hi = cfg.mel_mean + 3.0 * cfg.mel_std;
  let colOff = 0;
  for (const r of results) {
    const img = mctx.createImageData(r.ylen, F);
    for (let f = 0; f < r.ylen; f++) {
      for (let c = 0; c < F; c++) {
        const [rr, gg, bb] = magma((r.mel[c * cfg.MAX_MEL + f] - lo) / (hi - lo));
        const o = ((F - 1 - c) * r.ylen + f) * 4;
        img.data[o] = rr; img.data[o + 1] = gg; img.data[o + 2] = bb; img.data[o + 3] = 255;
      }
    }
    mctx.putImageData(img, colOff, 0);
    colOff += r.ylen;
  }

  // waveform min/max per output pixel column, precomputed
  const plotW = W - 2 * MARGIN;
  const wavCols = [];
  for (let px = 0; px < plotW; px++) {
    const s0 = Math.floor((px / plotW) * totalSamples);
    const s1 = Math.max(s0 + 1, Math.floor(((px + 1) / plotW) * totalSamples));
    let mn = 1;
    let mx = -1;
    for (let s = s0; s < s1; s++) {
      const v = wav[s];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    wavCols.push([mn, mx]);
  }

  // 2) frame rendering
  const TYPE = Math.min(3, 0.9 + text.length / 40);
  const GEN = Math.max(0.9, Math.min(synthMs / 1000, 3.0));
  const TAIL = 0.9;
  const capture = document.createElement('canvas');
  capture.width = W;
  capture.height = H;
  const ctx = capture.getContext('2d');

  let frameIndex = 0;
  const post = async () => {
    const jpeg = await new Promise((r) => capture.toBlob(r, 'image/jpeg', 0.92));
    const name = `frames/f_${String(frameIndex).padStart(4, '0')}.jpg`;
    await fetch(`/__save?name=${name}`, { method: 'POST', body: jpeg });
    frameIndex += 1;
  };

  const secLabel = (audioSec).toFixed(1);
  const genLabel = (synthMs / 1000).toFixed(1);

  function drawFrame(t) {
    ctx.fillStyle = '#0b0d10';
    ctx.fillRect(0, 0, W, H);

    // sentence, typewriter during TYPE
    ctx.font = '600 40px -apple-system, "SF Pro Display", "Segoe UI", sans-serif';
    ctx.textBaseline = 'top';
    const shown = t < TYPE
      ? text.slice(0, Math.floor((t / TYPE) * text.length))
      : text;
    ctx.fillStyle = t < TYPE + GEN ? '#e8eaed' : 'rgba(232,234,237,0.55)';
    const lines = wrapText(ctx, shown, W - 2 * MARGIN);
    lines.forEach((line, i) => ctx.fillText(line, MARGIN, 96 + i * 54));
    if (t < TYPE && Math.floor(t * 2.5) % 2 === 0) {
      const last = lines[lines.length - 1] ?? '';
      const cx = MARGIN + ctx.measureText(last).width + 6;
      ctx.fillRect(cx, 96 + (lines.length - 1) * 54, 3, 44);
    }

    if (t >= TYPE && t < TYPE + GEN) {
      // generating beat
      const p = (t - TYPE) / GEN;
      ctx.font = '500 26px -apple-system, "SF Pro Text", sans-serif';
      ctx.fillStyle = '#7cc4ff';
      const dots = '·'.repeat(1 + (Math.floor((t - TYPE) * 4) % 3));
      ctx.fillText(`generating speech on this device ${dots}`, MARGIN, 330);
      ctx.fillStyle = 'rgba(255,255,255,0.12)';
      ctx.fillRect(MARGIN, 380, plotW, 5);
      ctx.fillStyle = '#7cc4ff';
      ctx.fillRect(MARGIN, 380, plotW * p, 5);
    }

    if (t >= TYPE + GEN) {
      const tp = Math.min((t - TYPE - GEN) / audioSec, 1);
      const specY = 330;
      const specH = 220;
      const wavY = 575;
      const wavH = 90;
      // full spectrogram dimmed, played part at full brightness
      ctx.globalAlpha = 0.3;
      ctx.drawImage(mel, 0, 0, totalFrames, F, MARGIN, specY, plotW, specH);
      ctx.globalAlpha = 1;
      const px = Math.floor(tp * plotW);
      if (px > 0) {
        ctx.drawImage(mel, 0, 0, tp * totalFrames, F, MARGIN, specY, px, specH);
      }
      // waveform
      ctx.strokeStyle = 'rgba(124, 196, 255, 0.9)';
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      const mid = wavY + wavH / 2;
      for (let x = 0; x < px; x++) {
        const [mn, mx] = wavCols[x];
        ctx.moveTo(MARGIN + x + 0.5, mid - mx * (wavH / 2));
        ctx.lineTo(MARGIN + x + 0.5, mid - mn * (wavH / 2) + 1);
      }
      ctx.stroke();
      // playhead
      if (tp < 1) {
        ctx.strokeStyle = 'rgba(232,234,237,0.9)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(MARGIN + px, specY - 12);
        ctx.lineTo(MARGIN + px, wavY + wavH + 12);
        ctx.stroke();
      }
      // latency chip
      ctx.font = '500 24px -apple-system, "SF Pro Text", sans-serif';
      ctx.fillStyle = '#9aa0a6';
      const chip = `${secLabel} s of speech synthesized in ${genLabel} s`;
      ctx.fillText(chip, MARGIN, specY - 46);
    }

    // caption bar
    ctx.font = '500 23px -apple-system, "SF Pro Text", sans-serif';
    ctx.fillStyle = '#9aa0a6';
    ctx.textAlign = 'center';
    ctx.fillText('Type → voice, entirely in your browser · LiteRT.js (WebGPU + WASM) · nothing leaves the page', W / 2, H - 48);
    ctx.textAlign = 'left';
  }

  const totalSec = TYPE + GEN + audioSec + TAIL;
  const nFrames = Math.round(totalSec * FPS);
  for (let i = 0; i < nFrames; i++) {
    drawFrame(i / FPS);
    await post();
  }

  await fetch('/__save?name=frames/audio.wav', { method: 'POST', body: encodeWav(wav, cfg.sample_rate) });
  await fetch('/__save?name=frames/meta.json', {
    method: 'POST',
    body: JSON.stringify({
      fps: FPS, frames: frameIndex, typeSec: TYPE, genSec: GEN,
      audioSec: +audioSec.toFixed(3), tailSec: TAIL, synthMs: +synthMs.toFixed(0),
      audioOffsetSec: +(TYPE + GEN).toFixed(3), text, steps, seed,
    }, null, 2),
  });
  await fetch('/__save?name=frames/DONE', { method: 'POST', body: String(frameIndex) });
  return frameIndex;
}
