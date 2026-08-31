# page-demos

Small browser demos that run real ML models **entirely in your browser** with
[LiteRT.js](https://www.npmjs.com/package/@litertjs/core) (WebGPU). No server,
no upload — the input never leaves the page.

> An independent demo project built with LiteRT.js. Not an official Google project.

The root page (`index.html`) lists the demos; each model has its own page
with a WebGPU ↔ WASM switch, a one-shot warm-up run (discarded, so the
displayed latency is real), and an env label (`webgpu` / `wasm` /
`wasm·1-thread`).

## Demos

### Photo → 3D
A single photo becomes an orbitable 3D point cloud.
[MoGe-2](https://huggingface.co/litert-community/MoGe-2-LiteRT) (monocular
geometry, MIT) recovers a per-pixel 3D point map in one forward pass; drawn
straight to a point cloud with three.js. ~80 ms/photo on an M4 Max (Chrome
WebGPU); falls back to WASM when WebGPU is unavailable.

```sh
npm install
npm run dev        # http://localhost:5173/moge/
```

Inputs: photo file / drag & drop / paste / webcam.
Debug: `?img=<url>&backend=wasm`.

### Text → speech
Type a sentence and a voice speaks it — no cloud TTS API, nothing leaves the
page. [Matcha-TTS](https://huggingface.co/litert-community/Matcha-TTS)
(flow-matching acoustic model + HiFi-GAN vocoder, MIT, ~90 MB fp16) with G2P
from a 275k-word espeak-IPA dictionary plus a DeepPhonemizer fallback. Text
encoder and vocoder run on WebGPU, the CFM decoder on WASM/XNNPACK (the GPU
delegate mis-fuses that graph — see the model card). ~1.5 s to first audio on
an M4 Max at the default 4 ODE steps, RTF ≈ 0.5.

```sh
cd matcha-tts
npm install
npm run dev        # http://localhost:5173
```

Debug: `?text=…&steps=…&seed=…&voc=wasm` (see `matcha-tts/src/main.js`).

> **Grown into a Chrome extension →
> [Page Voice](https://github.com/john-rocky/page-voice)**: the same local
> TTS pipeline as a right-click "Read aloud" on any page, plus auto-reading
> of streaming ChatGPT/Claude/Gemini replies.

## How it works
1. Center/contain-fit the image → 448×448 NCHW float32 in [0,1].
2. `@litertjs/core` `loadAndCompile(bytes, { accelerator: 'webgpu' })`, one
   `model.run()` per photo.
3. Outputs (order-ambiguous in the .tflite) resolved by shape/range; masked
   points drawn as a colored cloud.

## Deploy

`npm run deploy` builds the landing page and both demos into one `dist/` and
pushes it to the `gh-pages` branch. (`npm run build` alone rebuilds only the
root project and empties `dist/` — use `build:all` when the TTS demo must be
included.)

## Credits
Showcase animal photos: [Pexels](https://www.pexels.com) (Pexels License, no
attribution required). See `SHOWCASE_CREDITS.md`.
