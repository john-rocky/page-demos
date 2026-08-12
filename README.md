# litertjs-demos

Small browser demos that run real ML models **entirely in your browser** with
[LiteRT.js](https://www.npmjs.com/package/@litertjs/core) (WebGPU). No server,
no upload — the input never leaves the page.

> An independent demo project built with LiteRT.js. Not an official Google project.

## Demos

### Photo → 3D
A single photo becomes an orbitable 3D point cloud.
[MoGe-2](https://huggingface.co/litert-community/MoGe-2-LiteRT) (monocular
geometry, MIT) recovers a per-pixel 3D point map in one forward pass; drawn
straight to a point cloud with three.js. ~80 ms/photo on an M4 Max (Chrome
WebGPU); falls back to WASM when WebGPU is unavailable.

```sh
npm install
npm run dev        # http://localhost:5173
```

Inputs: photo file / drag & drop / paste / webcam. Debug: `?img=<url>`.

## How it works
1. Center/contain-fit the image → 448×448 NCHW float32 in [0,1].
2. `@litertjs/core` `loadAndCompile(bytes, { accelerator: 'webgpu' })`, one
   `model.run()` per photo.
3. Outputs (order-ambiguous in the .tflite) resolved by shape/range; masked
   points drawn as a colored cloud.

## Credits
Showcase animal photos: [Pexels](https://www.pexels.com) (Pexels License, no
attribution required). See `SHOWCASE_CREDITS.md`.
