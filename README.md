# Photo → 3D point cloud, in your browser

One photo becomes an orbitable 3D point cloud, entirely client-side:
[MoGe-2](https://huggingface.co/litert-community/MoGe-2-LiteRT) (monocular
geometry, MIT) running on LiteRT.js with WebGPU. Nothing is uploaded — the
photo never leaves the page.

Measured on a Mac Studio (M4 Max), Chrome WebGPU: ~65–85 ms per inference.
First page load downloads the 129 MB model once (then cached via the Cache
Storage API). Falls back to WASM when WebGPU is unavailable (~4 s per run).

## Run

```sh
npm install
npm run dev      # http://localhost:5173
```

Inputs: photo file / drag & drop / paste (Cmd+V) / webcam capture.
Debug hook: `?img=<url>` runs a fetched image automatically.

## How it works

1. Center-crop → 448×448 → NCHW float32 in [0, 1] (the encoder applies its
   own normalization in-graph).
2. `@litertjs/core` `loadAndCompile(bytes, { accelerator: 'webgpu' })`,
   one `model.run()` per photo.
3. Outputs are order-ambiguous in the .tflite, so they are resolved by shape
   and range (points = the [448,448,3] map with values beyond ±1; normals are
   unit vectors; mask is the sigmoid [448,448,1]; scale is the scalar).
4. Points with mask ≤ 0.5 are dropped; the far-depth tail (> 6× median depth)
   is trimmed for viewing; the cloud stays in camera coordinates and the
   three.js camera starts AT the capture viewpoint, so the cloud initially
   looks exactly like the photo and the orbit reveals the parallax.
