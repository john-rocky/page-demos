/**
 * Photo → 3D point cloud, fully client-side.
 *
 * Pipeline: image → center-crop → 448×448 NCHW [0,1] float32 → MoGe-2
 * (LiteRT.js, WebGPU with WASM fallback) → per-pixel affine point map +
 * confidence mask → three.js point cloud colored from the photo.
 *
 * Model I/O (from the conversion script, LiteRT-Models/moge):
 *   input : [1, 3, 448, 448] float32, range [0, 1] (no ImageNet norm)
 *   outputs (order not guaranteed in the .tflite — resolved by shape/range):
 *     points [1,448,448,3] · normal [1,448,448,3] · mask(sigmoid) [1,448,448,1]
 *     · metric scale [1,1,1,1]
 */
import {
  Tensor,
  isWebGPUSupported,
  loadAndCompile,
  loadLiteRt,
} from '@litertjs/core';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const MODEL_URL =
  'https://huggingface.co/litert-community/MoGe-2-LiteRT/resolve/main/moge.tflite';
const SIZE = 448;
const MASK_THRESHOLD = 0.5;
// World-space depth (in units) the photo's median depth is mapped to.
const DEPTH_ANCHOR = 1.6;

const statusEl = document.getElementById('status');
const latencyEl = document.getElementById('latency');
const latencyValueEl = latencyEl.querySelector('b');
const backendEl = document.getElementById('backend');
const hintEl = document.getElementById('hint');
const fileEl = document.getElementById('file');
const camBtn = document.getElementById('cam');
const shutterBtn = document.getElementById('shutter');
const videoEl = document.getElementById('video');
const dropOverlay = document.getElementById('drop-overlay');

let model = null;
let accelerator = 'wasm';
let cloud = null;
let runCount = 0;

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

// --- three.js scene -------------------------------------------------------

const stage = document.getElementById('stage');
// preserveDrawingBuffer so canvas.captureStream() reliably reads finished
// frames (WebGL clears the buffer after present otherwise).
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
stage.appendChild(renderer.domElement);

// When recording the showcase, we drive frames explicitly so the captured
// stream matches the rendered frames exactly.
let recordTrack = null;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d10);
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
camera.position.set(0, 0, 2.6);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

// Instead of a full auto-rotate (which eventually shows the hollow back and
// blows up the far-background points as the camera swings close), gently
// oscillate the viewpoint left↔right around the capture direction — enough
// parallax to read as 3D, always looking at the subject's good side.
let autoOrbit = true;
let orbitClock = 0;
// Interactive default: a gentle ±18° swing. Showcase mode (below) drives a
// wider, richer "look-around" path per animal.
let orbitYawAmp = 0.32; // radians each way
let orbitPitchAmp = 0.0;
const ORBIT_PERIOD = 7; // seconds per full left-right-left cycle
// Set per-cloud so the whole subject stays framed as the camera swings.
let subjectCenter = new THREE.Vector3(0, 0, -DEPTH_ANCHOR);
let viewDistance = DEPTH_ANCHOR;
renderer.domElement.addEventListener('pointerdown', () => {
  autoOrbit = false;
});

function resize() {
  const { innerWidth: w, innerHeight: h } = window;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

let lastFrame = performance.now();
function animate() {
  const now = performance.now();
  const dt = Math.min((now - lastFrame) / 1000, 0.05);
  lastFrame = now;
  if (autoOrbit && cloud) {
    orbitClock += dt;
    const t = (orbitClock / ORBIT_PERIOD) * Math.PI * 2;
    const yaw = orbitYawAmp * Math.sin(t);
    const pitch = orbitPitchAmp * Math.sin(t * 2); // figure-8 look-around
    // Orbit around the subject's own centroid at the framing distance, so the
    // whole subject stays centered and in-frame while the view moves.
    const horiz = Math.cos(pitch) * viewDistance;
    camera.position.set(
      subjectCenter.x + Math.sin(yaw) * horiz,
      subjectCenter.y + Math.sin(pitch) * viewDistance,
      subjectCenter.z + Math.cos(yaw) * horiz,
    );
    camera.lookAt(subjectCenter);
  } else {
    controls.update();
  }
  renderer.render(scene, camera);
  if (recordTrack) recordTrack.requestFrame();
}
renderer.setAnimationLoop(animate);

// --- model loading --------------------------------------------------------

async function fetchModelBytes() {
  const cache = 'caches' in window ? await caches.open('moge-demo-v1') : null;
  if (cache) {
    const hit = await cache.match(MODEL_URL);
    if (hit) {
      status('Loading model from cache…');
      return new Uint8Array(await hit.arrayBuffer());
    }
  }
  const response = await fetch(MODEL_URL);
  if (!response.ok) throw new Error(`model download failed: HTTP ${response.status}`);
  const total = Number(response.headers.get('Content-Length')) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    const mb = (received / 1048576).toFixed(0);
    status(
      total
        ? `Downloading model (one-time)… ${mb} MB`
        : `Downloading model… ${mb} MB`,
      total ? received / total : null,
    );
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  if (cache) {
    await cache.put(MODEL_URL, new Response(bytes.slice().buffer));
  }
  return bytes;
}

async function boot() {
  try {
    status('Loading runtime…');
    try {
      await loadLiteRt('/litert-wasm/', { jspi: true });
    } catch {
      await loadLiteRt('/litert-wasm/', { jspi: false });
    }
    accelerator = isWebGPUSupported() ? 'webgpu' : 'wasm';
    backendEl.textContent =
      accelerator === 'webgpu' ? 'WebGPU · your device' : 'WASM (no WebGPU) · your device';

    const bytes = await fetchModelBytes();
    status('Compiling for ' + (accelerator === 'webgpu' ? 'WebGPU' : 'WASM') + '…');
    model = await loadAndCompile(bytes, { accelerator });
    status('Ready — choose a photo, use the camera, or drop an image.');

    const params = new URLSearchParams(location.search);
    const testUrl = params.get('img');
    if (testUrl) {
      const blob = await (await fetch(testUrl)).blob();
      await runOnImage(await createImageBitmap(blob));
    }
    if (params.get('frames') === '1') {
      await renderFramesShowcase();
    } else if (params.get('show') === '1') {
      await runShowcase(params.get('record') === '1');
    }
  } catch (err) {
    status(`Failed to start: ${err instanceof Error ? err.message : err}`);
    throw err;
  }
}

// --- showcase: several exotic animals in sequence, wider look-around orbit --
// Sources are Wikimedia Commons; licenses noted in SHOWCASE_CREDITS.md. Final
// image sourcing/attribution for any public post is the owner's to confirm.
// One striking real photo per animal class — diversity over "another lizard".
const SHOWCASE = [
  { url: 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Barn_Owl_R1_1791.jpg/1280px-Barn_Owl_R1_1791.jpg', label: 'Barn owl' },
  { url: 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Portrait_of_a_red_fox_in_Rautas_fj%C3%A4llurskog.jpg/1280px-Portrait_of_a_red_fox_in_Rautas_fj%C3%A4llurskog.jpg', label: 'Red fox' },
  { url: 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Dendrobates_azureus_%28Dendrobates_tinctorius%29_Edit.jpg/1280px-Dendrobates_azureus_%28Dendrobates_tinctorius%29_Edit.jpg', label: 'Poison dart frog' },
  { url: 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/Pulpo_com%C3%BAn_%28Octopus_vulgaris%29%2C_Parque_natural_de_la_Arr%C3%A1bida%2C_Portugal%2C_2020-07-21%2C_DD_33.jpg/1280px-Pulpo_com%C3%BAn_%28Octopus_vulgaris%29%2C_Parque_natural_de_la_Arr%C3%A1bida%2C_Portugal%2C_2020-07-21%2C_DD_33.jpg', label: 'Octopus' },
  { url: 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Panther_chameleon_%28Furcifer_pardalis%29_male_Nosy_Be.jpg/1280px-Panther_chameleon_%28Furcifer_pardalis%29_male_Nosy_Be.jpg', label: 'Panther chameleon' },
  { url: 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/62/European_praying_mantis_%28Mantis_religiosa%29_green_female_Dobruja.jpg/1280px-European_praying_mantis_%28Mantis_religiosa%29_green_female_Dobruja.jpg', label: 'Praying mantis' },
];

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Place the camera for the orbit phase at progress p∈[0,1]. A single-view
 * point cloud has no geometry on its far side, so the swing is deliberately
 * shallow (±~15° yaw, a touch of pitch) — enough parallax to read as 3D
 * without turning far enough to expose the hollow back. */
function poseCamera(p) {
  const yaw = 0.26 * Math.sin(p * Math.PI * 2);
  const pitch = 0.08 * Math.sin(p * Math.PI * 2 + Math.PI / 2);
  const horiz = Math.cos(pitch) * viewDistance;
  camera.position.set(
    subjectCenter.x + Math.sin(yaw) * horiz,
    subjectCenter.y + Math.sin(pitch) * viewDistance,
    subjectCenter.z + Math.cos(yaw) * horiz,
  );
  camera.lookAt(subjectCenter);
}

/** Draw the source image contain-fit (letterboxed) into a 2D context of
 * WxH — the same "whole photo" the point cloud is built from. */
function drawPhoto(ctx, img, W, H) {
  const scale = Math.min(W / img.width, H / img.height);
  const dw = img.width * scale;
  const dh = img.height * scale;
  ctx.fillStyle = '#0b0d10';
  ctx.fillRect(0, 0, W, H);
  ctx.drawImage(img, (W - dw) / 2, (H - dh) / 2, dw, dh);
}

/** Deterministic frame dump → server → ffmpeg (reliable where canvas
 * captureStream yields nothing). Renders each animal for FRAMES_PER frames
 * at a fixed 16:9 size, posts each as a JPEG the dev server writes to disk. */
async function renderFramesShowcase() {
  const W = 1280;
  const H = 720;
  const FPS = 30;
  // Per-animal beat: hold the 2D photo, dissolve into points, then orbit.
  const PHOTO = Math.round(FPS * 0.9);
  const FADE = Math.round(FPS * 0.5);
  const ORBIT = Math.round(FPS * 2.6);
  const FRAMES_PER = PHOTO + FADE + ORBIT;
  document.querySelector('.panel').style.display = 'none';
  autoOrbit = false;
  renderer.setAnimationLoop(null); // we render each frame by hand
  renderer.setPixelRatio(1);
  renderer.setSize(W, H);
  camera.aspect = W / H;
  camera.updateProjectionMatrix();

  const capture = document.createElement('canvas');
  capture.width = W;
  capture.height = H;
  const cctx = capture.getContext('2d');

  let frameIndex = 0;
  const postFrame = async () => {
    const jpeg = await new Promise((r) => capture.toBlob(r, 'image/jpeg', 0.92));
    const name = `frames/f_${String(frameIndex).padStart(4, '0')}.jpg`;
    await fetch(`/__save?name=${name}`, { method: 'POST', body: jpeg });
    frameIndex += 1;
  };

  for (const item of SHOWCASE) {
    let bitmap;
    try {
      const blob = await (await fetch(item.url)).blob();
      bitmap = await createImageBitmap(blob);
      await runOnImage(bitmap);
      autoOrbit = false; // we drive the camera per frame
    } catch {
      continue;
    }
    // Camera holds at the capture axis through photo + fade (cloud aligns
    // with the photo there); orbit only after.
    poseCamera(0);
    renderer.render(scene, camera);

    // Phase 1 — the flat photo.
    drawPhoto(cctx, bitmap, W, H);
    for (let f = 0; f < PHOTO; f++) await postFrame();

    // Phase 2 — dissolve the photo into the point cloud (same viewpoint).
    for (let f = 0; f < FADE; f++) {
      drawPhoto(cctx, bitmap, W, H);
      cctx.globalAlpha = (f + 1) / FADE;
      cctx.drawImage(renderer.domElement, 0, 0, W, H);
      cctx.globalAlpha = 1;
      await postFrame();
    }

    // Phase 3 — the point cloud, gentle orbit.
    for (let f = 0; f < ORBIT; f++) {
      poseCamera(f / ORBIT);
      renderer.render(scene, camera);
      cctx.drawImage(renderer.domElement, 0, 0, W, H);
      await postFrame();
    }
    status(`captured ${item.label}`);
  }
  await fetch('/__save?name=frames/DONE', { method: 'POST', body: String(frameIndex) });
  document.querySelector('.panel').style.display = '';
  status(`Frame dump complete (${frameIndex} frames).`);
  // Restore interactive rendering.
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  resize();
  renderer.setAnimationLoop(animate);
}

async function runShowcase(record) {
  document.querySelector('.panel').style.display = 'none';
  let recorder = null;
  let chunks = [];
  if (record && renderer.domElement.captureStream) {
    // captureStream(0): frames are pushed manually via track.requestFrame()
    // from the render loop — exact, and reliable on a WebGL canvas.
    const stream = renderer.domElement.captureStream(0);
    recordTrack = stream.getVideoTracks()[0];
    const mime = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
      ? 'video/webm;codecs=vp9'
      : 'video/webm';
    recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 16_000_000 });
    recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    recorder.start(200);
  }

  for (const item of SHOWCASE) {
    try {
      const blob = await (await fetch(item.url)).blob();
      const bitmap = await createImageBitmap(blob);
      await runOnImage(bitmap);
      // Shallow look-around (no hollow-back exposure); reset the clock so each
      // animal starts at the capture view (opens as the photo) then explores.
      orbitYawAmp = 0.26;
      orbitPitchAmp = 0.08;
      orbitClock = 0;
      await delay(4200); // ~2/3 of one look-around cycle, ending near center
    } catch {
      // Skip an image that fails to load; keep the reel going.
    }
  }
  // Restore interactive defaults.
  orbitYawAmp = 0.32;
  orbitPitchAmp = 0.0;

  if (recorder) {
    await delay(400);
    recorder.stop();
    await new Promise((resolve) => {
      recorder.onstop = resolve;
    });
    recordTrack = null;
    const out = new Blob(chunks, { type: 'video/webm' });
    // Save via the dev-server endpoint (programmatic downloads are blocked
    // without a user gesture); fall back to a download link if it fails.
    try {
      await fetch('/__save?name=moge-showcase.webm', { method: 'POST', body: out });
      status(`Showcase saved (${(out.size / 1048576).toFixed(1)} MB).`);
    } catch {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(out);
      a.download = 'moge-showcase.webm';
      a.click();
    }
  }
  document.querySelector('.panel').style.display = '';
}

// --- inference ------------------------------------------------------------

const cropCanvas = document.createElement('canvas');
cropCanvas.width = SIZE;
cropCanvas.height = SIZE;
const cropCtx = cropCanvas.getContext('2d', { willReadFrequently: true });

function preprocess(source, sourceWidth, sourceHeight) {
  // Contain-fit (letterbox), NOT center-crop: the whole photo must survive so
  // the subject is never cut off. The padded margins are recorded and later
  // excluded from the point cloud so they don't become a flat backdrop.
  const scale = Math.min(SIZE / sourceWidth, SIZE / sourceHeight);
  const drawW = Math.round(sourceWidth * scale);
  const drawH = Math.round(sourceHeight * scale);
  const offX = Math.floor((SIZE - drawW) / 2);
  const offY = Math.floor((SIZE - drawH) / 2);
  // Pad with edge-ish neutral so MoGe sees a plausible background rather than a
  // hard black frame; the pad is dropped from the cloud regardless.
  cropCtx.fillStyle = '#7f7f7f';
  cropCtx.fillRect(0, 0, SIZE, SIZE);
  cropCtx.drawImage(source, 0, 0, sourceWidth, sourceHeight, offX, offY, drawW, drawH);
  const { data } = cropCtx.getImageData(0, 0, SIZE, SIZE);
  const plane = SIZE * SIZE;
  const nchw = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    nchw[i] = data[i * 4] / 255;
    nchw[plane + i] = data[i * 4 + 1] / 255;
    nchw[2 * plane + i] = data[i * 4 + 2] / 255;
  }
  // Valid = inside the drawn content rect (1 = photo pixel, 0 = padding).
  const valid = new Uint8Array(plane);
  for (let y = offY; y < offY + drawH; y++) {
    for (let x = offX; x < offX + drawW; x++) valid[y * SIZE + x] = 1;
  }
  return { nchw, rgba: data, valid };
}

function sampleAbsMax(array) {
  let max = 0;
  const step = Math.max(1, Math.floor(array.length / 5000));
  for (let i = 0; i < array.length; i += step) {
    const v = Math.abs(array[i]);
    if (v > max) max = v;
  }
  return max;
}

/** The .tflite output order is not guaranteed; identify by shape and range
 * (same strategy as the reference Android app): points is the [h,w,3] map
 * with values beyond [-1,1]; normals are unit vectors. */
function resolveOutputs(buffers) {
  const plane = SIZE * SIZE;
  const big = buffers.filter((b) => b.length === plane * 3);
  const mask = buffers.find((b) => b.length === plane);
  const scale = buffers.find((b) => b.length === 1);
  if (big.length !== 2 || !mask || !scale) {
    throw new Error('unexpected model outputs');
  }
  const points = sampleAbsMax(big[0]) > 2 ? big[0] : big[1];
  return { points, mask, scale: scale[0] };
}

async function infer(nchw) {
  const input = Tensor.fromTypedArray(nchw, [1, 3, SIZE, SIZE]);
  const start = performance.now();
  const outputs = await model.run([input]);
  const buffers = [];
  for (const output of outputs) {
    buffers.push(await output.data());
  }
  const elapsed = performance.now() - start;
  for (const output of outputs) output.delete();
  input.delete();
  return { ...resolveOutputs(buffers), elapsed };
}

// --- point cloud ----------------------------------------------------------

function buildCloud(points, mask, rgba, valid) {
  const plane = SIZE * SIZE;
  const positions = [];
  const colors = [];
  const border = 2; // trim 1-2px inside the content edge (ambiguous-depth rim)
  for (let i = 0; i < plane; i++) {
    if (!valid[i]) continue; // letterbox padding — not part of the photo
    const px = i % SIZE;
    const py = (i / SIZE) | 0;
    // Skip pixels adjacent to a padding pixel (the content-edge rim smears).
    if (
      !valid[i - 1] || !valid[i + 1] ||
      !valid[i - SIZE] || !valid[i + SIZE] ||
      px < border || px >= SIZE - border || py < border || py >= SIZE - border
    ) continue;
    if (mask[i] <= MASK_THRESHOLD) continue;
    const x = points[i * 3];
    const y = points[i * 3 + 1];
    const z = points[i * 3 + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    // MoGe camera frame (x right, y down, z forward) → three.js (y up, -z forward)
    positions.push(x, -y, -z);
    colors.push(rgba[i * 4] / 255, rgba[i * 4 + 1] / 255, rgba[i * 4 + 2] / 255);
  }
  if (positions.length === 0) throw new Error('no confident points in this photo');

  // Keep the cloud in camera coordinates and view it FROM the capture
  // viewpoint (three camera at the origin): the cloud starts out looking
  // exactly like the photo, and orbiting reveals the parallax. Scale so the
  // median depth sits at DEPTH_ANCHOR world units, and trim the far tail
  // (deep background shells) which otherwise dwarfs the subject.
  const count = positions.length / 3;
  const depthSample = [];
  const step = Math.max(1, Math.floor(count / 20000));
  for (let i = 0; i < count; i += step) depthSample.push(-positions[i * 3 + 2]);
  depthSample.sort((a, b) => a - b);
  const medianDepth = Math.max(depthSample[Math.floor(depthSample.length / 2)], 1e-6);
  const maxDepth = medianDepth * 3.5;
  const minDepth = medianDepth * 0.45; // trim the near lip (front sand edge streaks)
  const fit = DEPTH_ANCHOR / medianDepth;

  const keptPositions = [];
  const keptColors = [];
  const xs = [];
  const ys = [];
  const zs = [];
  for (let i = 0; i < count; i++) {
    const depth = -positions[i * 3 + 2];
    if (depth > maxDepth || depth < minDepth) continue;
    const x = positions[i * 3] * fit;
    const y = positions[i * 3 + 1] * fit;
    const z = positions[i * 3 + 2] * fit;
    keptPositions.push(x, y, z);
    keptColors.push(colors[i * 3], colors[i * 3 + 1], colors[i * 3 + 2]);
    xs.push(x);
    ys.push(y);
    zs.push(z);
  }

  // Robust screen-plane framing from percentiles (ignores the smeared
  // silhouette rim and stray points): center on the 2–98% box, size on its
  // half-extent. The orbit pivots on this center so the subject stays put,
  // and the viewing distance fits that extent with margin.
  const median = (arr) => {
    const s = arr.slice().sort((a, b) => a - b);
    return s[s.length >> 1];
  };
  const center = new THREE.Vector3(median(xs), median(ys), median(zs));

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(keptPositions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(keptColors, 3));

  const material = new THREE.PointsMaterial({
    size: 0.0075,
    vertexColors: true,
    sizeAttenuation: true,
  });
  const cloudPoints = new THREE.Points(geometry, material);
  cloudPoints.userData.center = center;
  return cloudPoints;
}

async function runOnImage(source) {
  if (!model) return;
  const width = source.videoWidth ?? source.width;
  const height = source.videoHeight ?? source.height;
  status('Running…');
  try {
    const { nchw, rgba, valid } = preprocess(source, width, height);
    const { points, mask, elapsed } = await infer(nchw);
    if (cloud) {
      scene.remove(cloud);
      cloud.geometry.dispose();
      cloud.material.dispose();
    }
    cloud = buildCloud(points, mask, rgba, valid);
    scene.add(cloud);
    // Pivot the orbit on the subject's robust centroid so it stays centered
    // (never translates off-frame); keep the tuned viewing distance that
    // frames the subject large. Start on the capture axis so the cloud opens
    // looking exactly like the photo.
    subjectCenter = cloud.userData.center.clone();
    viewDistance = DEPTH_ANCHOR; // tuned framing: subject large, whole photo shown
    autoOrbit = true;
    orbitClock = 0;
    camera.position.set(subjectCenter.x, subjectCenter.y, subjectCenter.z + viewDistance);
    camera.lookAt(subjectCenter);
    controls.target.copy(subjectCenter);

    runCount += 1;
    latencyValueEl.textContent = `${elapsed.toFixed(0)} ms`;
    latencyEl.style.display = 'block';
    hintEl.style.display = 'block';
    status(runCount === 1 && accelerator === 'webgpu'
      ? 'Done (first run includes GPU warm-up — try another photo).'
      : 'Done.');
  } catch (err) {
    status(`Failed: ${err instanceof Error ? err.message : err}`);
  }
}

// --- inputs: file, drop, paste, webcam ------------------------------------

fileEl.addEventListener('change', async () => {
  const file = fileEl.files?.[0];
  if (file) await runOnImage(await createImageBitmap(file));
  fileEl.value = '';
});

window.addEventListener('dragover', (event) => {
  event.preventDefault();
  dropOverlay.style.display = 'flex';
});
window.addEventListener('dragleave', (event) => {
  if (event.relatedTarget === null) dropOverlay.style.display = 'none';
});
window.addEventListener('drop', async (event) => {
  event.preventDefault();
  dropOverlay.style.display = 'none';
  const file = event.dataTransfer?.files?.[0];
  if (file && file.type.startsWith('image/')) {
    await runOnImage(await createImageBitmap(file));
  }
});

window.addEventListener('paste', async (event) => {
  const item = [...(event.clipboardData?.items ?? [])].find((entry) =>
    entry.type.startsWith('image/'),
  );
  const file = item?.getAsFile();
  if (file) await runOnImage(await createImageBitmap(file));
});

let stream = null;
camBtn.addEventListener('click', async () => {
  if (stream) {
    stopCamera();
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' },
    });
    videoEl.srcObject = stream;
    videoEl.style.display = 'block';
    shutterBtn.style.display = 'inline-block';
    camBtn.textContent = 'Stop camera';
  } catch (err) {
    status(`Camera: ${err instanceof Error ? err.message : err}`);
  }
});

function stopCamera() {
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  videoEl.srcObject = null;
  videoEl.style.display = 'none';
  shutterBtn.style.display = 'none';
  camBtn.textContent = 'Use camera';
}

shutterBtn.addEventListener('click', async () => {
  if (videoEl.videoWidth) {
    await runOnImage(videoEl);
    stopCamera();
  }
});

boot();
