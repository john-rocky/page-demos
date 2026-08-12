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
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
stage.appendChild(renderer.domElement);

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
const ORBIT_AMPLITUDE = 0.32; // radians (~18°) each way — enough parallax, no background streak
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
renderer.setAnimationLoop(() => {
  const now = performance.now();
  const dt = Math.min((now - lastFrame) / 1000, 0.05);
  lastFrame = now;
  if (autoOrbit && cloud) {
    orbitClock += dt;
    const angle = ORBIT_AMPLITUDE * Math.sin((orbitClock / ORBIT_PERIOD) * Math.PI * 2);
    // Orbit around the subject's own centroid at the framing distance, so the
    // whole subject stays centered and in-frame as it swings.
    camera.position.set(
      subjectCenter.x + Math.sin(angle) * viewDistance,
      subjectCenter.y,
      subjectCenter.z + Math.cos(angle) * viewDistance,
    );
    camera.lookAt(subjectCenter);
  } else {
    controls.update();
  }
  renderer.render(scene, camera);
});

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

    const testUrl = new URLSearchParams(location.search).get('img');
    if (testUrl) {
      const blob = await (await fetch(testUrl)).blob();
      await runOnImage(await createImageBitmap(blob));
    }
  } catch (err) {
    status(`Failed to start: ${err instanceof Error ? err.message : err}`);
    throw err;
  }
}

// --- inference ------------------------------------------------------------

const cropCanvas = document.createElement('canvas');
cropCanvas.width = SIZE;
cropCanvas.height = SIZE;
const cropCtx = cropCanvas.getContext('2d', { willReadFrequently: true });

function preprocess(source, sourceWidth, sourceHeight) {
  const side = Math.min(sourceWidth, sourceHeight);
  const sx = (sourceWidth - side) / 2;
  const sy = (sourceHeight - side) / 2;
  cropCtx.drawImage(source, sx, sy, side, side, 0, 0, SIZE, SIZE);
  const { data } = cropCtx.getImageData(0, 0, SIZE, SIZE);
  const plane = SIZE * SIZE;
  const nchw = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    nchw[i] = data[i * 4] / 255;
    nchw[plane + i] = data[i * 4 + 1] / 255;
    nchw[2 * plane + i] = data[i * 4 + 2] / 255;
  }
  return { nchw, rgba: data };
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

function buildCloud(points, mask, rgba) {
  const plane = SIZE * SIZE;
  const positions = [];
  const colors = [];
  const border = 4; // photo-border pixels have ambiguous depth and smear into "curtains"
  for (let i = 0; i < plane; i++) {
    const px = i % SIZE;
    const py = (i / SIZE) | 0;
    if (px < border || px >= SIZE - border || py < border || py >= SIZE - border) continue;
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

  // Robust centroid (median per axis) — the subject's mass, ignoring the
  // smeared silhouette edges and far background. The orbit pivots here so the
  // subject stays centered and never translates off-frame while swinging.
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
    const { nchw, rgba } = preprocess(source, width, height);
    const { points, mask, elapsed } = await infer(nchw);
    if (cloud) {
      scene.remove(cloud);
      cloud.geometry.dispose();
      cloud.material.dispose();
    }
    cloud = buildCloud(points, mask, rgba);
    scene.add(cloud);
    // Pivot the orbit on the subject's robust centroid so it stays centered
    // (never translates off-frame); keep the tuned viewing distance that
    // frames the subject large. Start on the capture axis so the cloud opens
    // looking exactly like the photo.
    subjectCenter = cloud.userData.center.clone();
    viewDistance = DEPTH_ANCHOR; // matches the tuned framing of the capture view
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
