/**
 * Face API Server — persistent Express wrapper around face-api.js.
 * Replaces the Vercel cold-start function with an always-warm server.
 * POST /api/descriptor  → { descriptor: [128 floats] | null }
 * POST /api/match       → { matched, employeeId, distance }
 */
const express = require('express');
const cors    = require('cors');
const faceapi = require('face-api.js');
const { Canvas, Image, ImageData, loadImage } = require('canvas');
const path    = require('path');

faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app = express();
app.use(cors());
app.use(express.json({ limit: '6mb' }));

const MODEL_PATH = path.join(__dirname, 'models');
let _loaded  = false;
let _loadP   = null;
let _matcher = null;  // FaceMatcher, built on demand

function ensureModels() {
  if (_loaded) return Promise.resolve(true);
  if (_loadP)  return _loadP;
  console.log('[FaceAPI] Loading models from', MODEL_PATH);
  _loadP = Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_PATH),
    faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_PATH),
    faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_PATH),
  ]).then(() => {
    _loaded = true;
    console.log('[FaceAPI] Models ready');
    return true;
  });
  return _loadP;
}

// Pre-load on startup so first request is fast
ensureModels().catch(console.error);

// ── Health ────────────────────────────────────────────────────────────────────
app.get('/health', (_req, res) =>
  res.json({ status: 'ok', models_loaded: _loaded })
);

// ── Descriptor extraction (same API as Vercel function) ───────────────────────
app.post('/api/descriptor', async (req, res) => {
  try {
    await ensureModels();
    const { imageBase64 } = req.body ?? {};
    if (!imageBase64) return res.status(400).json({ error: 'Missing imageBase64' });

    const buf = Buffer.from(imageBase64, 'base64');
    const img = await loadImage(buf);

    const det = await faceapi
      .detectSingleFace(img)
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!det) return res.json({ descriptor: null });
    return res.json({ descriptor: Array.from(det.descriptor) });
  } catch (e) {
    console.error('[/api/descriptor]', e.message);
    return res.status(500).json({ error: e.message });
  }
});

// ── Build/update matcher from employee descriptors ────────────────────────────
// Body: { employees: [{ id, name, descriptors: [[128 floats]] }] }
app.post('/api/matcher/init', async (req, res) => {
  try {
    await ensureModels();
    const { employees } = req.body ?? {};
    if (!employees?.length) return res.status(400).json({ error: 'No employees' });

    const labeled = employees
      .filter(e => e.descriptors?.length > 0)
      .map(e => new faceapi.LabeledFaceDescriptors(
        e.id,
        e.descriptors.map(d => new Float32Array(d))
      ));

    _matcher = new faceapi.FaceMatcher(labeled, 0.5);
    return res.json({ ok: true, count: labeled.length });
  } catch (e) {
    return res.status(500).json({ error: e.message });
  }
});

// ── Match a live frame against stored matcher ─────────────────────────────────
// Body: { imageBase64 }
app.post('/api/match', async (req, res) => {
  try {
    await ensureModels();
    if (!_matcher) return res.status(409).json({ error: 'Matcher not initialised. POST /api/matcher/init first.' });

    const { imageBase64 } = req.body ?? {};
    if (!imageBase64) return res.status(400).json({ error: 'Missing imageBase64' });

    const buf = Buffer.from(imageBase64, 'base64');
    const img = await loadImage(buf);

    const det = await faceapi
      .detectSingleFace(img)
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!det) return res.json({ detected: false });

    const result  = _matcher.findBestMatch(det.descriptor);
    const matched = result.label !== 'unknown';
    return res.json({
      detected:   true,
      matched,
      employeeId: matched ? result.label : null,
      distance:   result.distance,
    });
  } catch (e) {
    return res.status(500).json({ error: e.message });
  }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () =>
  console.log(`[FaceAPI] Server listening on port ${PORT}`)
);
