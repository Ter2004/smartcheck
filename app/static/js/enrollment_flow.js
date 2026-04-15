// enrollment_flow.js — SmartCheck Face Enrollment
// URL endpoints are injected by the template via window.ENROLL_CONFIG.
// ─────────────────────────────────────────────
// State
// ─────────────────────────────────────────────
let currentStep  = 1;
let baselineEAR  = null;
let calibStream  = null;
let captureStream = null;
let verifyStream = null;
let faceMeshCapture = null;
let captureCamera   = null;
let calibEARValues  = [];
let calibrating     = false;

// ─── Shared FaceMesh singleton ────────────────
// MediaPipe WASM can only be initialised once per page.
// All steps share one FaceMesh; only the Camera and onResults handler change.
let _sharedFM   = null;
let _stepCamera = null;   // currently active Camera (stopped between steps)

function _getSharedFM(opts = {}) {
    if (!_sharedFM) {
        _sharedFM = new FaceMesh({ locateFile: f =>
            `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}` });
    }
    _sharedFM.setOptions({
        maxNumFaces: 1,
        refineLandmarks:          opts.refineLandmarks          ?? false,
        minDetectionConfidence:   opts.minDetectionConfidence   ?? 0.7,
        minTrackingConfidence:    opts.minTrackingConfidence    ?? 0.7,
    });
    return _sharedFM;
}

function _stopStepCamera() {
    if (_stepCamera) { try { _stepCamera.stop(); } catch(e) {} _stepCamera = null; }
}

// Auto-capture state
const TOTAL_FRAMES        = 5;
const CAPTURE_INTERVAL_MS = 2000;   // capture every 2 s
let capturedImages   = [];          // base64 frames that passed quality gate
let lastCaptureTime  = 0;
let capturePaused    = false;       // paused while sending to backend

// B3: Interactive challenge attempt cap
const MAX_CHALLENGE_ATTEMPTS  = 5;
const CHALLENGE_COOLDOWN_MS   = 30000;  // 30 s cooldown after 5 failures
let challengeAttempts = 0;

// ─── Face Continuity state ────────────────────────────────────────────────────
// Face continuity check ย้ายไปทำที่ server (session["liveness_embeddings"])
// Client ไม่รับหรือเก็บ embedding ใดๆ อีกต่อไป

// ─── Real-time anti-spoof frame analysis (shared across liveness + capture) ───
// Runs every 250 ms inside any MediaPipe onResults callback.
//
// Moiré check: compute row-wise SAD (horizontal pixel differences) over 64×64 face ROI.
//   Screens have a regular pixel grid → all rows produce similar SAD → low coefficient
//   of variation (CV). Real faces have varied texture (eyes/nose/cheeks) → high CV.
//
// Edge check: run Sobel on an expanded face region (+18% padding) and accumulate
//   magnitude per column (catches vertical phone/paper borders) and per row (horizontal).
//   A phone border shows as ONE spike >> average near the image border zone.

const _RT_CANVAS = document.createElement('canvas');
_RT_CANVAS.width = 64; _RT_CANVAS.height = 64;
const _RT_CTX = _RT_CANVAS.getContext('2d', { willReadFrequently: true });

const RT_THROTTLE_MS   = 250;   // max analysis rate (ms between runs)
const RT_CONSEC_THRESH = 4;     // consecutive alert frames before blocking
const RT_MOIRE_CV_MAX  = 0.20;  // CV below this → screen (uniform texture)
const RT_MOIRE_SAD_MIN = 2.5;   // min mean SAD — skip near-blank/dark frames
const RT_EDGE_RATIO    = 4.5;   // Sobel peak/average ratio to flag a straight edge

let _rtLastCheck   = 0;
let _rtMoireConsec = 0;
let _rtEdgeConsec  = 0;

function _rtResetCounters() { _rtMoireConsec = 0; _rtEdgeConsec = 0; }

/**
 * Analyse one video frame for Moiré + straight-edge spoof indicators.
 * @param {HTMLVideoElement} video
 * @param {Array} landmarks  — MediaPipe face landmark array
 * @returns {null | { moireAlert, edgeAlert, blocked, reason, sadCV, sadMean }}
 */
function _rtAnalyzeFrame(video, landmarks) {
    const now = Date.now();
    if (now - _rtLastCheck < RT_THROTTLE_MS) return null;
    _rtLastCheck = now;

    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return null;

    // ── Face bounding box ──────────────────────────────────────────────────
    let minX = 1, maxX = 0, minY = 1, maxY = 0;
    for (const p of landmarks) {
        if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
    }
    const faceX = minX * vw, faceY = minY * vh;
    const faceW = (maxX - minX) * vw, faceH = (maxY - minY) * vh;
    if (faceW < 30 || faceH < 30) return null;

    // ── 1. Moiré: row-SAD coefficient of variation ─────────────────────────
    _RT_CTX.drawImage(video, faceX, faceY, faceW, faceH, 0, 0, 64, 64);
    const fp = _RT_CTX.getImageData(0, 0, 64, 64).data;
    const gray = new Uint8Array(4096);
    for (let i = 0; i < 4096; i++) {
        gray[i] = (77 * fp[i*4] + 150 * fp[i*4+1] + 29 * fp[i*4+2]) >> 8;
    }

    const rowSAD = new Float32Array(64);
    for (let r = 0; r < 64; r++) {
        let s = 0;
        for (let c = 0; c < 63; c++) s += Math.abs(gray[r*64+c+1] - gray[r*64+c]);
        rowSAD[r] = s / 63;
    }
    const sadMean = rowSAD.reduce((a, b) => a + b, 0) / 64;
    const sadStd  = Math.sqrt(rowSAD.reduce((s, v) => s + (v - sadMean) ** 2, 0) / 64);
    const sadCV   = sadMean > RT_MOIRE_SAD_MIN ? sadStd / sadMean : 1.0;
    const moireAlert = sadCV < RT_MOIRE_CV_MAX && sadMean > RT_MOIRE_SAD_MIN;

    // ── 2. Edge detection: straight line at face-border zone ──────────────
    const pad  = 0.18;
    const expX = Math.max(0, faceX - faceW * pad);
    const expY = Math.max(0, faceY - faceH * pad);
    const expW = Math.min(vw - expX, faceW * (1 + 2 * pad));
    const expH = Math.min(vh - expY, faceH * (1 + 2 * pad));

    _RT_CTX.drawImage(video, expX, expY, expW, expH, 0, 0, 64, 64);
    const ep = _RT_CTX.getImageData(0, 0, 64, 64).data;
    const ge = new Int16Array(4096);
    for (let i = 0; i < 4096; i++) {
        ge[i] = (77 * ep[i*4] + 150 * ep[i*4+1] + 29 * ep[i*4+2]) >> 8;
    }

    const colSum = new Float32Array(64);
    const rowSum = new Float32Array(64);
    for (let r = 1; r < 63; r++) {
        for (let c = 1; c < 63; c++) {
            // Sobel X (horizontal gradient) → accumulates onto rowSum
            const sx = Math.abs(
                ge[(r-1)*64+c+1] + 2*ge[r*64+c+1] + ge[(r+1)*64+c+1] -
                ge[(r-1)*64+c-1] - 2*ge[r*64+c-1] - ge[(r+1)*64+c-1]
            );
            // Sobel Y (vertical gradient) → accumulates onto colSum
            const sy = Math.abs(
                ge[(r+1)*64+c-1] + 2*ge[(r+1)*64+c] + ge[(r+1)*64+c+1] -
                ge[(r-1)*64+c-1] - 2*ge[(r-1)*64+c] - ge[(r-1)*64+c+1]
            );
            colSum[c] += sy;
            rowSum[r] += sx;
        }
    }

    const avgCol = colSum.reduce((a, b) => a + b, 0) / 64;
    const avgRow = rowSum.reduce((a, b) => a + b, 0) / 64;
    let edgeAlert = false;
    if (avgCol > 10 || avgRow > 10) {
        let maxColVal = 0, maxColIdx = 0, maxRowVal = 0, maxRowIdx = 0;
        for (let i = 0; i < 64; i++) {
            if (colSum[i] > maxColVal) { maxColVal = colSum[i]; maxColIdx = i; }
            if (rowSum[i] > maxRowVal) { maxRowVal = rowSum[i]; maxRowIdx = i; }
        }
        // Border zone: columns 0-19 or 44-63, rows 0-19 or 44-63
        const colAtBorder = maxColIdx < 20 || maxColIdx > 43;
        const rowAtBorder = maxRowIdx < 20 || maxRowIdx > 43;
        edgeAlert = (avgCol > 10 && maxColVal > RT_EDGE_RATIO * avgCol && colAtBorder) ||
                    (avgRow > 10 && maxRowVal > RT_EDGE_RATIO * avgRow && rowAtBorder);
    }

    // ── Consecutive counters ──────────────────────────────────────────────
    _rtMoireConsec = moireAlert ? _rtMoireConsec + 1 : 0;
    _rtEdgeConsec  = edgeAlert  ? _rtEdgeConsec  + 1 : 0;

    const moireBlocked = _rtMoireConsec >= RT_CONSEC_THRESH;
    const edgeBlocked  = _rtEdgeConsec  >= RT_CONSEC_THRESH;
    const blocked = moireBlocked || edgeBlocked;

    return {
        moireAlert, edgeAlert, blocked,
        reason: moireBlocked
            ? 'ตรวจพบหน้าจอ (Moiré) — กรุณาใช้ใบหน้าจริงต่อหน้ากล้อง'
            : edgeBlocked
            ? 'ตรวจพบขอบวัตถุแปลกปลอม — กรุณานำโทรศัพท์/กระดาษออก'
            : null,
        sadCV: +sadCV.toFixed(3),
        sadMean: +sadMean.toFixed(1),
    };
}

// ─── EAR samples collected during Step 4 capture ─────────────────────────────
let earSamplesDuringCapture = [];

function _computeEAR(lm) {
    // Right eye landmarks: 33,160,158,133,153,144 — Left eye: 362,385,387,263,373,380
    const eye = (p1, p2, p3, p4, p5, p6) => {
        const d = (a, b) => Math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y);
        return (d(p2, p6) + d(p3, p5)) / (2 * d(p1, p4));
    };
    return (eye(33, 160, 158, 133, 153, 144) + eye(362, 385, 387, 263, 373, 380)) / 2;
}

function _stdDev(arr) {
    if (arr.length < 2) return 0;
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    return Math.sqrt(arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length);
}

// ─── Step 4 spoof fail cap ────────────────────────────────────────────────────
let step4SpoofFailConsecutive = 0;  // resets on pass
let step4SpoofFailTotal       = 0;  // never resets except fullRestart/restartCapture
const MAX_STEP4_SPOOF_CONSEC  = 3;
const MAX_STEP4_SPOOF_TOTAL   = 5;

// ─── Step 4 timeout ───────────────────────────────────────────────────────────
const STEP4_TIMEOUT_MS = 120000;   // 2 minutes
let step4Timer = null;

// ─── Enrollment session ID (for spoof_check rate limiting) ───────────────────
let enrollmentSessionId = null;

function _getSessionId() {
    if (!enrollmentSessionId) {
        enrollmentSessionId = crypto.randomUUID
            ? crypto.randomUUID()
            : Date.now().toString(36) + Math.random().toString(36).slice(2);
    }
    return enrollmentSessionId;
}

// ─── Debug logging ────────────────────────────────────────────────────────────
const DEBUG = false;
function _log(...args)  { if (DEBUG) console.log('[SmartCheck]', ...args); }
function _warn(...args) { if (DEBUG) console.warn('[SmartCheck]', ...args); }

// ─── T16: Step Transition Modal ───────────────────────────────────────────────
let _modalResolve = null;

function _showStepModal(icon, title, desc, btnText) {
    return new Promise(resolve => {
        _modalResolve = resolve;
        document.getElementById('stepModalIcon').textContent  = icon;
        document.getElementById('stepModalTitle').textContent = title;
        document.getElementById('stepModalDesc').textContent  = desc;
        document.getElementById('stepModalBtn').textContent   = btnText || 'เข้าใจแล้ว เริ่มเลย →';
        const modal = document.getElementById('stepModal');
        modal.style.display = 'flex';
    });
}

function _dismissStepModal() {
    document.getElementById('stepModal').style.display = 'none';
    if (_modalResolve) { _modalResolve(); _modalResolve = null; }
}

// ─── Blink micro-check constants (Step 2 pre-calibration) ────────────────────
const BLINK_DROP_RATIO    = 0.70;   // EAR must drop below openEAR × 0.70 to count as closing
const BLINK_RECOVER_RATIO = 0.65;   // EAR must recover above openEAR × 0.65 to complete cycle
const BLINK_TIMEOUT_MS    = 10000;  // 10 s to complete one blink
const MAX_BLINK_ATTEMPTS  = 3;      // hard block after 3 timeouts
const EAR_OPEN_THRESHOLD  = 0.15;   // below this = eyes not fully open (skip as baseline ref)
let blinkAttempts = 0;
// Eye landmark indices (MediaPipe Face Mesh)
const LEFT_EYE  = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE = [362, 385, 387, 263, 373, 380];

// ─── Gate check thresholds ────────────────────
// CHECK 1: Face centering (normalized 0-1)
const GATE_CENTER_X_MIN       = 0.25;   // face cx must be within 25%–75% of frame width
const GATE_CENTER_X_MAX       = 0.75;
const GATE_CENTER_Y_MIN       = 0.20;   // face cy must be within 20%–80% of frame height
const GATE_CENTER_Y_MAX       = 0.80;
// CHECK 2: Neutral expression (ratios relative to face dimensions)
const GATE_MOUTH_OPEN_RATIO   = 0.045;  // mouth gap / face height — reject if above
const GATE_SMILE_RATIO        = 0.44;   // mouth width / face width  — reject if above
const GATE_BROW_RAISE_RATIO   = 0.085;  // avg brow-to-eye dist / face height — reject if above

// ─────────────────────────────────────────────
// Step navigation
// ─────────────────────────────────────────────
function goToStep(n) {
    document.querySelectorAll('.step').forEach((el, i) => {
        el.classList.toggle('active', i + 1 === n);
    });
    for (let i = 1; i <= 4; i++) {
        const dot = document.getElementById('dot' + i);
        if (dot) {
            dot.classList.toggle('active', i <= n);
            dot.classList.toggle('done', i < n);
        }
    }
    currentStep = n;
}

// ─────────────────────────────────────────────
// Step 1 — Consent
// ─────────────────────────────────────────────
document.getElementById('consentCheck').addEventListener('change', function () {
    document.getElementById('btnConsent').disabled = !this.checked;
});

// ─── CSRF + device helpers ────────────────────────────────────────────────────
function _csrfToken() {
    const el = document.querySelector('meta[name="csrf-token"]');
    if (!el) { _warn('CSRF meta tag not found'); return ''; }
    return el.content || '';
}

function _deviceFingerprint() {
    let fp = localStorage.getItem('sc_device_fp');
    if (!fp) {
        fp = (crypto.randomUUID ? crypto.randomUUID()
              : Date.now().toString(36) + Math.random().toString(36).slice(2));
        localStorage.setItem('sc_device_fp', fp);
    }
    return fp;
}

// ─── Spoof check helpers ──────────────────────────────────────────────────────

function _captureFrameFromVideo(videoEl) {
    if (!videoEl || videoEl.readyState < 2 || !videoEl.videoWidth) return null;
    const c = document.createElement('canvas');
    c.width  = videoEl.videoWidth;
    c.height = videoEl.videoHeight;
    c.getContext('2d').drawImage(videoEl, 0, 0);
    return c.toDataURL('image/jpeg', 0.82);
}

// Fail-close: retry once; if both fail → return blocked result (never silently pass)
async function _callSpoofCheckSafe(imageB64) {
    const BACKOFF_MS = [2000, 4000, 8000];  // exponential backoff delays
    for (let attempt = 0; attempt < 2; attempt++) {
        try {
            const res = await fetch(ENROLL_CONFIG.spoofCheckUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-Token': _csrfToken(),
                },
                body: JSON.stringify({ image: imageB64, session_id: _getSessionId() }),
            });

            // 429 — rate limited: NOT a spoof fail, return _rateLimited so caller
            // can show a friendly wait message and retry after backoff
            if (res.status === 429) {
                let retryAfterSec = null;
                try {
                    const body = await res.json();
                    retryAfterSec = body.retry_after ? parseInt(body.retry_after) : null;
                } catch (_) {}
                const waitMs = retryAfterSec ? retryAfterSec * 1000
                                             : BACKOFF_MS[attempt] ?? 8000;
                _warn(`spoof_check rate limited — waiting ${waitMs}ms before retry`);
                if (attempt === 0) {
                    await new Promise(r => setTimeout(r, waitMs));
                    continue;  // retry once after backoff
                }
                return { is_real: false, confidence: 0,
                         message: `ระบบกำลังประมวลผล กรุณารอสักครู่`,
                         _networkError: true, _rateLimited: true };
            }

            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return await res.json();
        } catch (e) {
            _warn(`spoof_check attempt ${attempt + 1} failed:`, e);
            if (attempt === 0) await new Promise(r => setTimeout(r, 1000));
        }
    }
    return { is_real: false, confidence: 0,
             message: 'ไม่สามารถตรวจสอบได้ กรุณาลองใหม่อีกครั้ง', _networkError: true };
}

// Stop every active stream + camera; safe to call multiple times
function _stopAllStreams() {
    [calibStream, captureStream, verifyStream, livenessStream].forEach(s => {
        if (s) { try { s.getTracks().forEach(t => t.stop()); } catch(e) {} }
    });
    calibStream = captureStream = verifyStream = livenessStream = null;
    _stopStepCamera();
}

window.addEventListener('beforeunload', _stopAllStreams);

function _setSpoofLabel(labelId, isReal, confidence) {
    const el = document.getElementById(labelId);
    if (!el) return;
    console.log(`[spoof] ${labelId}: ${isReal ? 'REAL' : 'SPOOF'} ${confidence.toFixed(3)}`);
    el.textContent       = isReal ? 'REAL ✓' : '⚠️ SPOOF';
    el.style.background  = isReal ? 'rgba(34,197,94,0.85)' : 'rgba(239,68,68,0.85)';
    el.style.color       = '#fff';
    el.style.display     = 'block';
}

function _clearSpoofLabel(labelId) {
    const el = document.getElementById(labelId);
    if (el) el.style.display = 'none';
}

let _spoofWarnTimer = null;
function _showSpoofWarn() {
    const toast = document.getElementById('spoofWarnToast');
    if (!toast) return;
    // Reset animation
    toast.style.animation = 'none';
    toast.offsetHeight;   // reflow
    toast.style.animation = '';
    toast.style.display = 'block';
    if (_spoofWarnTimer) clearTimeout(_spoofWarnTimer);
    _spoofWarnTimer = setTimeout(() => { toast.style.display = 'none'; }, 4000);
}

// ─── Face Continuity ทำที่ server แล้ว (/api/enroll + /api/self_verify) ───────
// Client ไม่เก็บหรือตรวจ embedding อีกต่อไป

async function goToCalibration() {
    const btn = document.getElementById('btnConsent');
    if (btn) btn.disabled = true;   // prevent double-click during async consent fetch
    // A5: record PDPA consent server-side before proceeding
    try {
        await fetch(ENROLL_CONFIG.consentUrl, {
            method: 'POST',
            headers: { 'X-CSRF-Token': _csrfToken() },
        });
    } catch (e) { /* non-fatal — server will reject enroll if missing */ }

    // ไม่แสดง modal ซ้อนอีกชั้น — ผู้ใช้กด consent ไปแล้ว ไป Step 2 ทันที
    blinkAttempts      = 0;
    enrollmentSessionId = null;   // new session ID for spoof_check rate limiting
    goToStep(2);
    startCamera('videoCalib', stream => { calibStream = stream; });
}

// ─────────────────────────────────────────────
// Camera helpers
// ─────────────────────────────────────────────
async function startCamera(videoId, onStream) {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: 640, height: 480 }
        });
        const video = document.getElementById(videoId);
        video.srcObject = stream;
        if (onStream) onStream(stream);
    } catch (e) {
        alert('ไม่สามารถเปิดกล้องได้: ' + e.message);
    }
}

function stopStream(stream) {
    if (stream) stream.getTracks().forEach(t => t.stop());
}

// ─────────────────────────────────────────────
// EAR calculation
// ─────────────────────────────────────────────
function dist(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
}

function calcEAR(landmarks, indices) {
    const [p1, p2, p3, p4, p5, p6] = indices.map(i => landmarks[i]);
    return (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4));
}

// ─────────────────────────────────────────────
// Step 2 — EAR Calibration (with blink micro-check gate)
// ─────────────────────────────────────────────

function startCalibration() {
    if (calibrating) return;
    calibrating = true;

    document.getElementById('btnStartCalib').disabled = true;
    document.getElementById('btnRetryBlink').style.display  = 'none';
    document.getElementById('btnBlinkGiveUp').style.display = 'none';
    document.getElementById('calibProgressWrap').style.display = 'block';
    document.getElementById('faceGuideCalib').classList.remove('ok', 'fail');

    _runBlinkCheck();
}

// ── Phase A: wait for one blink cycle to confirm live face ───────────────────
function _runBlinkCheck() {
    const video       = document.getElementById('videoCalib');
    const canvas      = document.getElementById('canvasCalib');
    const guide       = document.getElementById('faceGuideCalib');
    const status      = document.getElementById('calibStatus');
    const progressBar = document.getElementById('calibProgressBar');

    status.textContent           = 'กรุณากะพริบตา 1 ครั้ง';
    progressBar.style.background = '#4f46e5';
    progressBar.style.width      = '100%';  // drains to 0 over 10 s

    let openEAR    = null;
    let blinkPhase = 'waiting_open';  // → waiting_drop → waiting_recover
    let blinkDone  = false;
    const startTime = performance.now();

    const faceMesh = _getSharedFM();

    faceMesh.onResults(results => {
        if (blinkDone) return;

        canvas.width  = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);

        const elapsed   = performance.now() - startTime;
        const remaining = Math.max(0, BLINK_TIMEOUT_MS - elapsed);
        progressBar.style.width = ((remaining / BLINK_TIMEOUT_MS) * 100) + '%';

        if (elapsed >= BLINK_TIMEOUT_MS) {
            blinkDone = true;
            _stopStepCamera();
            _onBlinkTimeout();
            return;
        }

        if (!results.multiFaceLandmarks?.length) {
            guide.classList.remove('ok', 'fail');
            // Don't reset mid-blink — face briefly disappears during eye closure
            // Only reset if we haven't started tracking yet
            if (blinkPhase === 'waiting_open') {
                openEAR = null;
                status.textContent = `ไม่พบใบหน้า — จัดหน้าให้อยู่ในกรอบ (${Math.ceil(remaining / 1000)}s)`;
            }
            return;
        }

        guide.classList.remove('fail');
        guide.classList.add('ok');
        const lm  = results.multiFaceLandmarks[0];
        const ctx = canvas.getContext('2d');
        drawFaceFeatures(ctx, lm, canvas.width, canvas.height, 'rgba(255,255,255,0.7)');

        const ear = (calcEAR(lm, LEFT_EYE) + calcEAR(lm, RIGHT_EYE)) / 2;

        if (blinkPhase === 'waiting_open') {
            if (ear >= EAR_OPEN_THRESHOLD) {
                openEAR    = ear;
                blinkPhase = 'waiting_drop';
            }
            status.textContent = `กรุณากะพริบตา 1 ครั้ง (${Math.ceil(remaining / 1000)}s)`;

        } else if (blinkPhase === 'waiting_drop') {
            if (ear > openEAR) openEAR = ear;   // track highest open EAR seen
            if (ear < openEAR * BLINK_DROP_RATIO) {
                blinkPhase = 'waiting_recover';
                status.textContent = '...';
            } else {
                status.textContent = `กรุณากะพริบตา 1 ครั้ง (${Math.ceil(remaining / 1000)}s)`;
            }

        } else if (blinkPhase === 'waiting_recover') {
            if (ear >= openEAR * BLINK_RECOVER_RATIO) {
                blinkDone = true;
                const blinkFrame = _captureFrameFromVideo(video);  // capture before stream stops
                _stopStepCamera();
                _onBlinkSuccess(blinkFrame);
            }
        }
    });

    const cam = new Camera(video, {
        onFrame: async () => { if (!blinkDone) await faceMesh.send({ image: video }); },
        width: 640, height: 480,
    });
    _stepCamera = cam;
    cam.start();
}

async function _onBlinkSuccess(blinkFrame) {
    const status      = document.getElementById('calibStatus');
    const guide       = document.getElementById('faceGuideCalib');
    const progressBar = document.getElementById('calibProgressBar');

    progressBar.style.background = '#22c55e';
    progressBar.style.width      = '100%';
    guide.classList.add('ok');
    status.textContent = '✓ กะพริบตาสำเร็จ — กำลังตรวจสอบใบหน้า...';

    // ── Inline spoof check (Step 2) ──────────────────────────────────────────
    {
        const frame = blinkFrame;
        if (!frame) {
            calibrating = false;
            status.textContent = 'กล้องยังไม่พร้อม — กรุณาลองใหม่';
            document.getElementById('btnRetryBlink').style.display = 'block';
            return;
        }
        const r = await _callSpoofCheckSafe(frame);
        _setSpoofLabel('spoofLabelCalib', r.is_real, r.confidence);
        if (!r.is_real) {
            _showSpoofWarn();
            calibrating = false;
            progressBar.style.background = '#ef4444';
            progressBar.style.width      = '0%';
            guide.classList.remove('ok');
            guide.classList.add('fail');
            status.textContent = r.message || 'ตรวจพบภาพปลอม กรุณาใช้ใบหน้าจริง';
            blinkAttempts++;
            setTimeout(() => _clearSpoofLabel('spoofLabelCalib'), 4000);
            if (blinkAttempts >= MAX_BLINK_ATTEMPTS) {
                document.getElementById('btnStartCalib').style.display  = 'none';
                document.getElementById('btnRetryBlink').style.display  = 'none';
                document.getElementById('btnBlinkGiveUp').style.display = 'block';
            } else {
                document.getElementById('btnRetryBlink').style.display = 'block';
            }
            return;
        }
        // embedding ถูกเก็บที่ server แล้ว (session["liveness_embeddings"])
        setTimeout(() => _clearSpoofLabel('spoofLabelCalib'), 2000);
    }
    // ─────────────────────────────────────────────────────────────────────────

    setTimeout(() => {
        progressBar.style.background = '#4f46e5';
        progressBar.style.width      = '0%';
        status.textContent = 'กำลังวัดค่า Baseline EAR... เปิดตาตามปกติ';
        calibEARValues = [];
        _startEARMeasurement();
    }, 300);
}

function _onBlinkTimeout() {
    blinkAttempts++;
    calibrating = false;

    const status      = document.getElementById('calibStatus');
    const guide       = document.getElementById('faceGuideCalib');
    const progressBar = document.getElementById('calibProgressBar');

    progressBar.style.width = '0%';
    guide.classList.remove('ok');
    guide.classList.add('fail');

    if (blinkAttempts >= MAX_BLINK_ATTEMPTS) {
        status.textContent = 'ไม่สามารถยืนยันได้ว่าเป็นใบหน้าจริง กรุณาติดต่อผู้ดูแลระบบ';
        document.getElementById('btnStartCalib').style.display  = 'none';
        document.getElementById('btnRetryBlink').style.display  = 'none';
        document.getElementById('btnBlinkGiveUp').style.display = 'block';
    } else {
        const left = MAX_BLINK_ATTEMPTS - blinkAttempts;
        status.textContent = `ไม่พบการกะพริบตา กรุณาใช้ใบหน้าจริงต่อหน้ากล้อง (เหลืออีก ${left} ครั้ง)`;
        document.getElementById('btnRetryBlink').style.display = 'block';
    }
}

// ── Phase B: measure baseline EAR for 5 s ────────────────────────────────────
function _startEARMeasurement() {
    const video       = document.getElementById('videoCalib');
    const canvas      = document.getElementById('canvasCalib');
    const guide       = document.getElementById('faceGuideCalib');
    const progressBar = document.getElementById('calibProgressBar');

    const DURATION_MS         = 2000;
    const EAR_BLINK_THRESHOLD = 0.15;   // C1: discard frames where eyes are closing
    let startTime = null;
    let calibDone = false;

    const faceMesh = _getSharedFM();

    faceMesh.onResults(results => {
        if (calibDone) return;
        canvas.width  = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);

        if (results.multiFaceLandmarks?.length > 0) {
            guide.classList.add('ok');
            const lm  = results.multiFaceLandmarks[0];
            drawFaceFeatures(canvas.getContext('2d'), lm, canvas.width, canvas.height, 'rgba(255,255,255,0.7)');
            const ear = (calcEAR(lm, LEFT_EYE) + calcEAR(lm, RIGHT_EYE)) / 2;

            if (startTime === null) startTime = performance.now();
            const elapsed = performance.now() - startTime;
            progressBar.style.width = Math.min((elapsed / DURATION_MS) * 100, 100) + '%';

            if (ear >= EAR_BLINK_THRESHOLD) calibEARValues.push(ear);
            document.getElementById('calibStatus').textContent =
                `EAR: ${ear.toFixed(4)} — กำลังวัด... (${Math.ceil((DURATION_MS - elapsed) / 1000)}s)`;

            if (elapsed >= DURATION_MS) {
                calibDone = true;
                _stopStepCamera();
                finishCalibration();
            }
        } else {
            guide.classList.remove('ok');
            document.getElementById('calibStatus').textContent = 'ไม่พบใบหน้า — จัดหน้าให้อยู่ในกรอบ';
            startTime = null;   // reset timer if face lost
        }
    });

    const cam = new Camera(video, {
        onFrame: async () => { if (!calibDone) await faceMesh.send({ image: video }); },
        width: 640, height: 480,
    });
    _stepCamera = cam;
    cam.start();
}

async function finishCalibration() {
    calibrating = false;
    if (calibEARValues.length === 0) {
        document.getElementById('calibStatus').textContent = 'วัดไม่สำเร็จ กรุณาลองใหม่';
        document.getElementById('btnRetryBlink').style.display = 'block';
        calibEARValues = [];
        return;
    }
    // C1: need at least 10 usable (non-blink) frames for a reliable baseline
    if (calibEARValues.length < 10) {
        document.getElementById('calibStatus').textContent = 'กะพริบตาบ่อยเกินไป — กรุณามองกล้องตาเปิดค้างไว้ แล้วกดลองใหม่';
        document.getElementById('btnRetryBlink').style.display = 'block';
        calibEARValues = [];
        return;
    }
    const sorted = [...calibEARValues].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    baselineEAR = median;

    document.getElementById('calibProgressBar').style.width = '100%';
    document.getElementById('calibStatus').textContent =
        `✓ Baseline EAR = ${median.toFixed(4)} — สำเร็จ!`;

    stopStream(calibStream);

    await _showStepModal(
        '🎭',
        'ยืนยันตัวตน',
        'ระบบจะสุ่มท่าทาง 2 อย่าง เช่น กะพริบตา, ยิ้ม, หันซ้าย-ขวา — ทำตามลำดับเพื่อยืนยันว่าเป็นคนจริง',
        'เข้าใจแล้ว เริ่มเลย →'
    );

    goToStep(3);
    startLivenessChallenge();
}

// ─────────────────────────────────────────────
// Step 3 — Interactive Challenge (2 actions)
// ─────────────────────────────────────────────
let livenessStream = null;

const CHALLENGE_ACTION_LABELS = {
    blink:          'กะพริบตา',
    smile:          'ยิ้มให้กว้าง',
    turn_left:      'หันหน้าไปทางซ้าย',
    turn_right:     'หันหน้าไปทางขวา',
    nod:            'พยักหน้าขึ้น-ลง',
    raise_eyebrows: 'ยกคิ้วขึ้น',
};

function _buildChallengePills(actions, currentIdx) {
    const wrap = document.getElementById('challengeSteps');
    wrap.innerHTML = '';
    actions.forEach((a, i) => {
        const pill = document.createElement('div');
        pill.style.cssText = `
            padding:0.3rem 0.85rem;border-radius:999px;font-size:0.8rem;font-weight:600;
            background:${i < currentIdx ? '#22c55e' : i === currentIdx ? '#4f46e5' : '#e2e8f0'};
            color:${i <= currentIdx ? '#fff' : '#94a3b8'};transition:background 0.3s`;
        pill.textContent = `${i + 1}. ${CHALLENGE_ACTION_LABELS[a] || a}`;
        wrap.appendChild(pill);
    });
}

async function startLivenessChallenge() {
    const actions = randomChallengeActions(2);

    _buildChallengePills(actions, 0);
    document.getElementById('challengeInstruction').style.display = 'none';  // hidden until pre-spoof check passes
    document.getElementById('livenessStatus').textContent = 'กำลังเปิดกล้อง...';

    try {
        livenessStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: 640, height: 480 }
        });
        document.getElementById('videoLiveness').srcObject = livenessStream;
    } catch (e) {
        alert('ไม่สามารถเปิดกล้องได้: ' + e.message);
        return;
    }

    // ── Spoof check before challenge (Step 3, check #1) ─────────────────────
    // Wait for video to actually render frames (canplay event or 3s fallback)
    {
        const videoLiv = document.getElementById('videoLiveness');
        await new Promise(resolve => {
            if (videoLiv.readyState >= 3) { setTimeout(resolve, 200); return; }
            const onReady = () => { videoLiv.removeEventListener('canplay', onReady); setTimeout(resolve, 300); };
            videoLiv.addEventListener('canplay', onReady);
            setTimeout(() => { videoLiv.removeEventListener('canplay', onReady); resolve(); }, 3000);
        });
    }
    document.getElementById('livenessStatus').textContent = 'กำลังตรวจสอบใบหน้า...';
    {
        const videoLiv = document.getElementById('videoLiveness');
        const frame1   = _captureFrameFromVideo(videoLiv);
        if (!frame1) {
            if (livenessStream) livenessStream.getTracks().forEach(t => t.stop());
            document.getElementById('livenessStatus').textContent = 'กล้องยังไม่พร้อม — กรุณาลองใหม่';
            setTimeout(() => startLivenessChallenge(), 2000);
            return;
        }
        const sc1 = await _callSpoofCheckSafe(frame1);
        _setSpoofLabel('spoofLabelLiveness', sc1.is_real, sc1.confidence);
        setTimeout(() => _clearSpoofLabel('spoofLabelLiveness'), 2000);
        if (!sc1.is_real) {
            if (sc1._networkError) {
                // Server validation/network error — skip pre-check, proceed to challenge
                _warn('spoof_check pre-challenge network error — skipping');
                setTimeout(() => _clearSpoofLabel('spoofLabelLiveness'), 1500);
            } else {
                _showSpoofWarn();
                if (livenessStream) livenessStream.getTracks().forEach(t => t.stop());
                challengeAttempts++;
                document.getElementById('challengeInstruction').style.display = 'none';
                if (challengeAttempts >= MAX_CHALLENGE_ATTEMPTS) {
                    challengeAttempts = 0;
                    document.getElementById('livenessStatus').textContent =
                        'ลองเกินจำนวนครั้งที่กำหนด — กรุณารอ 30 วินาที';
                    setTimeout(() => startLivenessChallenge(), CHALLENGE_COOLDOWN_MS);
                } else {
                    document.getElementById('livenessStatus').textContent =
                        (sc1.message || 'ตรวจพบภาพปลอม') + ` — กรุณาลองใหม่ (${challengeAttempts}/${MAX_CHALLENGE_ATTEMPTS})`;
                    setTimeout(() => startLivenessChallenge(), 2000);
                }
                return;
            }
        }
        // embedding ถูกเก็บที่ server แล้ว
    }
    // ─────────────────────────────────────────────────────────────────────────

    // Show challenge instruction only AFTER pre-spoof check completes to avoid
    // user performing the gesture during the check and needing to repeat it
    document.getElementById('challengeInstruction').style.display = 'block';
    document.getElementById('challengeActionLabel').textContent = CHALLENGE_ACTION_LABELS[actions[0]] || actions[0];

    const detector = new InteractiveChallengeDetector(
        document.getElementById('videoLiveness'),
        document.getElementById('canvasLiveness'),
        baselineEAR,
        (actionIdx, _total, statusText) => {
            _buildChallengePills(actions, actionIdx);
            document.getElementById('challengeActionLabel').textContent =
                CHALLENGE_ACTION_LABELS[actions[actionIdx]] || actions[actionIdx];
            document.getElementById('livenessStatus').textContent = statusText;
        },
        { faceMesh: _getSharedFM({ refineLandmarks: true }) }
    );

    const result = await detector.run(actions, 8000);

    if (!result.pass) {
        if (livenessStream) livenessStream.getTracks().forEach(t => t.stop());
        challengeAttempts++;
        document.getElementById('challengeInstruction').style.display = 'none';

        // B3: enforce attempt cap with cooldown
        if (challengeAttempts >= MAX_CHALLENGE_ATTEMPTS) {
            challengeAttempts = 0;
            document.getElementById('livenessStatus').textContent =
                'ลองเกินจำนวนครั้งที่กำหนด — กรุณารอ 30 วินาที';
            setTimeout(() => {
                document.getElementById('livenessStatus').textContent = 'พร้อมลองอีกครั้ง';
                setTimeout(() => startLivenessChallenge(), 500);
            }, CHALLENGE_COOLDOWN_MS);
        } else {
            document.getElementById('livenessStatus').textContent =
                (result.error || 'ไม่ผ่าน') + ` — กรุณาลองใหม่ (${challengeAttempts}/${MAX_CHALLENGE_ATTEMPTS})`;
            setTimeout(() => startLivenessChallenge(), 2000);
        }
        return;
    }
    // Reset counter on success
    challengeAttempts = 0;

    // ── Spoof check after challenge (Step 3, check #2) ───────────────────────
    {
        const videoLiv = document.getElementById('videoLiveness');
        const frame2   = _captureFrameFromVideo(videoLiv);
        if (frame2) {
            const sc2 = await _callSpoofCheckSafe(frame2);
            _setSpoofLabel('spoofLabelLiveness', sc2.is_real, sc2.confidence);
            if (!sc2.is_real && !sc2._networkError) {
                // Hard spoof detected post-challenge — require full restart (face swap suspected)
                _showSpoofWarn();
                if (livenessStream) livenessStream.getTracks().forEach(t => t.stop());
                document.getElementById('livenessStatus').textContent =
                    'ตรวจพบการสลับใบหน้า — กรุณาเริ่มใหม่';
                setTimeout(() => fullRestart(), 2500);
                return;
            }
            // is_real=true or network error — continue to Step 4
            // embedding ถูกเก็บที่ server แล้ว
            setTimeout(() => _clearSpoofLabel('spoofLabelLiveness'), 1500);
        }
    }
    // ─────────────────────────────────────────────────────────────────────────

    if (livenessStream) livenessStream.getTracks().forEach(t => t.stop());

    await _showStepModal(
        '📸',
        'ถ่ายรูปใบหน้า 5 รูป',
        'ระบบจะถ่ายรูปอัตโนมัติ 5 ครั้ง — มองตรงกล้อง ทำหน้าปกติ อยู่นิ่งๆ ในที่ที่มีแสงเพียงพอ',
        'พร้อมถ่ายรูป →'
    );

    goToStep(4);
    startCaptureWithDetection();
}

// ─────────────────────────────────────────────
// Face Feature Drawing (ตา + ปาก)
// ─────────────────────────────────────────────
const EYE_LEFT_IDX  = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246];
const EYE_RIGHT_IDX = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398];
const MOUTH_IDX     = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146];

function drawFaceFeatures(ctx, lm, w, h, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth   = 1.5;
    ctx.setLineDash([4, 3]);

    [[EYE_LEFT_IDX], [EYE_RIGHT_IDX], [MOUTH_IDX]].forEach(([indices]) => {
        ctx.beginPath();
        indices.forEach((idx, i) => {
            const x = lm[idx].x * w;
            const y = lm[idx].y * h;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.stroke();
    });

    ctx.setLineDash([]);
}

// ─────────────────────────────────────────────
// Step 4 — Auto-Capture (frontal, 5 frames)
// ─────────────────────────────────────────────

function _checkFrontal(lm) {
    const nose     = lm[1];
    const leftEar  = lm[234];
    const rightEar = lm[454];
    const forehead = lm[10];
    const chin     = lm[152];
    const faceW = rightEar.x - leftEar.x;
    const faceH = chin.y - forehead.y;
    if (faceW <= 0 || faceH <= 0) return false;
    const yaw   = ((nose.x - leftEar.x) / faceW - 0.5) * 90;
    const pitch = ((nose.y - forehead.y) / faceH - 0.5) * 90;
    return Math.abs(yaw) <= 15 && Math.abs(pitch) <= 15;
}

// ─── Gate helpers ─────────────────────────────────────────────────────────────

/**
 * CHECK 1 — Face centering.
 * Uses outer face landmarks (lm[234]=left ear, lm[454]=right ear,
 * lm[10]=forehead, lm[152]=chin) to compute the face centre in
 * normalised [0,1] coords and checks it falls within the allowed zone.
 */
function _checkCentering(lm) {
    const cx = (lm[234].x + lm[454].x) / 2;
    const cy = (lm[10].y  + lm[152].y) / 2;
    const ok = cx >= GATE_CENTER_X_MIN && cx <= GATE_CENTER_X_MAX
            && cy >= GATE_CENTER_Y_MIN && cy <= GATE_CENTER_Y_MAX;
    return { ok, cx, cy };
}

/**
 * CHECK 2 — Neutral expression gate.
 * All distances are normalised by face height/width so the check is
 * resolution-independent.  Returns { ok, reason }.
 *
 * Landmarks used:
 *   lm[13]  upper inner lip   lm[14]  lower inner lip
 *   lm[61]  left mouth corner lm[291] right mouth corner
 *   lm[65]  left brow inner   lm[159] left eye top
 *   lm[295] right brow inner  lm[386] right eye top
 */
function _checkNeutral(lm) {
    function d(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

    const faceH = d(lm[10], lm[152]);
    const faceW = d(lm[234], lm[454]);
    if (faceH === 0 || faceW === 0) return { ok: true, reason: null };

    // Mouth open: gap between inner upper/lower lip
    const mouthGap = d(lm[13], lm[14]) / faceH;
    if (mouthGap > GATE_MOUTH_OPEN_RATIO)
        return { ok: false, reason: 'กรุณาทำหน้าปกติ ไม่อ้าปาก' };

    // Smile / grimace: mouth corner span vs face width
    const mouthW = d(lm[61], lm[291]) / faceW;
    if (mouthW > GATE_SMILE_RATIO)
        return { ok: false, reason: 'กรุณาทำหน้าปกติ ไม่อ้าปาก' };

    // Eyebrows raised: inner brow to eye-top distance vs face height
    const leftBrow  = d(lm[65],  lm[159]) / faceH;
    const rightBrow = d(lm[295], lm[386]) / faceH;
    if ((leftBrow + rightBrow) / 2 > GATE_BROW_RAISE_RATIO)
        return { ok: false, reason: 'กรุณาทำหน้าปกติ ไม่ยกคิ้ว' };

    return { ok: true, reason: null };
}

// Laplacian variance blur detection — C2: normalize to 640×480 first
// so the threshold is resolution-independent (works same on 480p and 1080p)
function _checkBlur(videoEl) {
    const c   = document.getElementById('captureCanvas');
    c.width   = 640;
    c.height  = 480;
    const ctx = c.getContext('2d');
    ctx.drawImage(videoEl, 0, 0, 640, 480);
    const { data, width, height } = ctx.getImageData(0, 0, c.width, c.height);
    const kernel = [0, 1, 0, 1, -4, 1, 0, 1, 0];
    let sum = 0, sumSq = 0;
    const count = (width - 2) * (height - 2);
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            let val = 0;
            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    const idx = ((y + ky) * width + (x + kx)) * 4;
                    val += (0.299*data[idx] + 0.587*data[idx+1] + 0.114*data[idx+2])
                           * kernel[(ky+1)*3 + (kx+1)];
                }
            }
            sum += val; sumSq += val * val;
        }
    }
    const mean = sum / count;
    return (sumSq / count) - (mean * mean);
}

function _updateCaptureDots() {
    const wrap = document.getElementById('captureDots');
    wrap.innerHTML = '';
    for (let i = 0; i < TOTAL_FRAMES; i++) {
        const d = document.createElement('div');
        d.style.cssText = `width:10px;height:10px;border-radius:50%;background:${
            i < capturedImages.length ? '#22c55e' : '#e2e8f0'}`;
        wrap.appendChild(d);
    }
    document.getElementById('captureProgress').textContent = `${capturedImages.length}/${TOTAL_FRAMES}`;
}

function _flashCapture() {
    const el = document.getElementById('captureFlash');
    el.style.opacity = '0.6';
    setTimeout(() => { el.style.opacity = '0'; }, 150);
}

function _addThumbnail(b64) {
    const img = document.createElement('img');
    img.src = b64;
    img.style.cssText = 'width:52px;height:52px;object-fit:cover;border-radius:6px;border:2px solid #22c55e';
    document.getElementById('thumbnailRow').appendChild(img);
}

function startCaptureWithDetection() {
    const video  = document.getElementById('videoCapture');
    const canvas = document.getElementById('canvasCaptureDetect');
    const guide  = document.getElementById('faceGuideCapture');
    const status = document.getElementById('captureStatus');

    _updateCaptureDots();
    lastCaptureTime = 0;
    capturePaused   = false;

    faceMeshCapture = _getSharedFM({ minDetectionConfidence: 0.6, minTrackingConfidence: 0.6 });

    faceMeshCapture.onResults(async results => {
        if (capturePaused) return;
        canvas.width  = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx2 = canvas.getContext('2d');
        ctx2.clearRect(0, 0, canvas.width, canvas.height);

        const hasFace = results.multiFaceLandmarks?.length > 0;
        if (!hasFace) {
            guide.classList.remove('ok', 'fail');
            status.textContent = 'ไม่พบใบหน้า — จัดหน้าให้อยู่ในกรอบ';
            return;
        }

        const lm = results.multiFaceLandmarks[0];
        drawFaceFeatures(ctx2, lm, canvas.width, canvas.height, 'rgba(255,255,255,0.7)');

        // Collect EAR sample for temporal anti-spoof check
        earSamplesDuringCapture.push(_computeEAR(lm));

        // ── Real-time Moiré + edge analysis ───────────────────────────────
        const rtResult = _rtAnalyzeFrame(video, lm);
        if (rtResult) {
            if (rtResult.blocked) {
                guide.classList.remove('ok'); guide.classList.add('fail');
                status.textContent = rtResult.reason;
                _showSpoofWarn();
                step4SpoofFailConsecutive++;
                step4SpoofFailTotal++;
                if (step4SpoofFailConsecutive >= MAX_STEP4_SPOOF_CONSEC ||
                    step4SpoofFailTotal       >= MAX_STEP4_SPOOF_TOTAL) {
                    setTimeout(() => fullRestart(), 3000);
                } else {
                    setTimeout(() => _clearSpoofLabel('spoofLabelCapture'), 4000);
                    capturePaused = false;
                }
                return;
            }
            if (rtResult.moireAlert || rtResult.edgeAlert) {
                status.textContent = rtResult.moireAlert
                    ? '⚠️ ตรวจพบลักษณะหน้าจอ...'
                    : '⚠️ ตรวจพบขอบวัตถุแปลกปลอม...';
                return;
            }
        }

        // ── Inline helper: mark guide red and exit ──────────────────────────
        function _failGate(msg) {
            guide.classList.remove('ok');
            guide.classList.add('fail');
            status.textContent = msg;
        }

        // Face size check
        let minY = 1, maxY = 0;
        for (const p of lm) { if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y; }
        if (maxY - minY < 0.30) { _failGate('เข้าใกล้กล้องอีกหน่อย'); return; }

        // Frontal angle check (yaw/pitch ±15°)
        if (!_checkFrontal(lm)) { _failGate('กรุณามองตรงเข้ากล้อง'); return; }

        // CHECK 1 — Face centering
        const centering = _checkCentering(lm);
        if (!centering.ok) { _failGate('กรุณาจัดหน้าให้อยู่กึ่งกลางกล้อง'); return; }

        // CHECK 2 — Neutral expression
        const neutral = _checkNeutral(lm);
        if (!neutral.ok) { _failGate(neutral.reason); return; }

        // ── All checks passed — green border ────────────────────────────────
        guide.classList.remove('fail');
        guide.classList.add('ok');
        drawFaceFeatures(ctx2, lm, canvas.width, canvas.height, 'rgba(74,222,128,0.9)');

        const now = Date.now();
        if (now - lastCaptureTime < CAPTURE_INTERVAL_MS) {
            const remain = ((CAPTURE_INTERVAL_MS - (now - lastCaptureTime)) / 1000).toFixed(1);
            status.textContent = `✓ พบใบหน้า — ถ่ายถัดไปใน ${remain}s (${capturedImages.length}/${TOTAL_FRAMES})`;
            return;
        }

        // Blur check (Laplacian variance > 50)
        const blurScore = _checkBlur(video);
        if (blurScore < 20) { status.textContent = 'ภาพเบลอ — อยู่นิ่งๆ'; return; }

        // ─── Pre-capture: spoof + face continuity check ───────────────────
        lastCaptureTime = now;        // set now so cooldown prevents double-trigger
        capturePaused = true;         // pause loop during async check
        status.textContent = 'กำลังตรวจสอบ...';

        const snapB64 = _captureFrameFromVideo(video);
        if (!snapB64) {
            status.textContent = 'กล้องยังไม่พร้อม — กรุณารอสักครู่';
            capturePaused = false;
            return;
        }
        const sc = await _callSpoofCheckSafe(snapB64);
        _setSpoofLabel('spoofLabelCapture', sc.is_real, sc.confidence);

        if (!sc.is_real) {
            if (sc._networkError) {
                // Network / validation error — skip frame, don't penalise either counter
                _clearSpoofLabel('spoofLabelCapture');
                if (sc._rateLimited) {
                    // 429 rate limit — show friendly wait message, NOT a spoof fail
                    _warn('spoof_check rate limited — skipping frame');
                    status.textContent = sc.message || 'ระบบกำลังประมวลผล กรุณารอสักครู่';
                } else {
                    _warn('spoof_check network/validation error — skipping frame');
                }
                capturePaused = false;
                return;
            }
            // Hard spoof detected by model
            _showSpoofWarn();
            step4SpoofFailConsecutive++;
            step4SpoofFailTotal++;
            guide.classList.remove('ok'); guide.classList.add('fail');

            if (step4SpoofFailConsecutive >= MAX_STEP4_SPOOF_CONSEC) {
                status.textContent = 'ตรวจพบความผิดปกติต่อเนื่อง — กรุณาเริ่มใหม่';
                setTimeout(() => { _clearSpoofLabel('spoofLabelCapture'); fullRestart(); }, 3000);
                return;   // capturePaused stays true — fullRestart resets it
            }
            if (step4SpoofFailTotal >= MAX_STEP4_SPOOF_TOTAL) {
                status.textContent = 'ตรวจพบความผิดปกติหลายครั้ง — กรุณาเริ่มใหม่ในสภาพแวดล้อมที่มีแสงเพียงพอ';
                setTimeout(() => { _clearSpoofLabel('spoofLabelCapture'); fullRestart(); }, 3000);
                return;
            }

            const remainingConsec = MAX_STEP4_SPOOF_CONSEC - step4SpoofFailConsecutive;
            const remainingTotal  = MAX_STEP4_SPOOF_TOTAL  - step4SpoofFailTotal;
            const remaining = Math.min(remainingConsec, remainingTotal);
            status.textContent = `ตรวจพบความผิดปกติ — เหลือโอกาสอีก ${remaining} ครั้ง กรุณาใช้ใบหน้าจริง`;
            setTimeout(() => _clearSpoofLabel('spoofLabelCapture'), 4000);
            capturePaused = false;
            return;
        }

        // Face continuity ตรวจที่ server ตอน /api/enroll แล้ว
        step4SpoofFailConsecutive = 0;  // reset consecutive on pass; total stays
        setTimeout(() => _clearSpoofLabel('spoofLabelCapture'), 1500);
        capturePaused = false;
        // ─────────────────────────────────────────────────────────────────

        // ─── Capture frame ────────────────────────────────────────────────
        const cap = document.getElementById('captureCanvas');
        cap.width  = video.videoWidth;
        cap.height = video.videoHeight;
        cap.getContext('2d').drawImage(video, 0, 0);
        const imgB64 = cap.toDataURL('image/jpeg', 0.88);

        capturedImages.push(imgB64);
        _addThumbnail(imgB64);
        _updateCaptureDots();
        _flashCapture();
        status.textContent = `✓ บันทึกรูปที่ ${capturedImages.length}/${TOTAL_FRAMES}`;

        if (capturedImages.length >= TOTAL_FRAMES) {
            capturePaused = true;
            _stopStepCamera();
            faceMeshCapture = null;
            stopStream(captureStream);
            await _sendToEnroll();
        }
    });

    navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 640, height: 480 }
    }).then(stream => {
        captureStream = stream;
        video.srcObject = stream;
        captureCamera = new Camera(video, {
            onFrame: async () => { if (!capturePaused && faceMeshCapture) await faceMeshCapture.send({ image: video }); },
            width: 640, height: 480,
        });
        _stepCamera = captureCamera;
        captureCamera.start();
        status.textContent = 'มองตรงกล้อง อยู่นิ่ง ๆ — ระบบจะถ่ายอัตโนมัติ';

        // ── 2-minute timeout — restart capture if not done
        if (step4Timer) clearTimeout(step4Timer);
        step4Timer = setTimeout(() => {
            if (capturedImages.length < TOTAL_FRAMES) {
                capturePaused = true;
                _stopStepCamera();
                stopStream(captureStream);
                status.textContent = 'หมดเวลา — กรุณาลองใหม่';
                setTimeout(() => restartCapture(), 3000);
            }
        }, STEP4_TIMEOUT_MS);
    }).catch(e => alert('ไม่สามารถเปิดกล้องได้: ' + e.message));
}

// ─── B5: Double-submit guard ──────────────────────────────────────────────────
let _enrollSubmitting = false;

// ─── Send 5 frontal frames to /api/enroll ────────────────────────────────────
async function _sendToEnroll() {
    if (_enrollSubmitting) return;
    _enrollSubmitting = true;
    _showChecking('กำลังส่งข้อมูล...');
    _startProgress(_PROGRESS_MSGS_ENROLL);
    try {
        const res  = await fetch(ENROLL_CONFIG.enrollUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': _csrfToken(),
            },
            body: JSON.stringify({
                face_images:  capturedImages,
                baseline_ear: baselineEAR,
                ear_std:      _stdDev(earSamplesDuringCapture),
            }),
        });
        _finishProgress();
        const json = await res.json();
        _hideChecking();

        if (step4Timer) { clearTimeout(step4Timer); step4Timer = null; }

        if (json.status === 'need_more') {
            // Single outlier — remove bad frame, keep the rest + liveness embeddings intact
            for (const idx of [...json.removed_indices].sort((a, b) => b - a)) {
                capturedImages.splice(idx, 1);
            }
            document.getElementById('thumbnailRow').innerHTML = '';
            capturedImages.forEach(b => _addThumbnail(b));
            _updateCaptureDots();
            document.getElementById('captureStatus').textContent = json.message;
            setTimeout(() => restartCapture(), 2500);

        } else if (json.status === 'restart_capture') {
            // ≥2 outliers — reset captured frames, keep liveness embeddings
            document.getElementById('captureStatus').textContent = json.message;
            setTimeout(() => restartCapture(), 2500);

        } else if (json.status === 'pending_verify') {
            await _showStepModal(
                '✅',
                'ยืนยันตัวตนครั้งสุดท้าย',
                'ระบบจะถ่ายรูปอีก 1 รูป เพื่อเปรียบเทียบกับรูปที่ถ่ายไว้ มองตรงกล้องแล้วระบบจะถ่ายให้อัตโนมัติ',
                'พร้อมยืนยัน →'
            );
            _startSelfVerify();

        } else if (json.status === 'spoof_detected') {
            // Server ตรวจพบ spoof ใน capture frames
            _showSpoofWarn();
            const frameInfo = json.failed_frame ? ` (รูปที่ ${json.failed_frame})` : '';
            _showResult('error', (json.message || 'ตรวจพบภาพปลอม') + frameInfo);
            setTimeout(() => fullRestart(), 3000);

        } else if (json.status === 'continuity_fail') {
            // Server ตรวจพบว่า capture frames ไม่ตรงกับ liveness embeddings
            // ใช้ restartCapture() แทน fullRestart() — เก็บ baseline+liveness ไว้ แค่ถ่ายรูปใหม่
            _showResult('error', json.message || 'ใบหน้าไม่ตรงกัน กรุณาถ่ายรูปใหม่');
            setTimeout(() => restartCapture(), 3000);

        } else if (json.status === 'duplicate') {
            _showResult('error', json.message || 'ใบหน้านี้ถูกลงทะเบียนในระบบแล้ว กรุณาติดต่ออาจารย์');
            setTimeout(() => { window.location.href = ENROLL_CONFIG.dashboardUrl; }, 2500);

        } else if (json.status === 'blocked' || res.status === 403) {
            // เกินจำนวนครั้งที่อนุญาต — ปิดการลงทะเบียน ไม่แสดงปุ่ม retry
            _showResult('error', json.message || 'ลงทะเบียนเกินจำนวนครั้งที่กำหนด กรุณาติดต่ออาจารย์');
            document.querySelectorAll('#enrollContainer button').forEach(btn => {
                btn.disabled = true;
            });

        } else {
            // error + failed_frame hint ถ้ามี
            const frameHint = json.failed_frame ? ` (รูปที่ ${json.failed_frame})` : '';
            _showResult('error', (json.message || 'เกิดข้อผิดพลาด') + frameHint);
        }
    } catch (e) {
        _hideChecking();
        _showResult('error', 'ไม่สามารถเชื่อมต่อ server ได้');
    } finally {
        _enrollSubmitting = false;
    }
}

// ─── Self-Verify: 1 verification shot ────────────────────────────────────────
function _startSelfVerify() {
    document.getElementById('autoCaptureSection').style.display = 'none';
    document.getElementById('selfVerifySection').style.display  = 'block';

    const video  = document.getElementById('videoVerify');
    const canvas = document.getElementById('canvasVerifyDetect');
    const guide  = document.getElementById('faceGuideVerify');
    const status = document.getElementById('verifyStatus');

    let verifyReady = false;
    let countingDown = false;

    const fm = _getSharedFM();

    fm.onResults(async results => {
        if (countingDown) return;
        canvas.width = video.videoWidth; canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const hasFace = results.multiFaceLandmarks?.length > 0;
        if (!hasFace || !_checkFrontal(results.multiFaceLandmarks[0])) {
            guide.classList.remove('ok');
            status.textContent = hasFace ? 'กรุณามองตรงเข้ากล้อง' : 'ไม่พบใบหน้า';
            verifyReady = false;
            return;
        }
        drawFaceFeatures(ctx, results.multiFaceLandmarks[0], canvas.width, canvas.height, 'rgba(74,222,128,0.9)');
        guide.classList.add('ok');

        if (!verifyReady) {
            verifyReady = true;
            countingDown = true;
            // countdown 3→1 then capture
            const overlay = document.getElementById('verifyCountdownOverlay');
            const numEl   = document.getElementById('verifyCountNum');
            overlay.style.display = 'flex';
            for (let i = 3; i >= 1; i--) {
                numEl.textContent = i;
                await new Promise(r => setTimeout(r, 900));
            }
            overlay.style.display = 'none';

            // capture verify shot
            const cap = document.getElementById('captureCanvas');
            cap.width = video.videoWidth; cap.height = video.videoHeight;
            cap.getContext('2d').drawImage(video, 0, 0);
            const verifyB64 = cap.toDataURL('image/jpeg', 0.88);

            // Spoof check + face continuity ทำที่ server ใน /api/self_verify แล้ว
            status.textContent = 'กำลังส่งข้อมูล...';
            _stopStepCamera();
            stopStream(verifyStream);
            await _sendSelfVerify(verifyB64);
        }
    });

    navigator.mediaDevices.getUserMedia({ video: { facingMode:'user', width:640, height:480 } })
    .then(stream => {
        verifyStream = stream;
        video.srcObject = stream;
        const cam = new Camera(video, { onFrame: async () => { if (!countingDown) await fm.send({ image: video }); }, width:640, height:480 });
        _stepCamera = cam;
        cam.start();
        status.textContent = 'มองตรงกล้อง — ระบบจะถ่ายอัตโนมัติ';
    }).catch(e => alert('ไม่สามารถเปิดกล้องได้: ' + e.message));
}

async function _sendSelfVerify(faceImage) {
    if (_enrollSubmitting) return;
    _enrollSubmitting = true;
    _showChecking('กำลังยืนยันตัวตน...');
    _startProgress(_PROGRESS_MSGS_VERIFY);
    try {
        const res  = await fetch(ENROLL_CONFIG.selfVerifyUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': _csrfToken(),
            },
            body: JSON.stringify({
                face_image:         faceImage,
                device_fingerprint: _deviceFingerprint(),   // Sprint 1B
            }),
        });
        _finishProgress();
        const json = await res.json();
        _hideChecking();

        if (json.status === 'success') {
            // Sprint 1B: store the HMAC device token for future check-ins
            if (json.device_token) {
                localStorage.setItem('sc_device_token', json.device_token);
            }
            _showResult('success', json.message);
        } else if (json.status === 'retry') {
            // B2: still have attempts left — show message and let user try again
            _hideChecking();
            document.getElementById('selfVerifySection').style.display = 'block';
            document.getElementById('verifyStatus').textContent = json.message;
            // Re-start the self-verify camera after a short pause
            setTimeout(() => _startSelfVerify(), 2500);
        } else if (json.status === 'failed') {
            _showResult('error', json.message);
        } else if (json.status === 'spoof_detected') {
            // Server ตรวจพบ spoof ใน self-verify — แจ้ง toast + retry
            _showSpoofWarn();
            document.getElementById('selfVerifySection').style.display = 'block';
            document.getElementById('verifyStatus').textContent =
                json.message || 'ตรวจพบภาพปลอม — กรุณาใช้ใบหน้าจริง';
            setTimeout(() => _startSelfVerify(), 2500);
        } else if (json.status === 'continuity_fail') {
            // Server ตรวจพบว่า self-verify frame ไม่ตรงกับ liveness embeddings
            _showResult('error', json.message || 'ตรวจพบใบหน้าไม่ตรงกับ Liveness Check');
            setTimeout(() => fullRestart(), 3000);
        } else {
            _showResult('error', json.message || 'เกิดข้อผิดพลาด');
        }
    } catch (e) {
        _hideChecking();
        _showResult('error', 'ไม่สามารถเชื่อมต่อ server ได้');
    } finally {
        _enrollSubmitting = false;
    }
}

// ─── B1: Progress feedback helpers ───────────────────────────────────────────
let _progressTimers = [];
let _progressAnimFrame = null;
const _PROGRESS_MSGS_ENROLL = [
    { at: 0,    pct: 5,  msg: 'กำลังส่งรูปภาพ...' },
    { at: 4000, pct: 25, msg: 'ตรวจสอบ anti-spoofing...' },
    { at: 10000,pct: 45, msg: 'วิเคราะห์ใบหน้า...' },
    { at: 18000,pct: 62, msg: 'คำนวณ face embedding...' },
    { at: 28000,pct: 78, msg: 'บันทึกลงฐานข้อมูล...' },
    { at: 45000,pct: 88, msg: 'ใกล้เสร็จแล้ว...' },
];
const _PROGRESS_MSGS_VERIFY = [
    { at: 0,    pct: 8,  msg: 'กำลังส่งรูปยืนยัน...' },
    { at: 3000, pct: 30, msg: 'ตรวจสอบ anti-spoofing...' },
    { at: 8000, pct: 55, msg: 'เปรียบเทียบใบหน้า...' },
    { at: 15000,pct: 78, msg: 'บันทึกผลลัพธ์...' },
    { at: 25000,pct: 90, msg: 'ใกล้เสร็จแล้ว...' },
];

function _startProgress(msgs) {
    _clearProgress();
    const bar   = document.getElementById('enrollProgressBar');
    const label = document.getElementById('enrollProgressLabel');
    if (!bar) return;
    bar.style.width = '0%';
    msgs.forEach(({ at, pct, msg }) => {
        const t = setTimeout(() => {
            bar.style.width = pct + '%';
            if (label) label.textContent = msg;
        }, at);
        _progressTimers.push(t);
    });
}

function _finishProgress() {
    _clearProgress();
    const bar   = document.getElementById('enrollProgressBar');
    const label = document.getElementById('enrollProgressLabel');
    if (bar)   { bar.style.width = '100%'; bar.style.background = '#22c55e'; }
    if (label) label.textContent = 'เสร็จสิ้น';
}

function _clearProgress() {
    _progressTimers.forEach(t => clearTimeout(t));
    _progressTimers = [];
    const bar   = document.getElementById('enrollProgressBar');
    const label = document.getElementById('enrollProgressLabel');
    if (bar)   { bar.style.width = '0%'; bar.style.background = ''; }
    if (label) label.textContent = '';
}

function _showChecking(msg) {
    document.getElementById('autoCaptureSection').style.display  = 'none';
    document.getElementById('selfVerifySection').style.display   = 'none';
    document.getElementById('checkingSection').style.display     = 'block';
    document.getElementById('checkingMsg').textContent = msg;
}
function _hideChecking() {
    document.getElementById('checkingSection').style.display = 'none';
    _clearProgress();
}

function _showResult(type, msg) {
    // B6: always stop camera streams when reaching terminal state
    _stopAllStreams();
    goToStep(5);
    if (type === 'success') {
        document.getElementById('successView').style.display = 'block';
        document.getElementById('errorView').style.display   = 'none';
    } else {
        document.getElementById('successView').style.display = 'none';
        document.getElementById('errorView').style.display   = 'block';
        document.getElementById('errorMsg').textContent = msg;
    }
}

// Full restart — back to Step 2 (blink check), clears ALL state including liveness
function fullRestart() {
    if (step4Timer) { clearTimeout(step4Timer); step4Timer = null; }
    capturedImages      = [];
    enrollmentSessionId = null;
    lastCaptureTime     = 0;
    capturePaused       = false;
    blinkAttempts       = 0;
    challengeAttempts   = 0;
    calibEARValues      = [];
    baselineEAR         = null;
    calibrating         = false;
    step4SpoofFailConsecutive = 0;
    step4SpoofFailTotal       = 0;
    _enrollSubmitting        = false;   // B5: release double-submit lock on full restart
    earSamplesDuringCapture  = [];
    _rtResetCounters();

    // ล้าง liveness embeddings ที่ server (session["liveness_embeddings"])
    fetch(ENROLL_CONFIG.resetLivenessUrl, {
        method: 'POST',
        headers: { 'X-CSRF-Token': _csrfToken() },
    }).catch(e => _warn('reset-liveness fetch failed:', e));

    _stopAllStreams();

    document.getElementById('thumbnailRow').innerHTML = '';
    document.getElementById('autoCaptureSection').style.display = 'block';
    document.getElementById('selfVerifySection').style.display  = 'none';
    document.getElementById('checkingSection').style.display    = 'none';

    _clearSpoofLabel('spoofLabelCalib');
    _clearSpoofLabel('spoofLabelLiveness');
    _clearSpoofLabel('spoofLabelCapture');

    // Reset calibration button states so user can start calibration again
    const btnStart = document.getElementById('btnStartCalib');
    if (btnStart) { btnStart.disabled = false; btnStart.style.display = 'block'; }
    document.getElementById('btnRetryBlink').style.display  = 'none';
    document.getElementById('btnBlinkGiveUp').style.display = 'none';
    document.getElementById('calibProgressWrap').style.display = 'none';

    _updateCaptureDots();
    goToStep(2);
    startCamera('videoCalib', stream => { calibStream = stream; });
}

// Partial restart — back to Step 4 only; server-side liveness embeddings ยังอยู่
function restartCapture() {
    if (step4Timer) { clearTimeout(step4Timer); step4Timer = null; }
    capturedImages      = [];
    lastCaptureTime     = 0;
    capturePaused             = false;
    step4SpoofFailConsecutive = 0;
    step4SpoofFailTotal       = 0;
    _enrollSubmitting        = false;   // B5: release double-submit lock on capture restart
    earSamplesDuringCapture  = [];
    _rtResetCounters();

    stopStream(captureStream);
    captureStream = null;
    _stopStepCamera();

    document.getElementById('thumbnailRow').innerHTML = '';
    document.getElementById('autoCaptureSection').style.display = 'block';
    document.getElementById('selfVerifySection').style.display  = 'none';
    document.getElementById('checkingSection').style.display    = 'none';
    _clearSpoofLabel('spoofLabelCapture');

    _updateCaptureDots();
    goToStep(4);
    startCaptureWithDetection();
}
