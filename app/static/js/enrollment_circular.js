// enrollment_circular.js — Apple-style circular enrollment flow
// Phase 3: real pose detection via MediaPipe FaceMesh
// Loaded only when ENROLL_FLOW_MODE=circular
//
// Functions reused from enrollment_flow.js (top-level globals — no window exposure needed):
//   _getSharedFM(opts)         — shared FaceMesh singleton
//   _checkBlur(videoEl)        — Laplacian variance (>20 = sharp)
//   _checkCameraConditions(el) — brightness + backlight check → {ok, reason}
//   _captureFrameFromVideo(el) — returns base64 JPEG or null

console.log("[circular] module loaded");

// ─── Inject CSS ───────────────────────────────────────────────────────────────
(function () {
    const s = document.createElement('style');
    s.textContent = `
        #circularTickSvg line { transition: stroke 0.35s ease, stroke-width 0.2s ease; }
        #circFaceGuide        { transition: stroke 0.4s ease, stroke-dasharray 0.4s ease; }
    `;
    document.head.appendChild(s);
})();

// ─── Zone definitions ─────────────────────────────────────────────────────────
// 12 ticks, index 0 = 12 o'clock, going clockwise (30° apart).
// match(yaw°, pitch°) → true when head is in that zone.
// Yaw formula (from _checkFrontal): positive = turned right, negative = left.
// Pitch formula: positive = looking down, negative = looking up.
const CIRC_ZONES = {
    FRONT: { ticks: [0],        label: 'หน้าตรง', match: (y,p) => Math.abs(y) <= 5  && Math.abs(p) <= 5   },
    UP:    { ticks: [11, 1],    label: 'เงยหน้า', match: (y,p) => p >= -15 && p <= -5  && Math.abs(y) <= 10 },
    RIGHT: { ticks: [2, 3, 4],  label: 'หันขวา', match: (y,p) => y >= 10  && y <= 20  && Math.abs(p) <= 10 },
    DOWN:  { ticks: [5, 6, 7],  label: 'ก้มหน้า', match: (y,p) => p >= 5   && p <= 15  && Math.abs(y) <= 10 },
    LEFT:  { ticks: [8, 9, 10], label: 'หันซ้าย', match: (y,p) => y <= -10 && y >= -20 && Math.abs(p) <= 10 },
};

// ─── Visual constants ─────────────────────────────────────────────────────────
const CIRC_NUM_TICKS  = 12;
const CIRC_CX = 130, CIRC_CY = 130;
const CIRC_TICK_OUTER = 126, CIRC_TICK_INNER = 107, CIRC_TICK_W = 8;
const CIRC_GUIDE_R    = 98;
const COLOR_EMPTY = '#e2e8f0', COLOR_DONE = '#22c55e';

// ─── Detection constants ──────────────────────────────────────────────────────
const CIRC_BUF_SIZE    = 10;    // rolling landmark buffer length
const CIRC_AVG_FRAMES  = 5;     // frames to average for zone decision
const CIRC_HOLD_MS     = 400;   // ms the zone must be held before capture
const CIRC_TIMEOUT_MS  = 60000; // 60 s total timeout
const CIRC_FACE_MIN_H  = 0.30;  // minimum face height (normalised [0,1])
const CIRC_BLUR_MIN    = 20;    // minimum Laplacian variance

// ─── Mutable state ────────────────────────────────────────────────────────────
const _circDone = {};                   // zoneName → true  (captured + ticked)
window.circularCapturedFrames = {};     // zoneName → base64 JPEG

let _circCamera  = null;
let _circFM      = null;
let _circActive  = false;

const _circBuf   = [];                  // [{yaw, pitch, faceH, ts}] rolling buffer
let _circHoldZone  = null;              // zone name currently held
let _circHoldStart = null;              // Date.now() when hold began

let _circTimeoutTimer = null;

// ─── SVG init ─────────────────────────────────────────────────────────────────
function circularInit() {
    const svg = document.getElementById('circularTickSvg');
    if (!svg) return;

    const guide = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    guide.setAttribute('cx', CIRC_CX); guide.setAttribute('cy', CIRC_CY);
    guide.setAttribute('r', CIRC_GUIDE_R);
    guide.setAttribute('fill', 'none');
    guide.setAttribute('stroke', 'rgba(255,255,255,0.5)');
    guide.setAttribute('stroke-width', '2');
    guide.setAttribute('stroke-dasharray', '6 5');
    guide.id = 'circFaceGuide';
    svg.appendChild(guide);

    for (let i = 0; i < CIRC_NUM_TICKS; i++) {
        const deg = i * (360 / CIRC_NUM_TICKS) - 90;  // 0 = top, clockwise
        const rad = deg * Math.PI / 180;
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', (CIRC_CX + CIRC_TICK_INNER * Math.cos(rad)).toFixed(2));
        line.setAttribute('y1', (CIRC_CY + CIRC_TICK_INNER * Math.sin(rad)).toFixed(2));
        line.setAttribute('x2', (CIRC_CX + CIRC_TICK_OUTER * Math.cos(rad)).toFixed(2));
        line.setAttribute('y2', (CIRC_CY + CIRC_TICK_OUTER * Math.sin(rad)).toFixed(2));
        line.setAttribute('stroke', COLOR_EMPTY);
        line.setAttribute('stroke-width', CIRC_TICK_W);
        line.setAttribute('stroke-linecap', 'round');
        line.id = `circ-tick-${i}`;
        svg.appendChild(line);
    }
}

// ─── Pose math ────────────────────────────────────────────────────────────────
// Same formula as _checkFrontal() in enrollment_flow.js.
function _circPose(lm) {
    const nose = lm[1], leftEar = lm[234], rightEar = lm[454];
    const top  = lm[10],  chin   = lm[152];
    const faceW = rightEar.x - leftEar.x;
    const faceH = chin.y - top.y;
    if (faceW <= 0 || faceH <= 0) return null;
    return {
        yaw:   ((nose.x - leftEar.x) / faceW - 0.5) * 90,
        pitch: ((nose.y - top.y)     / faceH - 0.5) * 90,
        faceH,
    };
}

function _circWhichZone(yaw, pitch) {
    for (const [name, zone] of Object.entries(CIRC_ZONES)) {
        if (zone.match(yaw, pitch)) return name;
    }
    return null;
}

// ─── FaceMesh frame callback ──────────────────────────────────────────────────
function _circOnResults(results, video) {
    if (!_circActive) return;

    if (!results.multiFaceLandmarks?.length) {
        _circSetStatus('ไม่พบใบหน้า — จัดหน้าให้อยู่ในกรอบ');
        _circHoldZone = null; _circHoldStart = null;
        return;
    }

    const lm   = results.multiFaceLandmarks[0];
    const pose = _circPose(lm);
    if (!pose) return;

    const now = Date.now();
    _circBuf.push({ ...pose, ts: now });
    if (_circBuf.length > CIRC_BUF_SIZE) _circBuf.shift();
    if (_circBuf.length < CIRC_AVG_FRAMES) return;

    // Smooth over last N frames to suppress jitter
    const recent   = _circBuf.slice(-CIRC_AVG_FRAMES);
    const avgYaw   = recent.reduce((s, f) => s + f.yaw,   0) / CIRC_AVG_FRAMES;
    const avgPitch = recent.reduce((s, f) => s + f.pitch, 0) / CIRC_AVG_FRAMES;
    const avgFaceH = recent.reduce((s, f) => s + f.faceH, 0) / CIRC_AVG_FRAMES;

    const zone = _circWhichZone(avgYaw, avgPitch);

    if (zone !== _circHoldZone) {
        // Zone changed — reset hold timer
        _circHoldZone  = zone;
        _circHoldStart = zone ? now : null;
        if (zone && !_circDone[zone]) {
            _circSetStatus(CIRC_ZONES[zone].label + ' — ค้างไว้สักครู่...');
        } else if (!zone) {
            _circSetStatus('หมุนหน้าช้าๆ');
        }
    } else if (zone && !_circDone[zone] && _circHoldStart) {
        // Same zone held — check if long enough
        if (now - _circHoldStart >= CIRC_HOLD_MS) {
            _circTryCaptureZone(zone, video, avgFaceH);
        }
    }
}

// ─── Quality-gated capture ────────────────────────────────────────────────────
function _circTryCaptureZone(zoneName, video, avgFaceH) {
    if (_circDone[zoneName]) return;

    // Gate 1: face size
    if (avgFaceH < CIRC_FACE_MIN_H) {
        _circSetStatus('เข้าใกล้กล้องอีกหน่อย');
        return;
    }
    // Gate 2: blur
    if (_checkBlur(video) < CIRC_BLUR_MIN) {
        _circSetStatus('ภาพเบลอ — ถือกล้องให้นิ่ง');
        return;
    }
    // Gate 3: lighting
    const cond = _checkCameraConditions(video);
    if (!cond.ok) {
        _circSetStatus(cond.reason);
        return;
    }
    // Capture
    const frame = _captureFrameFromVideo(video);
    if (!frame) return;

    // Commit (mark done before any async work to prevent double-capture)
    window.circularCapturedFrames[zoneName] = frame;
    circularMarkZone(zoneName);                         // sets _circDone[zoneName], updates ticks
    if (navigator.vibrate) navigator.vibrate(50);
    _circSetStatus('✓ ' + CIRC_ZONES[zoneName].label);

    const total = Object.keys(CIRC_ZONES).length;
    if (Object.keys(_circDone).length >= total) {
        setTimeout(_circHandleAllComplete, 150);        // defer to allow tick animation to play
    }
}

// ─── Camera management ────────────────────────────────────────────────────────
function startCircularCapture() {
    const video = document.getElementById('videoCircular');
    if (!video || _circActive) return;

    _circSetStatus('กำลังเปิดกล้อง...');

    const fm = _getSharedFM({ refineLandmarks: false });
    _circFM  = fm;
    fm.onResults(results => _circOnResults(results, video));

    _circCamera = new Camera(video, {
        onFrame: async () => {
            if (_circActive) await fm.send({ image: video });
        },
        width: 640, height: 480,
    });

    _circCamera.start()
        .then(() => {
            _circActive       = true;
            _circTimeoutTimer = setTimeout(_circHandleTimeout, CIRC_TIMEOUT_MS);
            const btn = document.getElementById('btnCircularRestart');
            if (btn) btn.style.display = 'block';
            _circSetStatus('');
        })
        .catch(e => _circSetStatus('ไม่สามารถเปิดกล้องได้: ' + e.message));
}

function stopCircularCapture() {
    _circActive = false;
    clearTimeout(_circTimeoutTimer);
    _circTimeoutTimer = null;
    if (_circCamera) { try { _circCamera.stop(); } catch(e) {} _circCamera = null; }
    if (_circFM)     { try { _circFM.onResults(() => {}); } catch(e) {} _circFM = null; }
    _circBuf.length  = 0;
    _circHoldZone    = null;
    _circHoldStart   = null;
}

// ─── Completion & timeout ─────────────────────────────────────────────────────
function _circHandleAllComplete() {
    stopCircularCapture();

    const instr = document.getElementById('circularInstruction');
    const stat  = document.getElementById('circularStatus');
    const chk   = document.getElementById('circularCheckingOverlay');
    const btn   = document.getElementById('btnCircularRestart');

    if (instr) instr.style.display = 'none';
    if (stat)  stat.style.display  = 'none';
    if (chk)   chk.style.display   = 'block';
    if (btn)   btn.style.display   = 'none';

    // Phase 3: log captured frames; Phase 5 will POST to /api/enroll
    console.log('[circular] All 5 zones captured:', Object.keys(window.circularCapturedFrames));
    Object.entries(window.circularCapturedFrames).forEach(([zone, frame]) =>
        console.log(`  ${zone}: ${frame.substring(0, 80)}...`)
    );
}

function _circHandleTimeout() {
    stopCircularCapture();
    _circSetStatus('หมดเวลา — กรุณาลองใหม่');
    setTimeout(() => {
        circularReset();
        window.circularCapturedFrames = {};
        startCircularCapture();
    }, 3000);
}

// ─── User-visible restart ─────────────────────────────────────────────────────
function circularRestart() {
    stopCircularCapture();
    window.circularCapturedFrames = {};
    circularReset();

    const chk   = document.getElementById('circularCheckingOverlay');
    const instr = document.getElementById('circularInstruction');
    const stat  = document.getElementById('circularStatus');
    const btn   = document.getElementById('btnCircularRestart');

    if (chk)   { chk.style.display = 'none'; }
    if (instr) { instr.style.display = ''; instr.textContent = 'หมุนหน้าช้าๆ เป็นวงกลม'; instr.style.color = ''; }
    if (stat)  { stat.style.display = ''; stat.textContent = ''; }
    if (btn)   btn.style.display = 'none';

    startCircularCapture();
}

// ─── Tick marking (called by real capture + Phase 2 test buttons) ─────────────
function circularMarkZone(zoneName) {
    const zone = CIRC_ZONES[zoneName];
    if (!zone || _circDone[zoneName]) return;
    _circDone[zoneName] = true;

    zone.ticks.forEach(i => {
        const el = document.getElementById(`circ-tick-${i}`);
        if (el) {
            el.setAttribute('stroke', COLOR_DONE);
            el.setAttribute('stroke-width', CIRC_TICK_W + 1);
        }
    });

    const done  = Object.keys(_circDone).length;
    const total = Object.keys(CIRC_ZONES).length;
    const prog  = document.getElementById('circularProgress');
    if (prog) prog.textContent = `${done}/${total} มุมเก็บแล้ว`;

    if (done >= total) {
        // All ticks done — turn face-guide ring green + update progress text
        const guide = document.getElementById('circFaceGuide');
        if (guide) {
            guide.setAttribute('stroke', COLOR_DONE);
            guide.setAttribute('stroke-width', '3');
            guide.setAttribute('stroke-dasharray', 'none');
        }
        if (prog) { prog.textContent = '✓ ครบทุกมุม!'; prog.style.color = '#16a34a'; }
    }
}

// ─── Reset (also called from test buttons / restart) ─────────────────────────
function circularReset() {
    Object.keys(_circDone).forEach(k => delete _circDone[k]);

    for (let i = 0; i < CIRC_NUM_TICKS; i++) {
        const el = document.getElementById(`circ-tick-${i}`);
        if (el) {
            el.setAttribute('stroke', COLOR_EMPTY);
            el.setAttribute('stroke-width', CIRC_TICK_W);
        }
    }

    const prog  = document.getElementById('circularProgress');
    const guide = document.getElementById('circFaceGuide');
    if (prog)  { prog.textContent = `0/${Object.keys(CIRC_ZONES).length} มุมเก็บแล้ว`; prog.style.color = ''; }
    if (guide) {
        guide.setAttribute('stroke', 'rgba(255,255,255,0.5)');
        guide.setAttribute('stroke-width', '2');
        guide.setAttribute('stroke-dasharray', '6 5');
    }
}

// ─── Status helper ────────────────────────────────────────────────────────────
function _circSetStatus(text) {
    const el = document.getElementById('circularStatus');
    if (el) el.textContent = text;
}

// ─── Override startLivenessChallenge ─────────────────────────────────────────
// enrollment_flow.js calls startLivenessChallenge() after EAR check.
// In circular mode we intercept it here (this file loads after enrollment_flow.js).
window.startLivenessChallenge = function () {
    console.log('[circular] intercepted startLivenessChallenge → startCircularCapture');
    startCircularCapture();
};

// ─── Boot ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', circularInit);
