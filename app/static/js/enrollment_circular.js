// enrollment_circular.js — Apple-style circular enrollment flow
// Phase 3: real pose detection  |  Phase 4: surprise blink anti-spoof
// Loaded only when ENROLL_FLOW_MODE=circular
//
// Functions reused from enrollment_flow.js (top-level globals):
//   _getSharedFM(opts)         — shared FaceMesh singleton
//   _computeEAR(lm)            — Eye Aspect Ratio from 468 landmarks
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
        @keyframes _circBlinkIn { from { opacity:0; transform:scale(0.92); }
                                  to   { opacity:1; transform:scale(1); } }
        #circBlinkOverlay     { animation: _circBlinkIn 0.2s ease; }
    `;
    document.head.appendChild(s);
})();

// ─── Zone definitions ─────────────────────────────────────────────────────────
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

// ─── Pose-detection constants ─────────────────────────────────────────────────
const CIRC_BUF_SIZE    = 10;
const CIRC_AVG_FRAMES  = 5;
const CIRC_HOLD_MS     = 400;
const CIRC_TIMEOUT_MS  = 60000;
const CIRC_FACE_MIN_H  = 0.30;
const CIRC_BLUR_MIN    = 20;

// ─── Blink-challenge constants ────────────────────────────────────────────────
const BLINK_WINDOW_MS  = 3000;   // detection window
const BLINK_EAR_CLOSE  = 0.15;   // EAR below this = eyes closing
const BLINK_EAR_OPEN   = 0.22;   // EAR above this = eyes opened again
const BLINK_REOPEN_MS  = 500;    // max close→open time for a valid blink
const BLINK_STDDEV_MIN = 0.03;   // EAR std-dev gate (rejects static photo noise)

// ─── Mutable state ────────────────────────────────────────────────────────────
const _circDone = {};
window.circularCapturedFrames = {};
window.circularBlinkDone      = false;

let _circCamera  = null;
let _circFM      = null;
let _circActive  = false;

const _circBuf = [];
let _circHoldZone  = null;
let _circHoldStart = null;
let _circTimeoutTimer = null;

// Blink-challenge state
let _circBlinkActive    = false;
let _circBlinkTriggerAt = 2 + Math.floor(Math.random() * 2); // 2 or 3 zones
const _blinkEarBuf      = [];
let _blinkStartTime     = null;
let _blinkTimer         = null;
let _blinkCountInterval = null;
let _blinkClosed        = false;
let _blinkCloseTime     = null;

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
        const deg = i * (360 / CIRC_NUM_TICKS) - 90;
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

// ─── Pose math (same formula as _checkFrontal in enrollment_flow.js) ──────────
function _circPose(lm) {
    const nose = lm[1], leftEar = lm[234], rightEar = lm[454];
    const top  = lm[10], chin   = lm[152];
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
        if (!_circBlinkActive) {
            _circSetStatus('ไม่พบใบหน้า — จัดหน้าให้อยู่ในกรอบ');
            _circHoldZone = null; _circHoldStart = null;
        }
        return;
    }

    const lm = results.multiFaceLandmarks[0];

    // Blink challenge takes priority — bypass zone detection while active
    if (_circBlinkActive) {
        _circBlinkCheckFrame(lm);
        return;
    }

    const pose = _circPose(lm);
    if (!pose) return;

    const now = Date.now();
    _circBuf.push({ ...pose, ts: now });
    if (_circBuf.length > CIRC_BUF_SIZE) _circBuf.shift();
    if (_circBuf.length < CIRC_AVG_FRAMES) return;

    const recent   = _circBuf.slice(-CIRC_AVG_FRAMES);
    const avgYaw   = recent.reduce((s, f) => s + f.yaw,   0) / CIRC_AVG_FRAMES;
    const avgPitch = recent.reduce((s, f) => s + f.pitch, 0) / CIRC_AVG_FRAMES;
    const avgFaceH = recent.reduce((s, f) => s + f.faceH, 0) / CIRC_AVG_FRAMES;

    const zone = _circWhichZone(avgYaw, avgPitch);

    if (zone !== _circHoldZone) {
        _circHoldZone  = zone;
        _circHoldStart = zone ? now : null;
        if (zone && !_circDone[zone]) _circSetStatus(CIRC_ZONES[zone].label + ' — ค้างไว้สักครู่...');
        else if (!zone)               _circSetStatus('หมุนหน้าช้าๆ');
    } else if (zone && !_circDone[zone] && _circHoldStart) {
        if (now - _circHoldStart >= CIRC_HOLD_MS) {
            _circTryCaptureZone(zone, video, avgFaceH);
        }
    }
}

// ─── Quality-gated capture ────────────────────────────────────────────────────
function _circTryCaptureZone(zoneName, video, avgFaceH) {
    if (_circDone[zoneName]) return;

    if (avgFaceH < CIRC_FACE_MIN_H)         { _circSetStatus('เข้าใกล้กล้องอีกหน่อย');   return; }
    if (_checkBlur(video) < CIRC_BLUR_MIN)  { _circSetStatus('ภาพเบลอ — ถือกล้องให้นิ่ง'); return; }
    const cond = _checkCameraConditions(video);
    if (!cond.ok)                            { _circSetStatus(cond.reason);               return; }

    const frame = _captureFrameFromVideo(video);
    if (!frame) return;

    window.circularCapturedFrames[zoneName] = frame;
    circularMarkZone(zoneName);
    if (navigator.vibrate) navigator.vibrate(50);
    _circSetStatus('✓ ' + CIRC_ZONES[zoneName].label);

    const doneCount = Object.keys(_circDone).length;
    const total     = Object.keys(CIRC_ZONES).length;

    if (doneCount >= total) {
        setTimeout(_circHandleAllComplete, 150);
    } else if (doneCount >= _circBlinkTriggerAt && !window.circularBlinkDone) {
        // Surprise blink challenge after capturing 2 or 3 zones
        setTimeout(_circStartBlinkChallenge, 600);  // brief pause after tick feedback
    }
}

// ─── Blink challenge ──────────────────────────────────────────────────────────
function _circStartBlinkChallenge() {
    if (window.circularBlinkDone || _circBlinkActive) return;
    _circBlinkActive = true;

    // Reset detection state
    _blinkEarBuf.length = 0;
    _blinkStartTime     = Date.now();
    _blinkClosed        = false;
    _blinkCloseTime     = null;

    // Show overlay
    const ov = document.getElementById('circBlinkOverlay');
    if (ov) ov.style.display = 'flex';

    // Countdown label
    let remaining = Math.ceil(BLINK_WINDOW_MS / 1000);
    const cd = document.getElementById('circBlinkCountdown');
    if (cd) cd.textContent = remaining;
    _blinkCountInterval = setInterval(() => {
        remaining = Math.max(0, remaining - 1);
        if (cd) cd.textContent = remaining;
    }, 1000);

    // Hard timeout
    _blinkTimer = setTimeout(_circBlinkTimeout, BLINK_WINDOW_MS);
}

function _circBlinkCheckFrame(lm) {
    const ear = _computeEAR(lm);
    _blinkEarBuf.push(ear);

    // State machine: OPEN → CLOSING (ear drops) → REOPEN (ear rises) = blink
    if (!_blinkClosed && ear < BLINK_EAR_CLOSE) {
        _blinkClosed    = true;
        _blinkCloseTime = Date.now();
    } else if (_blinkClosed && ear > BLINK_EAR_OPEN) {
        const reopenMs = Date.now() - _blinkCloseTime;
        if (reopenMs <= BLINK_REOPEN_MS) {
            _circBlinkDetected();
        } else {
            // Too slow — eyes creeping open, not a real blink; reset and keep watching
            _blinkClosed = false; _blinkCloseTime = null;
        }
    }
}

function _circBlinkDetected() {
    clearTimeout(_blinkTimer);
    clearInterval(_blinkCountInterval);

    const earArr   = [..._blinkEarBuf];
    const minEar   = Math.min(...earArr);
    const maxEar   = Math.max(...earArr);
    const startEar = earArr[0] || 0;
    const mean     = earArr.reduce((s, v) => s + v, 0) / earArr.length;
    const stdDev   = Math.sqrt(earArr.reduce((s, v) => s + (v - mean) ** 2, 0) / earArr.length);

    console.log(`[BLINK] start_ear=${startEar.toFixed(3)} min_ear=${minEar.toFixed(3)}` +
                ` max_ear=${maxEar.toFixed(3)} std_dev=${stdDev.toFixed(3)} detected=true`);

    if (stdDev < BLINK_STDDEV_MIN) {
        // EAR variation too small — likely camera noise on a static image, not a real blink
        console.log('[BLINK] rejected: std_dev below threshold (static face suspected)');
        _circBlinkFail('ไม่พบการกะพริบตา — อาจเป็นภาพนิ่ง กรุณาลองใหม่');
        return;
    }

    // Confirmed real blink
    window.circularBlinkDone = true;
    _circBlinkActive = false;

    const ov = document.getElementById('circBlinkOverlay');
    if (ov) ov.style.display = 'none';

    _circSetStatus('✓ กะพริบตาผ่านแล้ว');
    setTimeout(() => { if (!_circBlinkActive) _circSetStatus(''); }, 1200);
}

function _circBlinkTimeout() {
    clearInterval(_blinkCountInterval);

    const earArr   = [..._blinkEarBuf];
    const minEar   = earArr.length ? Math.min(...earArr) : 0;
    const maxEar   = earArr.length ? Math.max(...earArr) : 0;
    const startEar = earArr[0] || 0;
    const mean     = earArr.length ? earArr.reduce((s,v)=>s+v,0)/earArr.length : 0;
    const stdDev   = earArr.length ? Math.sqrt(earArr.reduce((s,v)=>s+(v-mean)**2,0)/earArr.length) : 0;

    console.log(`[BLINK] start_ear=${startEar.toFixed(3)} min_ear=${minEar.toFixed(3)}` +
                ` max_ear=${maxEar.toFixed(3)} std_dev=${stdDev.toFixed(3)} detected=false (timeout)`);

    _circBlinkFail('ไม่พบการกะพริบตา — อาจเป็นภาพนิ่ง กรุณาลองใหม่');
}

function _circBlinkFail(reason) {
    _circBlinkActive = false;
    clearTimeout(_blinkTimer);
    clearInterval(_blinkCountInterval);

    const ov = document.getElementById('circBlinkOverlay');
    if (ov) ov.style.display = 'none';

    stopCircularCapture();

    const stat = document.getElementById('circularStatus');
    if (stat) { stat.textContent = reason; stat.style.color = '#dc2626'; }

    setTimeout(() => {
        if (stat) stat.style.color = '';
        circularRestart();
    }, 3000);
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
    _circActive      = false;
    _circBlinkActive = false;
    clearTimeout(_circTimeoutTimer); _circTimeoutTimer = null;
    clearTimeout(_blinkTimer);       _blinkTimer       = null;
    clearInterval(_blinkCountInterval); _blinkCountInterval = null;
    if (_circCamera) { try { _circCamera.stop(); } catch(e) {} _circCamera = null; }
    if (_circFM)     { try { _circFM.onResults(() => {}); } catch(e) {} _circFM = null; }
    _circBuf.length = 0;
    _circHoldZone = null; _circHoldStart = null;
    _blinkEarBuf.length = 0;
    _blinkClosed = false; _blinkCloseTime = null;
}

// ─── Completion & timeout ─────────────────────────────────────────────────────
function _circHandleAllComplete() {
    stopCircularCapture();

    document.getElementById('circularInstruction').style.display = 'none';
    document.getElementById('circularStatus').style.display      = 'none';
    document.getElementById('circularCheckingOverlay').style.display = 'block';
    document.getElementById('btnCircularRestart').style.display  = 'none';

    console.log('[circular] All 5 zones captured:', Object.keys(window.circularCapturedFrames));
    Object.entries(window.circularCapturedFrames).forEach(([zone, frame]) =>
        console.log(`  ${zone}: ${frame.substring(0, 80)}...`)
    );
    // Phase 5 will POST to /api/enroll here
}

function _circHandleTimeout() {
    stopCircularCapture();
    _circSetStatus('หมดเวลา — กรุณาลองใหม่');
    setTimeout(() => { circularReset(); window.circularCapturedFrames = {}; startCircularCapture(); }, 3000);
}

// ─── User-visible restart ─────────────────────────────────────────────────────
function circularRestart() {
    stopCircularCapture();
    window.circularCapturedFrames = {};
    window.circularBlinkDone      = false;
    _circBlinkTriggerAt = 2 + Math.floor(Math.random() * 2); // re-randomise
    circularReset();

    const ov    = document.getElementById('circBlinkOverlay');
    const chk   = document.getElementById('circularCheckingOverlay');
    const instr = document.getElementById('circularInstruction');
    const stat  = document.getElementById('circularStatus');
    const btn   = document.getElementById('btnCircularRestart');

    if (ov)    ov.style.display    = 'none';
    if (chk)   chk.style.display   = 'none';
    if (instr) { instr.style.display = ''; instr.textContent = 'หมุนหน้าช้าๆ เป็นวงกลม'; instr.style.color = ''; }
    if (stat)  { stat.style.display = ''; stat.textContent = ''; stat.style.color = ''; }
    if (btn)   btn.style.display   = 'none';

    startCircularCapture();
}

// ─── Tick marking ─────────────────────────────────────────────────────────────
function circularMarkZone(zoneName) {
    const zone = CIRC_ZONES[zoneName];
    if (!zone || _circDone[zoneName]) return;
    _circDone[zoneName] = true;

    zone.ticks.forEach(i => {
        const el = document.getElementById(`circ-tick-${i}`);
        if (el) { el.setAttribute('stroke', COLOR_DONE); el.setAttribute('stroke-width', CIRC_TICK_W + 1); }
    });

    const done  = Object.keys(_circDone).length;
    const total = Object.keys(CIRC_ZONES).length;
    const prog  = document.getElementById('circularProgress');
    if (prog) prog.textContent = `${done}/${total} มุมเก็บแล้ว`;

    if (done >= total) {
        const guide = document.getElementById('circFaceGuide');
        if (guide) { guide.setAttribute('stroke', COLOR_DONE); guide.setAttribute('stroke-width', '3'); guide.setAttribute('stroke-dasharray', 'none'); }
        if (prog)  { prog.textContent = '✓ ครบทุกมุม!'; prog.style.color = '#16a34a'; }
    }
}

// ─── Reset ────────────────────────────────────────────────────────────────────
function circularReset() {
    Object.keys(_circDone).forEach(k => delete _circDone[k]);
    for (let i = 0; i < CIRC_NUM_TICKS; i++) {
        const el = document.getElementById(`circ-tick-${i}`);
        if (el) { el.setAttribute('stroke', COLOR_EMPTY); el.setAttribute('stroke-width', CIRC_TICK_W); }
    }
    const prog  = document.getElementById('circularProgress');
    const guide = document.getElementById('circFaceGuide');
    if (prog)  { prog.textContent = `0/${Object.keys(CIRC_ZONES).length} มุมเก็บแล้ว`; prog.style.color = ''; }
    if (guide) { guide.setAttribute('stroke','rgba(255,255,255,0.5)'); guide.setAttribute('stroke-width','2'); guide.setAttribute('stroke-dasharray','6 5'); }
}

function _circSetStatus(text) {
    const el = document.getElementById('circularStatus');
    if (el) el.textContent = text;
}

// ─── Override startLivenessChallenge ─────────────────────────────────────────
window.startLivenessChallenge = function () {
    console.log('[circular] intercepted startLivenessChallenge → startCircularCapture');
    startCircularCapture();
};

// ─── Boot ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', circularInit);
