// enrollment_circular.js — Apple-style circular enrollment flow
// Loaded only when ENROLL_FLOW_MODE=circular

console.log("[circular] module loaded");

// ─── Inject CSS for smooth tick transitions ───────────────────────────────────
(function () {
    const s = document.createElement('style');
    s.textContent = `
        #circularTickSvg line {
            transition: stroke 0.35s ease, stroke-width 0.2s ease;
        }
        #circFaceGuide {
            transition: stroke 0.4s ease, stroke-dasharray 0.4s ease;
        }
    `;
    document.head.appendChild(s);
})();

// ─── Zone → tick index mapping ────────────────────────────────────────────────
// 12 ticks, index 0 = 12 o'clock, going clockwise (30° apart).
// Each zone owns a contiguous arc of the ring:
//   FRONT  =  top center      (single "crown" tick at 12 o'clock)
//   UP     =  flanking FRONT  (11 o'clock + 1 o'clock)
//   RIGHT  =  right quadrant  (2, 3, 4 o'clock)
//   DOWN   =  bottom          (5, 6, 7 o'clock)
//   LEFT   =  left quadrant   (8, 9, 10 o'clock)
const CIRC_ZONES = {
    FRONT: { ticks: [0],        label: 'หน้าตรง' },
    UP:    { ticks: [11, 1],    label: 'เงยหน้า' },
    RIGHT: { ticks: [2, 3, 4],  label: 'หันขวา' },
    DOWN:  { ticks: [5, 6, 7],  label: 'ก้มหน้า' },
    LEFT:  { ticks: [8, 9, 10], label: 'หันซ้าย' },
};

// ─── Visual constants ─────────────────────────────────────────────────────────
const CIRC_NUM_TICKS  = 12;
const CIRC_CX         = 130;     // SVG center x (viewBox 0 0 260 260)
const CIRC_CY         = 130;     // SVG center y
const CIRC_TICK_OUTER = 126;     // outer radius of tick line
const CIRC_TICK_INNER = 107;     // inner radius (tick length = 19 px)
const CIRC_TICK_W     = 8;       // stroke-width
const CIRC_GUIDE_R    = 98;      // dashed face-guide circle radius
const COLOR_EMPTY     = '#e2e8f0';
const COLOR_ACTIVE    = '#4f46e5';
const COLOR_DONE      = '#22c55e';

const _circDone = {};            // zoneName → true once captured

// ─── Init: draw SVG face guide + tick marks ───────────────────────────────────
function circularInit() {
    const svg = document.getElementById('circularTickSvg');
    if (!svg) return;

    // Face-guide dashed circle
    const guide = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    guide.setAttribute('cx', CIRC_CX);
    guide.setAttribute('cy', CIRC_CY);
    guide.setAttribute('r',  CIRC_GUIDE_R);
    guide.setAttribute('fill', 'none');
    guide.setAttribute('stroke', 'rgba(255,255,255,0.5)');
    guide.setAttribute('stroke-width', '2');
    guide.setAttribute('stroke-dasharray', '6 5');
    guide.id = 'circFaceGuide';
    svg.appendChild(guide);

    // 12 tick lines
    for (let i = 0; i < CIRC_NUM_TICKS; i++) {
        // angle: 0 = top (−90° in standard math), clockwise
        const deg = i * (360 / CIRC_NUM_TICKS) - 90;
        const rad = deg * Math.PI / 180;
        const x1  = CIRC_CX + CIRC_TICK_INNER * Math.cos(rad);
        const y1  = CIRC_CY + CIRC_TICK_INNER * Math.sin(rad);
        const x2  = CIRC_CX + CIRC_TICK_OUTER * Math.cos(rad);
        const y2  = CIRC_CY + CIRC_TICK_OUTER * Math.sin(rad);

        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x1.toFixed(2));
        line.setAttribute('y1', y1.toFixed(2));
        line.setAttribute('x2', x2.toFixed(2));
        line.setAttribute('y2', y2.toFixed(2));
        line.setAttribute('stroke', COLOR_EMPTY);
        line.setAttribute('stroke-width', CIRC_TICK_W);
        line.setAttribute('stroke-linecap', 'round');
        line.id = `circ-tick-${i}`;
        svg.appendChild(line);
    }
}

// ─── Camera ───────────────────────────────────────────────────────────────────
let _circStream = null;

async function startCircularCamera() {
    const video = document.getElementById('videoCircular');
    if (!video || _circStream) return;
    try {
        _circStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: 640, height: 480 },
        });
        video.srcObject = _circStream;
    } catch (e) {
        console.warn('[circular] camera:', e);
        const el = document.getElementById('circularInstruction');
        if (el) el.textContent = 'ไม่สามารถเปิดกล้องได้';
    }
}

function stopCircularCamera() {
    if (_circStream) {
        _circStream.getTracks().forEach(t => t.stop());
        _circStream = null;
    }
    const video = document.getElementById('videoCircular');
    if (video) video.srcObject = null;
}

// ─── Zone marking (Phase 2: called by test buttons; Phase 3: called by pose detector) ──
function circularMarkZone(zoneName) {
    const zone = CIRC_ZONES[zoneName];
    if (!zone || _circDone[zoneName]) return;
    _circDone[zoneName] = true;

    // Light up ticks for this zone
    zone.ticks.forEach(i => {
        const el = document.getElementById(`circ-tick-${i}`);
        if (el) {
            el.setAttribute('stroke', COLOR_DONE);
            el.setAttribute('stroke-width', CIRC_TICK_W + 1);  // slightly thicker when done
        }
    });

    const doneCount = Object.keys(_circDone).length;
    const total     = Object.keys(CIRC_ZONES).length;
    const progEl    = document.getElementById('circularProgress');
    if (progEl) progEl.textContent = `${doneCount}/${total} มุมเก็บแล้ว`;

    if (doneCount >= total) _onCircularComplete();
}

function _onCircularComplete() {
    const instr = document.getElementById('circularInstruction');
    const prog  = document.getElementById('circularProgress');
    const guide = document.getElementById('circFaceGuide');

    if (instr) { instr.textContent = '✓ สแกนครบทุกมุมแล้ว'; instr.style.color = '#16a34a'; }
    if (prog)  { prog.textContent  = '✓ ครบทุกมุม!';        prog.style.color  = '#16a34a'; }
    if (guide) {
        guide.setAttribute('stroke', COLOR_DONE);
        guide.setAttribute('stroke-width', '3');
        guide.setAttribute('stroke-dasharray', 'none');
    }
}

function circularReset() {
    Object.keys(_circDone).forEach(k => delete _circDone[k]);

    for (let i = 0; i < CIRC_NUM_TICKS; i++) {
        const el = document.getElementById(`circ-tick-${i}`);
        if (el) {
            el.setAttribute('stroke', COLOR_EMPTY);
            el.setAttribute('stroke-width', CIRC_TICK_W);
        }
    }

    const instr = document.getElementById('circularInstruction');
    const prog  = document.getElementById('circularProgress');
    const guide = document.getElementById('circFaceGuide');

    if (instr) { instr.textContent = 'หมุนหน้าช้าๆ เป็นวงกลม'; instr.style.color = ''; }
    if (prog)  { prog.textContent  = `0/${Object.keys(CIRC_ZONES).length} มุมเก็บแล้ว`; prog.style.color = ''; }
    if (guide) {
        guide.setAttribute('stroke', 'rgba(255,255,255,0.5)');
        guide.setAttribute('stroke-width', '2');
        guide.setAttribute('stroke-dasharray', '6 5');
    }
}

// ─── Override startLivenessChallenge ─────────────────────────────────────────
// enrollment_flow.js calls startLivenessChallenge() after EAR check completes.
// In circular mode we intercept it so the circular camera starts instead of
// the 2-action liveness challenge.
window.startLivenessChallenge = function () {
    console.log('[circular] intercepted startLivenessChallenge → startCircularCamera');
    startCircularCamera();
};

// ─── Boot ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', circularInit);
