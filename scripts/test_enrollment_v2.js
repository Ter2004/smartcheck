// node scripts/test_enrollment_v2.js — pure measurements of app/static/js/enrollment_v2.js
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const src = fs.readFileSync(path.join(__dirname, '../app/static/js/enrollment_v2.js'), 'utf8');
const ctx = vm.createContext({ console, Math, performance: { now: () => 0 } });
vm.runInContext(src + '\nglobalThis.api = {EV2, ev2EyeYaw, ev2Frontal, ev2FaceBox, ev2InOval, ev2Ear, ev2Mar, ev2Lighting, ev2Blur, Ev2BlinkDetector, Ev2Keeper};', ctx);
const A = ctx.api;

// A synthetic frontal face: eye corners, nose, ears, forehead, chin, lids and lips.
function face({ noseX = 0.5, earOpen = 0.30, top = 0.25, bottom = 0.75 } = {}) {
    const lm = Array.from({ length: 468 }, () => ({ x: 0.5, y: 0.5 }));
    const set = (i, x, y) => { lm[i] = { x, y }; };
    set(33, 0.40, 0.45); set(133, 0.46, 0.45); set(362, 0.54, 0.45); set(263, 0.60, 0.45);
    set(1, noseX, 0.55); set(234, 0.35, 0.5); set(454, 0.65, 0.5); set(10, 0.5, top); set(152, 0.5, bottom);
    // eyelids: vertical gap = earOpen x eye width (0.06 wide in x at 640 px, scale y to 480)
    const gap = earOpen * 0.06 * 640 / 480;
    for (const [p1, p2, p4, p5, cx] of [[160, 158, 153, 144, 0.43], [385, 387, 373, 380, 0.57]]) {
        set(p1, cx - 0.01, 0.45 - gap / 2); set(p2, cx + 0.01, 0.45 - gap / 2);
        set(p4, cx + 0.01, 0.45 + gap / 2); set(p5, cx - 0.01, 0.45 + gap / 2);
    }
    set(13, 0.5, 0.65); set(14, 0.5, 0.651); set(78, 0.46, 0.65); set(308, 0.54, 0.65);
    return lm;
}
const near = (a, b, tol = 1e-6) => Math.abs(a - b) < tol;

// Yaw: same formula and sign as liveness_challenge.yaw_ratio (+ = nose to image right = turn_left).
assert.ok(near(A.ev2EyeYaw(face(), 640, 480), 0), 'frontal yaw 0');
assert.ok(A.ev2EyeYaw(face({ noseX: 0.56 }), 640, 480) > A.EV2.TURN_YAW, 'nose right -> turn_left beyond 0.30');
assert.ok(A.ev2EyeYaw(face({ noseX: 0.44 }), 640, 480) < -A.EV2.TURN_YAW, 'nose left -> turn_right');

// Frontal gate (+-15 deg proxy) and the oval (height >= 0.30, centred).
assert.equal(A.ev2Frontal(face()), true);
assert.equal(A.ev2Frontal(face({ noseX: 0.62 })), false);
assert.equal(A.ev2InOval(A.ev2FaceBox(face())), true);
assert.equal(A.ev2InOval(A.ev2FaceBox(face({ top: 0.45, bottom: 0.70 }))), false, 'too small');

// EAR: open eyes read the configured ratio; blink = drop below 0.70 x baseline, then back.
assert.ok(near(A.ev2Ear(face({ earOpen: 0.30 }), 640, 480), 0.30, 1e-3), 'EAR');
const blink = new A.Ev2BlinkDetector();
for (let i = 0; i < 10; i++) blink.push(0.30);
assert.equal(blink.blinked, false);
assert.equal(blink.eyesOpen(0.15), false, 'closed eyes are not used for capture');
blink.push(0.12);                       // eyes close
assert.equal(blink.blinked, false);
blink.push(0.29);                       // and open again
assert.equal(blink.blinked, true, 'blink detected');
const noBlink = new A.Ev2BlinkDetector();
for (let i = 0; i < 30; i++) noBlink.push(0.30 + (i % 2) * 0.01);
assert.equal(noBlink.blinked, false, 'small jitter is not a blink');

// Keeper keeps the best N by score.
const k = new A.Ev2Keeper(3);
[5, 1, 9, 7, 3].forEach(s => { if (k.wants(s)) k.add({ score: s }); });
assert.equal(JSON.stringify(k.items.map(i => i.score)), "[9,7,5]");   // vm-realm array
assert.equal(k.wants(4), false);

// Lighting and blur on synthetic RGBA frames.
const flat = (W, H, v) => { const d = new Uint8ClampedArray(W * H * 4); for (let i = 0; i < d.length; i += 4) { d[i] = d[i + 1] = d[i + 2] = v; d[i + 3] = 255; } return d; };
assert.ok(near(A.ev2Lighting(flat(320, 240, 40)).brightness, 40, 1e-6), 'dark frame');
const backlit = flat(320, 240, 220);
for (let y = 61; y < 180; y++) for (let x = 81; x < 240; x++) { const i = (y * 320 + x) * 4; backlit[i] = backlit[i + 1] = backlit[i + 2] = 60; }
assert.ok(A.ev2Lighting(backlit).backlight > A.EV2.BACKLIGHT_MAX, 'bright edges, dark centre');
assert.ok(A.ev2Blur(flat(640, 480, 128)) < A.EV2.BLUR_MIN, 'flat frame is blurry');
const sharp = flat(640, 480, 0);
for (let y = 0; y < 480; y++) for (let x = 0; x < 640; x++) if ((x + y) % 2) { const i = (y * 640 + x) * 4; sharp[i] = sharp[i + 1] = sharp[i + 2] = 255; }
assert.ok(A.ev2Blur(sharp) > A.EV2.BLUR_MIN, 'checkerboard is sharp');

console.log('test_enrollment_v2.js: all assertions passed');
