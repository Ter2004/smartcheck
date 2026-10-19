// enrollment_v2.js — v2 face enrollment (FACE_ENGINE=arcface): consent → one camera
// session with no buttons → result. The browser scores frames and keeps the best
// ones per pose bucket; the server re-checks everything (app/services/enroll_v2.py).
//
// Gate values are the ones the v1 page uses (enrollment_flow.js): face height
// >= 0.30, centre 25-75% x / 20-80% y, frontal +-15 deg proxy, brightness >= 55,
// edge/centre <= 2.2, Laplacian variance >= 20 at 640x480. Turn: eyeYaw beyond
// 0.30 (server formula on MediaPipe points; 0.30 here reaches the server's turn
// limit for 94% of LFW faces, 3,870 faces, 2026-10-04). Blink: EAR below 0.70 x
// the open-eye baseline, then back (the rule the v1 blink action used).

const EV2 = {
    FACE_MIN_H: 0.30,
    CENTER_X: [0.25, 0.75], CENTER_Y: [0.20, 0.80],
    FRONTAL_MAX_DEG: 15,
    BRIGHT_MIN: 55, BACKLIGHT_MAX: 2.2, BLUR_MIN: 20,
    TURN_YAW: 0.30,
    BLINK_RATIO: 0.70,
    BLINK_PROMPT_MS: 3000,   // ask for a blink only if none is seen by then
    ALIGN_HOLD_MS: 600,      // face stays in the oval this long before the session starts
    LOOK_MS: 2000,           // frontal capture window
    TURN_TIMEOUT_MS: 15000,
    SAMPLE_MS: 200,          // frame scoring cadence
    KEEP_FRONTAL: 5, KEEP_TURN: 3,   // <= 8 frames posted; the server keeps 3 + 2
    SESSION_MAX_MS: 100000,  // below the server challenge TTL (120 s)
};
const EV2_TURN_LABELS = { turn_left: 'หันหน้าไปทางซ้ายช้า ๆ', turn_right: 'หันหน้าไปทางขวาช้า ๆ' };

// ─── Pure measurements (also run by scripts/test_enrollment_v2.js) ─────────────

/** liveness_challenge.yaw_ratio on MediaPipe points, in pixels. + = nose toward image right = turn_left. */
function ev2EyeYaw(lm, w, h) {
    const c = (i, j) => ({ x: (lm[i].x + lm[j].x) / 2 * w, y: (lm[i].y + lm[j].y) / 2 * h });
    let a = c(33, 133), b = c(362, 263);
    if (a.x > b.x) [a, b] = [b, a];
    const ax = b.x - a.x, ay = b.y - a.y, d = ax * ax + ay * ay;
    if (d < 1e-6) return 0;
    return ((lm[1].x * w - (a.x + b.x) / 2) * ax + (lm[1].y * h - (a.y + b.y) / 2) * ay) / d;
}

/** v1 _checkFrontal: yaw/pitch proxies from ears, forehead, chin and nose. */
function ev2Frontal(lm) {
    const faceW = lm[454].x - lm[234].x, faceH = lm[152].y - lm[10].y;
    if (faceW <= 0 || faceH <= 0) return false;
    const yaw = ((lm[1].x - lm[234].x) / faceW - 0.5) * 90;
    const pitch = ((lm[1].y - lm[10].y) / faceH - 0.5) * 90;
    return Math.abs(yaw) <= EV2.FRONTAL_MAX_DEG && Math.abs(pitch) <= EV2.FRONTAL_MAX_DEG;
}

/** Face height (fraction of the frame) and centre, from the landmark span. */
function ev2FaceBox(lm) {
    let minY = 1, maxY = 0;
    for (const p of lm) { if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y; }
    return { h: maxY - minY, cx: (lm[234].x + lm[454].x) / 2, cy: (lm[10].y + lm[152].y) / 2 };
}

function ev2InOval(box) {
    return box.h >= EV2.FACE_MIN_H && box.cx >= EV2.CENTER_X[0] && box.cx <= EV2.CENTER_X[1]
        && box.cy >= EV2.CENTER_Y[0] && box.cy <= EV2.CENTER_Y[1];
}

/** Eye aspect ratio, 6-point, averaged over both eyes, in pixels. */
function ev2Ear(lm, w, h) {
    const d = (i, j) => Math.hypot((lm[i].x - lm[j].x) * w, (lm[i].y - lm[j].y) * h);
    const eye = (p) => (d(p[1], p[5]) + d(p[2], p[4])) / (2 * Math.max(d(p[0], p[3]), 1e-6));
    return (eye([33, 160, 158, 133, 153, 144]) + eye([362, 385, 387, 263, 373, 380])) / 2;
}

/** Mouth opening (inner lips) over mouth width; lower = more closed. Used to rank, not to gate. */
function ev2Mar(lm, w, h) {
    const gap = Math.hypot((lm[13].x - lm[14].x) * w, (lm[13].y - lm[14].y) * h);
    const width = Math.hypot((lm[78].x - lm[308].x) * w, (lm[78].y - lm[308].y) * h);
    return width > 0 ? gap / width : 0;
}

/** v1 _checkCameraConditions on RGBA pixels of a 320x240 frame. */
function ev2Lighting(data, W = 320, H = 240) {
    let total = 0, n = 0, cSum = 0, cN = 0, eSum = 0, eN = 0;
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const i = (y * W + x) * 4;
            const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
            total += lum; n++;
            if (x > W * 0.25 && x < W * 0.75 && y > H * 0.25 && y < H * 0.75) { cSum += lum; cN++; }
            else { eSum += lum; eN++; }
        }
    }
    const brightness = n ? total / n : 0;
    const backlight = cN && cSum > 0 ? (eSum / eN) / (cSum / cN) : 0;
    return { brightness, backlight };
}

/** v1 _checkBlur: Laplacian variance of a 640x480 RGBA frame. */
function ev2Blur(data, W = 640, H = 480) {
    let sum = 0, sumSq = 0;
    const lum = (i) => 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    for (let y = 1; y < H - 1; y++) {
        for (let x = 1; x < W - 1; x++) {
            const i = (y * W + x) * 4;
            const v = lum(i - 4) + lum(i + 4) + lum(i - W * 4) + lum(i + W * 4) - 4 * lum(i);
            sum += v; sumSq += v * v;
        }
    }
    const count = (W - 2) * (H - 2);
    const mean = sum / count;
    return sumSq / count - mean * mean;
}

/** Blink: EAR drops below BLINK_RATIO x the open-eye baseline, then comes back. */
class Ev2BlinkDetector {
    constructor() { this.samples = []; this.down = false; this.blinked = false; }
    baseline() {
        if (this.samples.length < 5) return NaN;
        const s = [...this.samples].sort((a, b) => a - b);
        return s[Math.floor(s.length * 0.75)];   // upper quartile: eyes open
    }
    eyesOpen(ear) { const b = this.baseline(); return !Number.isFinite(b) || ear >= b * EV2.BLINK_RATIO; }
    push(ear) {
        if (!Number.isFinite(ear)) return this.blinked;
        const b = this.baseline();
        if (Number.isFinite(b)) {
            if (!this.down && ear < b * EV2.BLINK_RATIO) this.down = true;
            else if (this.down && ear >= b * EV2.BLINK_RATIO) { this.down = false; this.blinked = true; }
        }
        if (!this.down) { this.samples.push(ear); if (this.samples.length > 60) this.samples.shift(); }
        return this.blinked;
    }
}

/** Best `limit` frames by score. */
class Ev2Keeper {
    constructor(limit) { this.limit = limit; this.items = []; }
    wants(score) { return this.items.length < this.limit || score > this.items[this.items.length - 1].score; }
    add(item) {
        this.items.push(item);
        this.items.sort((a, b) => b.score - a.score);
        if (this.items.length > this.limit) this.items.pop();
    }
    get size() { return this.items.length; }
}

// ─── Page controller ──────────────────────────────────────────────────────────

const ev2 = {
    stream: null, fm: null, running: false, phase: 'idle', challenge: null, t0: 0,
    lastSample: 0, alignSince: 0, lookSince: 0, turnSince: 0,
    blink: null, frontal: null, turn: null, sending: false,
};

function $(id) { return document.getElementById(id); }
function ev2Csrf() { return document.querySelector('meta[name="csrf-token"]')?.content || ''; }
function ev2Show(id) { document.querySelectorAll('.ev2-screen').forEach(s => s.classList.toggle('active', s.id === id)); }
function ev2Say(text, hint = '') { $('ev2Instruction').textContent = text; $('ev2Hint').textContent = hint; }

function ev2Ring(fraction, ok = false) {
    const ring = $('ev2Ring');
    const len = ring.getTotalLength ? ring.getTotalLength() : 760;
    ring.style.strokeDasharray = `${len}`;
    ring.style.strokeDashoffset = `${len * (1 - Math.max(0, Math.min(1, fraction)))}`;
    ring.classList.toggle('ok', ok);
}

async function ev2Post(url, body) {
    const res = await fetch(url, {
        method: 'POST', headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': ev2Csrf() },
        body: JSON.stringify(body || {}),
    });
    let json = {};
    try { json = await res.json(); } catch (_) { /* non-JSON error page */ }
    return { status: res.status, json };
}

async function ev2Start() {
    $('ev2Start').disabled = true;
    $('ev2ConsentError').textContent = '';
    try {
        const consent = await ev2Post(ENROLL_V2_CONFIG.consentUrl);
        if (consent.status !== 200) throw new Error('consent');
        // Same user gesture: the browser asks for camera permission now.
        ev2.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 640, height: 480 } });
        const vc = await detectVirtualCamera(ev2.stream);
        if (vc.blocked) {
            ev2.stream.getTracks().forEach(t => t.stop());
            throw new Error(`ไม่อนุญาตให้ใช้กล้องเสมือน (${vc.label}) — กรุณาใช้กล้องจริง`);
        }
        $('ev2Video').srcObject = ev2.stream;
        ev2Show('ev2Camera');
        await ev2NewSession();
    } catch (e) {
        $('ev2Start').disabled = !$('ev2ConsentCheck').checked;
        $('ev2ConsentError').textContent = e.message && e.message !== 'consent'
            ? (e.name === 'NotAllowedError' ? 'กรุณาอนุญาตให้ใช้กล้องแล้วลองใหม่' : e.message)
            : 'บันทึกความยินยอมไม่สำเร็จ กรุณาลองใหม่';
    }
}

async function ev2NewSession() {
    ev2Say('กำลังเตรียม...');
    ev2Ring(0);
    const r = await ev2Post(ENROLL_V2_CONFIG.challengeUrl);
    if (r.status !== 200 || !r.json.challenge_id) { ev2Fail({ fix: 'เริ่มไม่สำเร็จ กรุณาลองใหม่' }); return; }
    Object.assign(ev2, {
        challenge: r.json, t0: performance.now(), phase: 'align', lastSample: 0,
        alignSince: 0, lookSince: 0, turnSince: 0, sending: false,
        blink: new Ev2BlinkDetector(), frontal: new Ev2Keeper(EV2.KEEP_FRONTAL), turn: new Ev2Keeper(EV2.KEEP_TURN),
    });
    ev2Say('จัดใบหน้าให้อยู่ในกรอบวงรี');
    if (!ev2.fm) {
        ev2.fm = new FaceMesh({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/${f}` });
        ev2.fm.setOptions({ maxNumFaces: 2, refineLandmarks: false, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
        ev2.fm.onResults(ev2OnResults);
    }
    if (!ev2.running) { ev2.running = true; ev2Loop(); }
}

async function ev2Loop() {
    const video = $('ev2Video');
    while (ev2.running) {
        if (video.readyState >= 2 && ev2.phase !== 'idle') {
            try { await ev2.fm.send({ image: video }); } catch (_) { /* next frame */ }
        }
        await new Promise(r => requestAnimationFrame(r));
    }
}

/** Score the current video frame; returns null when a gate fails (with the reason). */
function ev2Score(video, lm) {
    const c = $('ev2Canvas');
    c.width = 320; c.height = 240;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(video, 0, 0, 320, 240);
    const light = ev2Lighting(ctx.getImageData(0, 0, 320, 240).data);
    if (light.brightness < EV2.BRIGHT_MIN) return { fail: 'แสงน้อยเกินไป — หันหน้าเข้าหาแสง' };
    if (light.backlight > EV2.BACKLIGHT_MAX) return { fail: 'แสงจ้าด้านหลัง — หันหน้าเข้าหาแสง' };
    c.width = 640; c.height = 480;
    ctx.drawImage(video, 0, 0, 640, 480);
    const blur = ev2Blur(ctx.getImageData(0, 0, 640, 480).data);
    if (blur < EV2.BLUR_MIN) return { fail: 'ภาพเบลอ — อยู่นิ่ง ๆ' };
    const w = video.videoWidth || 640, h = video.videoHeight || 480;
    return { score: blur * (1 - Math.min(ev2Mar(lm, w, h), 0.5)) };
}

function ev2Grab(video) {
    const c = $('ev2Canvas');
    c.width = video.videoWidth; c.height = video.videoHeight;
    c.getContext('2d').drawImage(video, 0, 0);
    return c.toDataURL('image/jpeg', 0.88);
}

function ev2OnResults(results) {
    if (ev2.sending || ev2.phase === 'idle') return;
    const video = $('ev2Video');
    const now = performance.now();
    if (now - ev2.t0 > EV2.SESSION_MAX_MS) { ev2Fail({ fix: 'หมดเวลา — กรุณาลองใหม่' }); return; }
    const faces = results.multiFaceLandmarks || [];
    if (faces.length !== 1) {
        ev2.alignSince = 0;
        ev2Ring(ev2.phase === 'align' ? 0 : ev2Progress(now));
        ev2Say(faces.length ? 'ให้มีเพียงคุณคนเดียวในกล้อง' : 'จัดใบหน้าให้อยู่ในกรอบวงรี');
        return;
    }
    const lm = faces[0];
    const w = video.videoWidth || 640, h = video.videoHeight || 480;
    const box = ev2FaceBox(lm);
    if (!ev2InOval(box)) {
        ev2.alignSince = 0;
        ev2Say(box.h < EV2.FACE_MIN_H ? 'เข้าใกล้กล้องอีกหน่อย' : 'จัดใบหน้าให้อยู่กลางกรอบ');
        if (ev2.phase === 'align') ev2Ring(0);
        return;
    }
    const ear = ev2Ear(lm, w, h);
    const blinked = ev2.blink.push(ear);

    if (ev2.phase === 'align') {
        if (!ev2Frontal(lm)) { ev2.alignSince = 0; ev2Say('มองตรงเข้ากล้อง'); ev2Ring(0); return; }
        ev2.alignSince = ev2.alignSince || now;
        const held = (now - ev2.alignSince) / EV2.ALIGN_HOLD_MS;
        ev2Ring(held, true);
        ev2Say('ดีมาก — อยู่นิ่ง ๆ');
        if (held >= 1) { ev2.phase = 'look'; ev2.lookSince = now; ev2Ring(0); }
        return;
    }

    if (now - ev2.lastSample < EV2.SAMPLE_MS) return;
    ev2.lastSample = now;
    const t = Math.round(now - ev2.t0);
    const yaw = ev2EyeYaw(lm, w, h);

    if (ev2.phase === 'look') {
        const elapsed = now - ev2.lookSince;
        ev2Say('มองกล้องค้างไว้', !blinked && elapsed > EV2.BLINK_PROMPT_MS ? 'กะพริบตา 1 ครั้ง' : '');
        ev2Ring(ev2Progress(now), true);
        if (ev2Frontal(lm) && ev2.blink.eyesOpen(ear)) {
            const s = ev2Score(video, lm);
            if (s.fail) ev2Say(s.fail);
            else if (ev2.frontal.wants(s.score)) {
                ev2.frontal.add({ score: s.score, frame: { image: ev2Grab(video), t, pose: 'frontal', client_yaw: +yaw.toFixed(3) } });
            }
        }
        if (elapsed >= EV2.LOOK_MS && blinked && ev2.frontal.size >= 3) {
            ev2.phase = 'turn'; ev2.turnSince = now;
        }
        return;
    }

    if (ev2.phase === 'turn') {
        const want = ev2.challenge.direction;
        ev2Say(EV2_TURN_LABELS[want]);
        ev2Ring(ev2Progress(now), true);
        const turned = want === 'turn_left' ? yaw > EV2.TURN_YAW : yaw < -EV2.TURN_YAW;
        if (turned) {
            const s = ev2Score(video, lm);
            if (!s.fail && ev2.turn.wants(s.score)) {
                ev2.turn.add({ score: s.score, frame: { image: ev2Grab(video), t, pose: 'turn', client_yaw: +yaw.toFixed(3) } });
            }
        }
        if (ev2.turn.size >= EV2.KEEP_TURN) { ev2Submit(); return; }
        if (now - ev2.turnSince > EV2.TURN_TIMEOUT_MS) ev2Fail({ fix: 'ไม่พบการหันหน้า — ' + EV2_TURN_LABELS[want] });
    }
}

function ev2Progress(now) {
    if (ev2.phase === 'look') return 0.5 * Math.min(1, (now - ev2.lookSince) / EV2.LOOK_MS);
    if (ev2.phase === 'turn') return 0.5 + 0.5 * (ev2.turn.size / EV2.KEEP_TURN);
    return 0;
}

async function ev2Submit() {
    ev2.sending = true;
    ev2.phase = 'idle';
    ev2Ring(1, true);
    ev2Show('ev2Result');
    $('ev2Checking').hidden = false; $('ev2Success').hidden = true; $('ev2Failure').hidden = true;
    const frames = [...ev2.frontal.items, ...ev2.turn.items].map(x => x.frame).sort((a, b) => a.t - b.t);
    let r;
    try {
        r = await ev2Post(ENROLL_V2_CONFIG.enrollUrl, { challenge_id: ev2.challenge.challenge_id, frames });
    } catch (_) {
        ev2Fail({ fix: 'เชื่อมต่อเซิร์ฟเวอร์ไม่ได้ กรุณาลองใหม่' });
        return;
    }
    if (r.json.status === 'enrolled' || r.json.status === 'already_enrolled') {
        ev2Stop();
        $('ev2Checking').hidden = true; $('ev2Success').hidden = false;
        return;
    }
    if (r.status === 429) { ev2Fail({ fix: 'ลองบ่อยเกินไป — รอสักครู่แล้วลองใหม่' }); return; }
    ev2Fail({ fix: r.json.fix || r.json.message || 'ลงทะเบียนไม่สำเร็จ', final: !!r.json.contact_staff,
              message: r.json.contact_staff ? r.json.message : '' });
}

function ev2Fail({ fix, final = false, message = '' }) {
    ev2.phase = 'idle';
    ev2.sending = false;
    ev2Show('ev2Result');
    $('ev2Checking').hidden = true; $('ev2Success').hidden = true; $('ev2Failure').hidden = false;
    $('ev2Fix').textContent = final ? message : fix;
    $('ev2Retry').hidden = final;
    $('ev2Home').hidden = !final;
    if (final) ev2Stop();
}

function ev2Stop() {
    ev2.running = false;
    if (ev2.stream) { ev2.stream.getTracks().forEach(t => t.stop()); ev2.stream = null; }
}

if (typeof document !== 'undefined' && document.getElementById('ev2Consent')) {
    $('ev2ConsentCheck').addEventListener('change', e => { $('ev2Start').disabled = !e.target.checked; });
    $('ev2Start').addEventListener('click', ev2Start);
    $('ev2Retry').addEventListener('click', async () => {
        ev2Show('ev2Camera');
        if (!ev2.stream) {
            try {
                ev2.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 640, height: 480 } });
                $('ev2Video').srcObject = ev2.stream;
            } catch (_) { ev2Fail({ fix: 'เปิดกล้องไม่ได้ กรุณาอนุญาตการใช้กล้อง' }); return; }
        }
        await ev2NewSession();   // no re-consent: the consent row from screen 1 still stands
    });
    window.addEventListener('pagehide', ev2Stop);
}
