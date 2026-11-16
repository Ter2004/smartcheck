// enrollment_v2.js — v2 face enrollment (FACE_ENGINE=arcface): consent → one camera
// session with no buttons → result. Five poses, each captured automatically when
// the face reaches it: frontal first, then left / right / up / down in the random
// order the server issued. The browser scores frames and keeps the best ones per
// pose; the server re-checks everything (app/services/enroll_v2.py).
//
// Gate values are the ones the v1 page uses (enrollment_flow.js): face height
// >= 0.30, centre 25-75% x / 20-80% y, frontal +-15 deg proxy, brightness >= 55,
// edge/centre <= 2.2, Laplacian variance >= 20 at 640x480. Left/right: eyeYaw
// beyond 0.30 (server formula on MediaPipe points; 0.30 here reaches the server's
// turn limit for 94% of LFW faces, 3,870 faces, 2026-10-04). Up/down: the v1 nose
// ratio moves 0.07 from this person's own frontal baseline (the top of the v1 nod
// gesture's range; 8.7-13.3 deg of server pitch on 1,500 LFW faces,
// docs/face-stack-v2/client-pitch-calibration.json) while yaw stays inside the
// frontal gate, because the server checks up/down frames as frontal-yaw frames.
// Blink: EAR below 0.70 x the open-eye baseline, then back (the rule the v1
// blink action used).

const EV2 = {
    FACE_MIN_H: 0.30,
    FACE_BIG_H: 0.90,        // hint only ("ถอยออกหน่อย"); the server copes with close-ups
    CENTER_X: [0.25, 0.75], CENTER_Y: [0.20, 0.80],
    FRONTAL_MAX_DEG: 15,
    PITCH_STEP: 0.07,        // up/down: nose-ratio change from the frontal baseline
    BRIGHT_MIN: 55, BACKLIGHT_MAX: 2.2, BLUR_MIN: 20,
    TURN_YAW: 0.30,
    BLINK_RATIO: 0.70,
    BLINK_PROMPT_MS: 3000,   // ask for a blink only if none is seen by then
    ALIGN_HOLD_MS: 600,      // face stays in the oval this long before the session starts
    LOOK_MS: 2000,           // frontal capture window
    POSE_TIMEOUT_MS: 12000,
    DONE_FLASH_MS: 450,      // the ✅ after each pose
    SAMPLE_MS: 200,          // frame scoring cadence
    KEEP_FRONTAL: 4, KEEP_POSE: 2,   // <= 12 frames posted; the server keeps 3 + 1 per pose
    SESSION_MAX_MS: 100000,  // below the server challenge TTL (120 s)
};
const EV2_POSE_TEXT = {
    left: '⬅️ หันซ้ายนิดนึง', right: '➡️ หันขวานิดนึง', up: '⬆️ เงยนิดนึง', down: '⬇️ ก้มนิดนึง',
};

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

/** v1 yaw proxy (enrollment_flow.js _checkFrontal): nose between the face edges 234 and 454. */
function ev2YawProxy(lm) {
    const faceW = lm[454].x - lm[234].x;
    return faceW > 0 ? ((lm[1].x - lm[234].x) / faceW - 0.5) * 90 : 90;
}

/** v1 nose ratio (mediapipe_liveness.js nosePitch): nose between forehead 10 and chin 152; smaller = looking up. */
function ev2NoseRatio(lm) {
    const faceH = lm[152].y - lm[10].y;
    return faceH > 0 ? (lm[1].y - lm[10].y) / faceH : 0.5;
}

/** v1 _checkFrontal: yaw and pitch proxies within +-15. */
function ev2Frontal(lm) {
    if (lm[152].y - lm[10].y <= 0) return false;
    return Math.abs(ev2YawProxy(lm)) <= EV2.FRONTAL_MAX_DEG
        && Math.abs((ev2NoseRatio(lm) - 0.5) * 90) <= EV2.FRONTAL_MAX_DEG;
}

/** Has the face reached `pose`? m = {eyeYaw, yawProxy, up}; up = baseline nose ratio - current (+ = looking up). */
function ev2PoseReached(pose, m) {
    if (pose === 'left') return m.eyeYaw > EV2.TURN_YAW;
    if (pose === 'right') return m.eyeYaw < -EV2.TURN_YAW;
    if (Math.abs(m.yawProxy) > EV2.FRONTAL_MAX_DEG) return false;   // up/down keep a frontal yaw
    if (pose === 'up') return m.up >= EV2.PITCH_STEP;
    if (pose === 'down') return m.up <= -EV2.PITCH_STEP;
    return false;
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

function ev2Median(values) {
    if (!values.length) return 0;
    const s = [...values].sort((a, b) => a - b);
    return s[Math.floor(s.length / 2)];
}

// ─── Page controller ──────────────────────────────────────────────────────────

const ev2 = {
    stream: null, fm: null, running: false, phase: 'idle', challenge: null, order: [], t0: 0,
    lastSample: 0, alignSince: 0, lookSince: 0, poseIdx: 0, poseSince: 0, flashUntil: 0,
    blink: null, frontal: null, poses: {}, ratioSamples: [], ratioBase: 0.5, sending: false, audio: null,
};

function $(id) { return document.getElementById(id); }
function ev2Csrf() { return document.querySelector('meta[name="csrf-token"]')?.content || ''; }
function ev2Show(id) { document.querySelectorAll('.ev2-screen').forEach(s => s.classList.toggle('active', s.id === id)); }
function ev2Say(text, hint = '') { $('ev2Instruction').textContent = text; $('ev2Hint').textContent = hint; }

// Five ring segments: frontal, then the challenge's poses in order.
function ev2Segments(states) {
    const segs = document.querySelectorAll('.ev2-seg');
    segs.forEach((seg, i) => {
        const len = seg.getTotalLength ? seg.getTotalLength() : 760;
        const part = len / segs.length, gap = 10;
        seg.style.strokeDasharray = `${part - gap} ${len - part + gap}`;
        seg.style.strokeDashoffset = `${-(i * part)}`;
        seg.setAttribute('class', `ev2-seg ${states[i] || 'todo'}`);
    });
}

function ev2SegmentStates() {
    const done = ev2.phase === 'pose' ? 1 + ev2.poseIdx : ev2.phase === 'submit' ? 5 : 0;
    return [0, 1, 2, 3, 4].map(i => i < done ? 'done' : i === done && ev2.phase !== 'align' ? 'active' : 'todo');
}

function ev2Arrow(pose) {
    document.querySelectorAll('.ev2-arrow').forEach(a => a.classList.toggle('show', a.dataset.pose === pose));
}

// Feedback when a pose is captured: a short tone (unlocked by the start tap) and a vibration (Android).
function ev2Reward() {
    try {
        if (ev2.audio) {
            const t = ev2.audio.currentTime, o = ev2.audio.createOscillator(), g = ev2.audio.createGain();
            o.frequency.value = 880;
            g.gain.setValueAtTime(0.0001, t);
            g.gain.exponentialRampToValueAtTime(0.25, t + 0.01);
            g.gain.exponentialRampToValueAtTime(0.0001, t + 0.15);
            o.connect(g).connect(ev2.audio.destination);
            o.start(t); o.stop(t + 0.16);
        }
    } catch (_) { /* sound is optional */ }
    try { navigator.vibrate?.(40); } catch (_) { /* iOS Safari has no vibration */ }
}

function ev2UnlockAudio() {
    try {
        ev2.audio = ev2.audio || new (window.AudioContext || window.webkitAudioContext)();
        if (ev2.audio.state === 'suspended') ev2.audio.resume();
    } catch (_) { ev2.audio = null; }
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
    ev2UnlockAudio();                     // inside the tap: iOS only allows audio after a gesture
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
        $('ev2ConsentError').textContent = e.message === 'consent'
            ? 'บันทึกการยินยอมไม่สำเร็จ (ติดต่อเซิร์ฟเวอร์ไม่ได้ หรือหลุดจากระบบ) — รีเฟรชหน้าแล้วลองใหม่'
            : ev2CameraError(e);
    }
}

/** What went wrong with the camera, in words a student can act on. */
function ev2CameraError(e) {
    switch (e && e.name) {
        case 'NotAllowedError': case 'SecurityError':
            return 'เบราว์เซอร์ไม่อนุญาตให้ใช้กล้อง — กดรูปกล้อง/กุญแจที่ช่องที่อยู่เว็บ เลือก "อนุญาต" แล้วลองใหม่';
        case 'NotFoundError': case 'OverconstrainedError':
            return 'ไม่พบกล้องในเครื่องนี้ — ใช้เครื่องที่มีกล้องหน้า หรือเสียบกล้องแล้วลองใหม่';
        case 'NotReadableError': case 'AbortError':
            return 'กล้องถูกโปรแกรมอื่นใช้อยู่ (เช่น Zoom, Teams, LINE) — ปิดโปรแกรมนั้นแล้วลองใหม่';
        default:
            return (e && e.message) || 'เปิดกล้องไม่ได้ — รีเฟรชหน้าแล้วลองใหม่';
    }
}

async function ev2NewSession() {
    ev2Say('กำลังเตรียม...');
    ev2Arrow(null);
    const r = await ev2Post(ENROLL_V2_CONFIG.challengeUrl);
    const order = r.json && Array.isArray(r.json.order) ? r.json.order : null;
    if (r.status !== 200 || !r.json.challenge_id || !order || order.length !== 4) {
        if (r.status === 409) { ev2Fail({ final: true, message: 'บัญชีนี้ลงทะเบียนใบหน้าไว้แล้ว ไม่ต้องลงซ้ำ' }); return; }
        if (r.status === 429) { ev2Fail({ fix: 'กดเริ่มบ่อยเกินไป — รอประมาณ 1 นาทีแล้วกดลองใหม่' }); return; }
        ev2Fail({ fix: r.json.message || `เริ่มการถ่ายไม่สำเร็จ (รหัส ${r.status}) — รีเฟรชหน้าแล้วลองใหม่` });
        return;
    }
    Object.assign(ev2, {
        challenge: r.json, order, t0: performance.now(), phase: 'align', lastSample: 0,
        alignSince: 0, lookSince: 0, poseIdx: 0, poseSince: 0, flashUntil: 0, sending: false,
        blink: new Ev2BlinkDetector(), frontal: new Ev2Keeper(EV2.KEEP_FRONTAL),
        poses: Object.fromEntries(order.map(p => [p, new Ev2Keeper(EV2.KEEP_POSE)])),
        ratioSamples: [], ratioBase: 0.5,
    });
    ev2Segments(ev2SegmentStates());
    ev2Say('👤 วางหน้าในวงรี');
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

/** Score the current video frame; returns {fail} when a lighting or blur gate fails. */
function ev2Score(video, lm) {
    const c = $('ev2Canvas');
    c.width = 320; c.height = 240;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(video, 0, 0, 320, 240);
    const light = ev2Lighting(ctx.getImageData(0, 0, 320, 240).data);
    if (light.brightness < EV2.BRIGHT_MIN) return { fail: 'แสงน้อยเกินไป — หาที่สว่างกว่านี้' };
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
    c.getContext('2d').drawImage(video, 0, 0);     // the raw (unmirrored) frame
    return c.toDataURL('image/jpeg', 0.88);
}

function ev2OnResults(results) {
    if (ev2.sending || ev2.phase === 'idle' || ev2.phase === 'submit') return;
    const video = $('ev2Video');
    const now = performance.now();
    if (now - ev2.t0 > EV2.SESSION_MAX_MS) {
        ev2Fail({ fix: 'หมดเวลา (ใช้เวลาถ่ายเกิน 100 วินาที) — กดลองใหม่ แล้วทำตามลูกศรต่อเนื่องจนจบ' });
        return;
    }
    if (now < ev2.flashUntil) return;                        // showing ✅
    const faces = results.multiFaceLandmarks || [];
    if (faces.length !== 1) {
        ev2.alignSince = 0;
        ev2Say(faces.length ? 'เจอหลายหน้า — ให้อยู่คนเดียว' : '👤 วางหน้าในวงรี');
        return;
    }
    const lm = faces[0];
    const w = video.videoWidth || 640, h = video.videoHeight || 480;
    const box = ev2FaceBox(lm);
    if (!ev2InOval(box)) {
        ev2.alignSince = 0;
        ev2Say(box.h < EV2.FACE_MIN_H ? 'ขยับเข้าใกล้กล้องอีกนิด' : 'เลื่อนหน้าให้อยู่กลางวงรี');
        return;
    }
    const tooBig = box.h > EV2.FACE_BIG_H ? 'ถอยออกหน่อย' : '';
    const ear = ev2Ear(lm, w, h);
    const blinked = ev2.blink.push(ear);

    if (ev2.phase === 'align') {
        if (!ev2Frontal(lm)) { ev2.alignSince = 0; ev2Say('มองตรง', tooBig); return; }
        ev2.alignSince = ev2.alignSince || now;
        ev2Say('มองตรง — อยู่นิ่ง ๆ', tooBig);
        if (now - ev2.alignSince >= EV2.ALIGN_HOLD_MS) {
            ev2.phase = 'look'; ev2.lookSince = now;
            ev2Segments(ev2SegmentStates());
        }
        return;
    }

    if (now - ev2.lastSample < EV2.SAMPLE_MS) return;
    ev2.lastSample = now;
    const t = Math.round(now - ev2.t0);
    const yaw = ev2EyeYaw(lm, w, h);
    const ratio = ev2NoseRatio(lm);

    if (ev2.phase === 'look') {
        const elapsed = now - ev2.lookSince;
        ev2Say('มองตรง', !blinked && elapsed > EV2.BLINK_PROMPT_MS ? 'กะพริบตา 1 ครั้ง' : tooBig);
        if (ev2Frontal(lm) && ev2.blink.eyesOpen(ear)) {
            ev2.ratioSamples.push(ratio);
            const s = ev2Score(video, lm);
            if (s.fail) ev2Say(s.fail);
            else if (ev2.frontal.wants(s.score)) {
                ev2.frontal.add({ score: s.score, frame: { image: ev2Grab(video), t, pose: 'frontal', client_yaw: +yaw.toFixed(3) } });
            }
        }
        if (elapsed >= EV2.LOOK_MS && blinked && ev2.frontal.size >= 3) {
            ev2.ratioBase = ev2Median(ev2.ratioSamples);
            ev2.phase = 'pose'; ev2.poseIdx = 0; ev2.poseSince = now;
            ev2.flashUntil = now + EV2.DONE_FLASH_MS;
            ev2Reward();
            ev2Say('✅');
            ev2Segments(ev2SegmentStates());
        }
        return;
    }

    // phase 'pose'
    const pose = ev2.order[ev2.poseIdx];
    ev2Arrow(pose);
    ev2Say(EV2_POSE_TEXT[pose], tooBig);
    const up = ev2.ratioBase - ratio;
    const eyesOk = pose === 'up' || pose === 'down' || ev2.blink.eyesOpen(ear);   // eyelids drop when looking down
    if (ev2PoseReached(pose, { eyeYaw: yaw, yawProxy: ev2YawProxy(lm), up }) && eyesOk) {
        const s = ev2Score(video, lm);
        if (s.fail) ev2Say(s.fail);
        else if (ev2.poses[pose].wants(s.score)) {
            // client_pitch: nose-ratio change from the frontal baseline, + = up (the server's pitch sign)
            ev2.poses[pose].add({ score: s.score, frame: { image: ev2Grab(video), t, pose,
                client_yaw: +yaw.toFixed(3), client_pitch: +up.toFixed(3) } });
        }
    }
    if (ev2.poses[pose].size >= EV2.KEEP_POSE) {
        ev2Reward();
        ev2.flashUntil = now + EV2.DONE_FLASH_MS;
        ev2.poseIdx += 1; ev2.poseSince = now + EV2.DONE_FLASH_MS;
        ev2Arrow(null);
        if (ev2.poseIdx >= ev2.order.length) { ev2Submit(); return; }
        ev2Say('✅');
        ev2Segments(ev2SegmentStates());
        return;
    }
    if (now - ev2.poseSince > EV2.POSE_TIMEOUT_MS) {
        const tip = pose === 'up' || pose === 'down' ? ' โดยหน้ายังตรง ไม่หันข้าง' : '';
        ev2Fail({ fix: `ไม่เห็นท่า "${EV2_POSE_TEXT[pose]}" ภายใน 12 วินาที — กดลองใหม่ แล้วทำท่าให้ชัดขึ้นอีกนิด${tip}` });
    }
}

// Server checks take a few seconds (4-11 s in the first PC test); the bar is an estimate, not real progress.
const EV2_BAR_TAU_MS = 3000;
function ev2BarStart() {
    ev2BarStop();
    const t0 = performance.now(), bar = $('ev2Bar');
    bar.style.width = '0%';
    ev2.barTimer = setInterval(() => {
        bar.style.width = `${(90 * (1 - Math.exp(-(performance.now() - t0) / EV2_BAR_TAU_MS))).toFixed(1)}%`;
    }, 200);
}
function ev2BarStop(done = false) {
    if (ev2.barTimer) { clearInterval(ev2.barTimer); ev2.barTimer = null; }
    if (done) $('ev2Bar').style.width = '100%';
}

async function ev2Submit() {
    ev2.sending = true;
    ev2.phase = 'submit';
    ev2Segments(ev2SegmentStates());
    ev2Say('🎉 เสร็จแล้ว!');
    ev2Show('ev2Result');
    ev2Pop(false);
    $('ev2Checking').hidden = false; $('ev2Success').hidden = true; $('ev2Failure').hidden = true;
    ev2BarStart();
    const frames = [...ev2.frontal.items, ...ev2.order.flatMap(p => ev2.poses[p].items)]
        .map(x => x.frame).sort((a, b) => a.t - b.t);
    let r;
    try {
        r = await ev2Post(ENROLL_V2_CONFIG.enrollUrl, { challenge_id: ev2.challenge.challenge_id, frames });
    } catch (_) {
        ev2Fail({ fix: 'ส่งรูปไปตรวจไม่สำเร็จ (อินเทอร์เน็ตหลุด หรือเซิร์ฟเวอร์ปิดอยู่) — ตรวจเน็ตแล้วกดลองใหม่' });
        return;
    }
    if (r.json.status === 'enrolled' || r.json.status === 'already_enrolled') {
        ev2BarStop(true);
        ev2Stop();
        $('ev2Checking').hidden = true; $('ev2Success').hidden = false;
        return;
    }
    if (r.status === 429) { ev2Fail({ fix: 'ส่งบ่อยเกินไป ระบบให้พักก่อน — รอประมาณ 1 นาทีแล้วกดลองใหม่' }); return; }
    const left = Number.isInteger(r.json.attempts_left) ? ` (ลองได้อีก ${r.json.attempts_left} ครั้ง)` : '';
    const reason = r.json.fix || r.json.message
        || `ลงทะเบียนไม่สำเร็จ (รหัส ${r.status}) — กดลองใหม่ ถ้ายังไม่ได้ให้แจ้งผู้ดูแลระบบพร้อมรหัสนี้`;
    // After the last allowed try, say both what to do now and what went wrong this time.
    const final = !!r.json.contact_staff;
    const last = final && r.json.fix && r.json.fix !== r.json.message ? `\nสาเหตุครั้งล่าสุด: ${r.json.fix}` : '';
    ev2Fail({ fix: reason + left, final, message: final ? r.json.message + last : '' });
}

function ev2Fail({ fix, final = false, message = '' }) {
    ev2BarStop();
    ev2.phase = 'idle';
    ev2.sending = false;
    ev2Arrow(null);
    // The failure pops up over the screen the student was on (owner, 2026-11-16); the
    // checking screen is swapped back to the camera so the pop-up has something behind it.
    const result = $('ev2Result');
    if (!result || result.classList.contains('active')) ev2Show('ev2Camera');
    $('ev2Checking').hidden = true; $('ev2Success').hidden = true; $('ev2Failure').hidden = false;
    ev2Pop(true);
    $('ev2Fix').textContent = final ? message : fix;
    $('ev2Retry').hidden = final;
    // "Back to today" stays visible: the pop-up covers the tab bar, so it is the only way out.
    if (final) ev2Stop();
}

/** Shows or hides the failure pop-up (#ev2FailPop, moved under <body> so no card can clip it). */
function ev2Pop(on) {
    const pop = typeof document !== 'undefined' && document.getElementById('ev2FailPop');
    if (!pop) return;
    if (pop.parentNode !== document.body) document.body.appendChild(pop);
    pop.hidden = !on;
    if (on) { const t = document.getElementById('ev2FailTitle'); if (t && t.focus) t.focus({ preventScroll: true }); }
}

function ev2Stop() {
    ev2.running = false;
    if (ev2.stream) { ev2.stream.getTracks().forEach(t => t.stop()); ev2.stream = null; }
}

if (typeof document !== 'undefined' && document.getElementById('ev2Consent')) {
    $('ev2ConsentCheck').addEventListener('change', e => { $('ev2Start').disabled = !e.target.checked; });
    $('ev2Start').addEventListener('click', ev2Start);
    $('ev2Retry').addEventListener('click', async () => {
        ev2Pop(false);
        ev2UnlockAudio();
        ev2Show('ev2Camera');
        if (!ev2.stream) {
            try {
                ev2.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 640, height: 480 } });
                $('ev2Video').srcObject = ev2.stream;
            } catch (e) { ev2Fail({ fix: ev2CameraError(e) }); return; }
        }
        await ev2NewSession();   // no re-consent: the consent row from screen 1 still stands
    });
    window.addEventListener('pagehide', ev2Stop);
}
