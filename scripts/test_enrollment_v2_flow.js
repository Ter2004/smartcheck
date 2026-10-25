// node scripts/test_enrollment_v2_flow.js — the enrollment_v2.js page controller, driven with
// synthetic MediaPipe landmarks: align -> look + blink -> the four poses in the issued order -> submit.
// The DOM, camera, FaceMesh and fetch are stubs; frame scoring (canvas) is replaced.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const src = fs.readFileSync(path.join(__dirname, '../app/static/js/enrollment_v2.js'), 'utf8');

function element(id) {
    const classes = new Set();
    return {
        id, textContent: '', hidden: false, disabled: false, style: {}, dataset: {}, attrs: {},
        readyState: 4, videoWidth: 640, videoHeight: 480, srcObject: null,
        classList: {
            toggle(c, force) { (force === undefined ? !classes.has(c) : force) ? classes.add(c) : classes.delete(c); },
            contains: (c) => classes.has(c),
        },
        setAttribute(k, v) { this.attrs[k] = v; },
        getTotalLength: () => 760,
        addEventListener() {},
    };
}

function page(order) {
    const byId = new Map();
    const screens = ['ev2Consent', 'ev2Camera', 'ev2Result'].map(element);
    const segs = Array.from({ length: 5 }, () => element('seg'));
    const arrows = ['left', 'right', 'up', 'down'].map(p => Object.assign(element('arrow'), { dataset: { pose: p } }));
    const posts = [];
    let vibrations = 0;
    let clock = 1000;
    const document = {
        getElementById: (id) => id === 'ev2Consent' ? null      // skip the button wiring
            : byId.get(id) || byId.set(id, element(id)).get(id),
        querySelectorAll: (sel) => ({ '.ev2-screen': screens, '.ev2-seg': segs, '.ev2-arrow': arrows })[sel] || [],
        querySelector: () => null,
    };
    class FaceMesh { setOptions() {} onResults() {} send() { return Promise.resolve(); } }
    const ctx = vm.createContext({
        console, Math, JSON, Promise, Object, Array, Number,
        document, FaceMesh, window: { addEventListener() {} },
        navigator: { vibrate: () => { vibrations++; return true; } },
        performance: { now: () => clock },
        requestAnimationFrame: () => {},                      // the capture loop parks after its first frame
        ENROLL_V2_CONFIG: { consentUrl: '/consent', challengeUrl: '/challenge', enrollUrl: '/enroll' },
        fetch: async (url, opts) => {
            posts.push({ url, body: JSON.parse(opts.body) });
            const json = url === '/challenge' ? { challenge_id: 'c1', order, expires_in: 120 } : { status: 'enrolled' };
            return { status: 200, json: async () => json };
        },
    });
    vm.runInContext(src, ctx);
    vm.runInContext("ev2Score = () => ({ score: 100 }); ev2Grab = () => 'data:image/jpeg;base64,AAAA';", ctx);
    const api = vm.runInContext('({ ev2, ev2NewSession, ev2OnResults, EV2 })', ctx);
    return {
        api, posts, segs, arrows, el: (id) => document.getElementById(id),
        vibrations: () => vibrations,
        feed(lm, frames = 1, ms = 100) {
            for (let i = 0; i < frames; i++) { clock += ms; api.ev2OnResults({ multiFaceLandmarks: lm ? [lm] : [] }); }
        },
    };
}

// Same synthetic face as test_enrollment_v2.js: nose x moves for turns, nose y for up/down.
function face({ noseX = 0.5, noseY = 0.55, earOpen = 0.30 } = {}) {
    const lm = Array.from({ length: 468 }, () => ({ x: 0.5, y: 0.5 }));
    const set = (i, x, y) => { lm[i] = { x, y }; };
    set(33, 0.40, 0.45); set(133, 0.46, 0.45); set(362, 0.54, 0.45); set(263, 0.60, 0.45);
    set(1, noseX, noseY); set(234, 0.35, 0.5); set(454, 0.65, 0.5); set(10, 0.5, 0.25); set(152, 0.5, 0.75);
    const gap = earOpen * 0.06 * 640 / 480;
    for (const [p1, p2, p4, p5, cx] of [[160, 158, 153, 144, 0.43], [385, 387, 373, 380, 0.57]]) {
        set(p1, cx - 0.01, 0.45 - gap / 2); set(p2, cx + 0.01, 0.45 - gap / 2);
        set(p4, cx + 0.01, 0.45 + gap / 2); set(p5, cx - 0.01, 0.45 + gap / 2);
    }
    set(13, 0.5, 0.65); set(14, 0.5, 0.651); set(78, 0.46, 0.65); set(308, 0.54, 0.65);
    return lm;
}
const FRONTAL = face();
const POSED = { left: face({ noseX: 0.56 }), right: face({ noseX: 0.44 }), up: face({ noseY: 0.51 }), down: face({ noseY: 0.59 }) };
const tick = () => new Promise(r => setTimeout(r, 0));

async function lookDone(p) {
    p.feed(FRONTAL, 8);                                  // align: 800 ms in the oval, frontal
    assert.equal(p.api.ev2.phase, 'look');
    p.feed(FRONTAL, 10);                                 // eyes open: EAR baseline and frontal frames
    p.feed(face({ earOpen: 0.12 }), 1);                  // blink
    p.feed(FRONTAL, 12);                                 // > 2 s in look
    assert.equal(p.api.ev2.phase, 'pose');
    p.feed(FRONTAL, 5);                                  // the ✅ flash: frames ignored
}

(async () => {
    // 1. Happy path in a shuffled order: 12 frames, capture order, signs, feedback, ring.
    {
        const order = ['down', 'right', 'up', 'left'];
        const p = page(order);
        await p.api.ev2NewSession();
        await lookDone(p);
        for (const pose of order) {
            p.feed(POSED[pose], 1);
            assert.equal(p.arrows.filter(a => a.classList.contains('show')).map(a => a.dataset.pose).join(), pose, `${pose} arrow only`);
            p.feed(POSED[pose], 6);                       // two samples at 200 ms, then the flash
            p.feed(FRONTAL, 3);
        }
        await tick(); await tick();
        const sent = p.posts.find(x => x.url === '/enroll');
        assert.ok(sent, 'submitted');
        const frames = sent.body.frames;
        assert.equal(sent.body.challenge_id, 'c1');
        assert.equal(frames.length, 12);
        assert.equal(frames.map(f => f.pose).join(), ['frontal', 'frontal', 'frontal', 'frontal',
            ...order.flatMap(x => [x, x])].join(), 'frontal first, then the issued order');
        assert.ok(frames.every((f, i) => i === 0 || f.t > frames[i - 1].t), 'times increase');
        const by = (pose) => frames.filter(f => f.pose === pose);
        assert.ok(by('up').every(f => f.client_pitch >= p.api.EV2.PITCH_STEP), 'up: + nose-ratio change');
        assert.ok(by('down').every(f => f.client_pitch <= -p.api.EV2.PITCH_STEP), 'down: - change');
        assert.ok(by('left').every(f => f.client_yaw > 0.30) && by('right').every(f => f.client_yaw < -0.30));
        assert.ok(by('frontal').every(f => f.client_pitch === undefined), 'frontal frames carry no pitch');
        assert.equal(p.vibrations(), 5, 'one buzz per completed pose');
        assert.equal(p.el('ev2Success').hidden, false, 'success screen');
        assert.equal(p.segs.map(s => s.attrs.class).join(), Array(5).fill('ev2-seg done').join());
    }

    // 2. The opposite turn and a turned head while looking up are not captured; timeout fails the session.
    {
        const order = ['left', 'up', 'right', 'down'];
        const p = page(order);
        await p.api.ev2NewSession();
        await lookDone(p);
        p.feed(POSED.right, 20);                          // asked left, turns right
        assert.equal(p.api.ev2.poseIdx, 0);
        p.feed(POSED.left, 7);
        assert.equal(p.api.ev2.order[p.api.ev2.poseIdx], 'up');
        p.feed(face({ noseX: 0.56, noseY: 0.51 }), 20);   // up while turned: the server would not see frontal yaw
        assert.equal(p.api.ev2.poseIdx, 1, 'turned-up frames are not up frames');
        p.feed(FRONTAL, 130);                              // > 12 s without the pose
        assert.equal(p.el('ev2Failure').hidden, false);
        assert.match(p.el('ev2Fix').textContent, /เงย/);
        assert.equal(p.posts.some(x => x.url === '/enroll'), false, 'nothing submitted');
        assert.equal(p.arrows.some(a => a.classList.contains('show')), false, 'arrows hidden after failure');
    }

    // 3. Leaving the oval or a second face resets the align hold; no capture without a face.
    {
        const p = page(['up', 'down', 'left', 'right']);
        await p.api.ev2NewSession();
        p.feed(FRONTAL, 4);
        p.api.ev2OnResults({ multiFaceLandmarks: [FRONTAL, FRONTAL] });
        assert.match(p.el('ev2Instruction').textContent, /หลายหน้า/);
        p.feed(FRONTAL, 4);
        assert.equal(p.api.ev2.phase, 'align', 'the 600 ms hold restarts');
        p.feed(FRONTAL, 3);
        assert.equal(p.api.ev2.phase, 'look');
    }
    console.log('test_enrollment_v2_flow.js: all assertions passed');
})().catch((e) => { console.error(e); process.exit(1); });
