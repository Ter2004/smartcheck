const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const context = vm.createContext({console});
vm.runInContext(fs.readFileSync(path.join(__dirname, '../app/static/js/mediapipe_liveness.js'), 'utf8') +
    '\nglobalThis.api = {buildActionChecker, eyeRollDeg, TILT_DEG, ACTION_LABELS};', context);
const {buildActionChecker, eyeRollDeg, TILT_DEG, ACTION_LABELS} = context.api;
const video = {videoWidth: 640, videoHeight: 480};

// Frontal face; rotate every point about the centre in pixels. +deg lowers the image-right eye.
function face(deg = 0, noseX = 0.5) {
    const lm = Array.from({length: 468}, () => ({x: 0.5, y: 0.5}));
    Object.assign(lm, {33: {x: 0.42, y: 0.45}, 263: {x: 0.58, y: 0.45}, 1: {x: noseX, y: 0.52},
                       234: {x: 0.35, y: 0.5}, 454: {x: 0.65, y: 0.5}});
    const t = deg * Math.PI / 180;
    return lm.map(p => {
        const x = (p.x - 0.5) * 640, y = (p.y - 0.5) * 480;
        return {x: 0.5 + (x * Math.cos(t) - y * Math.sin(t)) / 640, y: 0.5 + (x * Math.sin(t) + y * Math.cos(t)) / 480};
    });
}
const run = (checker, frames, baseline = 0) => frames.map(lm => checker.check(lm, baseline).done).some(Boolean);

assert.ok(Math.abs(eyeRollDeg(face(20), video) - 20) < 1e-6, 'roll in pixel space, sign = tilt_left');
assert.ok(Math.abs(eyeRollDeg(face(0), video)) < 1e-9);
assert.ok(TILT_DEG >= 15, 'browser asks for more than the server minimum');
assert.match(ACTION_LABELS.tilt_left, /เอียง/);

const left = () => buildActionChecker('tilt_left', 0.3, video);
assert.equal(run(left(), [face(25), face(25), face(25)]), true, 'three tilted frames complete it');
assert.equal(run(left(), [face(25), face(25), face(0), face(25)]), false, 'frames must be consecutive');
assert.equal(run(left(), Array(5).fill(face(-25))), false, 'the other way never counts');
assert.equal(run(left(), Array(5).fill(face(TILT_DEG - 3))), false, 'too small');
assert.equal(run(left(), Array(5).fill(face(25, 0.72))), false, 'tilted but turned is not a tilt');
// Phone held at -15 degrees: 25 absolute is +40 from there, 0 absolute only +15.
assert.equal(run(left(), Array(5).fill(face(0)), -15), false);
assert.equal(run(left(), Array(3).fill(face(10)), -15), true);
assert.equal(run(buildActionChecker('tilt_right', 0.3, video), Array(3).fill(face(-25))), true);
console.log('PASS: tilt checker direction, threshold, consecutive frames, turned faces and held-phone baseline');
