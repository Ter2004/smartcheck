const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const read = name => fs.readFileSync(path.join(__dirname, '../app/static/js/', name), 'utf8');
const context = vm.createContext({console});
vm.runInContext(read('mediapipe_liveness.js') + read('checkin_flow.js') +
    '\nglobalThis.api = {buildActionChecker, eyeYaw, noseRelX, TURN_YAW, CheckinFlow};', context);
const {buildActionChecker, eyeYaw, noseRelX, TURN_YAW, CheckinFlow} = context.api;
const video = {videoWidth: 640, videoHeight: 480};

// Eye centres at x 0.43 / 0.57 (corners ±0.03), cheeks at 0.35 / 0.65. `noseX` moves the
// nose tip, `cheekL`/`cheekR` move the face outline; +deg rotates every point (a tilt).
function face({noseX = 0.5, cheekL = 0.35, cheekR = 0.65, deg = 0} = {}) {
    const lm = Array.from({length: 468}, () => ({x: 0.5, y: 0.5}));
    Object.assign(lm, {33: {x: 0.40, y: 0.45}, 133: {x: 0.46, y: 0.45}, 362: {x: 0.54, y: 0.45},
                       263: {x: 0.60, y: 0.45}, 1: {x: noseX, y: 0.52}, 234: {x: cheekL, y: 0.5},
                       454: {x: cheekR, y: 0.5}});
    const t = deg * Math.PI / 180;
    return lm.map(p => {
        const x = (p.x - 0.5) * 640, y = (p.y - 0.5) * 480;
        return {x: 0.5 + (x * Math.cos(t) - y * Math.sin(t)) / 640, y: 0.5 + (x * Math.sin(t) + y * Math.cos(t)) / 480};
    });
}
const near = (a, b) => Math.abs(a - b) < 1e-9;
const run = (checker, frames) => frames.map(lm => checker.check(lm, 0).done).some(Boolean);

// Same formula as liveness_challenge.yaw_ratio: nose offset / eye distance, sign = turn_left.
assert.ok(near(eyeYaw(face(), video), 0), 'frontal reads 0');
assert.ok(near(eyeYaw(face({noseX: 0.57}), video), 0.5), 'nose at the image-right eye centre reads +0.5');
assert.ok(eyeYaw(face({noseX: 0.43}), video) < 0, 'nose toward image-left is turn_right');
assert.ok(near(eyeYaw(face({noseX: 0.55, deg: 25}), video), eyeYaw(face({noseX: 0.55}), video)),
          'measured along the eye line, so a tilt does not read as a turn');
assert.ok(near(CheckinFlow.eyeYaw(face({noseX: 0.55, deg: 10}), 480, 640), eyeYaw(face({noseX: 0.55, deg: 10}), {videoWidth: 480, videoHeight: 640})),
          'check-in and enrollment use the same formula');
assert.equal(CheckinFlow.TURN_YAW, TURN_YAW, 'check-in and enrollment use the same threshold');
assert.ok(TURN_YAW > 0.20, 'browser asks for more than the server TURN_MIN_YAW');

const turnLeft = () => buildActionChecker('turn_left', video);
const turnRight = () => buildActionChecker('turn_right', video);
const yawFace = yaw => face({noseX: 0.5 + yaw * 0.14});   // eye distance is 0.14 wide
assert.equal(run(turnLeft(), Array(3).fill(yawFace(0.35))), true, 'three turned frames complete it');
assert.equal(run(turnLeft(), [yawFace(0.35), yawFace(0.35), yawFace(0), yawFace(0.35)]), false, 'frames must be consecutive');
assert.equal(run(turnLeft(), Array(5).fill(yawFace(TURN_YAW - 0.02))), false, 'too small');
assert.equal(run(turnLeft(), Array(5).fill(yawFace(-0.35))), false, 'the other way never counts');
assert.equal(run(turnRight(), Array(3).fill(yawFace(-0.35))), true);
assert.equal(run(turnRight(), Array(5).fill(yawFace(0.35))), false);

// 2026-10-04 regression: the far cheek drops out of view first, so noseRelX passed 0.65
// while the nose had barely left the eye midpoint (server read 0.09-0.18 and failed it).
const cheekOnly = face({noseX: 0.52, cheekR: 0.56});
assert.ok(noseRelX(cheekOnly) > 0.65 && eyeYaw(cheekOnly, video) < 0.20);
assert.equal(run(turnLeft(), Array(5).fill(cheekOnly)), false, 'a small turn that only moved the outline does not count');
console.log('PASS: turn checker uses the server yaw formula, threshold, direction, consecutive frames and check-in parity');
