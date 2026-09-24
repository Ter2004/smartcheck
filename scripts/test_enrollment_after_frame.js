// Regression: the post-challenge ("after") frame must come from a live camera.
// MediaPipe's Camera opens its own stream on the video element and stops it when
// the gesture detector finishes, so a frame grabbed right then is black (~2.5 KB,
// rejected as frame_too_small) and every enrollment failed the head-turn check.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.join(__dirname, '..');

function stream(id) {
    const s = {id, live: true};
    s.getTracks = () => [{stop() { s.live = false; }}];
    return s;
}

(async () => {
    let opened = 0, failOpen = false;
    const element = {style: {}, classList: {add() {}, remove() {}, toggle() {}}, addEventListener() {},
        checked: false, textContent: ''};
    const video = {readyState: 4, videoWidth: 640, videoHeight: 480, srcObject: null, async play() {}};
    const context = vm.createContext({console, Date,
        _rtResetCounters() {}, stopStream() {},
        document: {
            // The canvas records which stream was on the video when it was drawn.
            createElement: () => {
                let drawn = null;
                return {style: {}, getContext: () => ({drawImage: v => { drawn = v.srcObject; }}),
                        toDataURL: () => 'frame:' + (drawn && drawn.live ? drawn.id : 'black')};
            },
            head: {appendChild() {}},
            getElementById: id => id === 'stepCircular' ? null : element,
            querySelectorAll: () => [element, element, element, element], addEventListener() {}},
        window: {addEventListener() {}}, localStorage: {getItem: () => null},
        navigator: {mediaDevices: {getUserMedia: async () => {
            if (failOpen) throw Error('NotAllowedError');
            return stream('camera-' + ++opened);
        }}},
        setTimeout: fn => { fn(); return 0; }, clearTimeout() {}, clearInterval() {},
        ENROLL_CONFIG: {},
    });
    for (const file of ['step4_failure_tracker.js', 'enrollment_flow.js']) {
        vm.runInContext(fs.readFileSync(path.join(root, 'app/static/js', file), 'utf8'), context);
    }
    context.video = video;

    // State right after detector.run(): our stream was replaced by MediaPipe's,
    // which the detector then stopped.
    const ours = stream('before-challenge');
    const mediapipe = stream('mediapipe-camera');
    vm.runInContext('livenessStream = globalThis.ours;', Object.assign(context, {ours}));
    video.srcObject = mediapipe;
    mediapipe.getTracks()[0].stop();

    assert.equal(vm.runInContext('_captureFrameFromVideo(video)', context), 'frame:black',
        'the old capture point sees a stopped stream');

    const frame = await vm.runInContext('_recaptureAfterChallenge(video)', context);
    assert.equal(frame, 'frame:camera-1', 'the after frame comes from a freshly opened camera');
    assert.equal(ours.live, false, 'the previous stream is released first');
    assert.equal(video.srcObject.id, 'camera-1');
    assert.equal(vm.runInContext('livenessStream.id', context), 'camera-1',
        'the new stream is tracked so the flow can stop it later');
    assert.equal(element.textContent, 'มองตรงเข้ากล้องอีกครั้ง');

    failOpen = true;
    assert.equal(await vm.runInContext('_recaptureAfterChallenge(video)', context), null,
        'no camera → no after frame, so the challenge is retried');

    const source = fs.readFileSync(path.join(root, 'app/static/js/enrollment_flow.js'), 'utf8');
    assert.match(source, /const frame2\s+= await _recaptureAfterChallenge\(videoLiv\)/,
        'the post-challenge check uses the reopened camera');
    console.log('PASS: post-challenge frame is taken from a live, reopened camera');
})().catch(error => { console.error(error); process.exitCode = 1; });
