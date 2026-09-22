const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const read = name => fs.readFileSync(path.join(__dirname, '../app/static/js/', name), 'utf8');
function harness(debug, options = {}) {
    let now = 0, brightness = 100, callback, created = 0, streams = 0, closes = 0, submits = 0;
    const timers = new Map(), elements = new Map();
    let timerId = 0;
    const node = () => ({style: {}, textContent: '', classList: {toggle() {}, remove() {}, add() {}},
        appendChild(child) {child.parentElement = this;}, after(child) {child.parentElement = this.parentElement;},
        getContext() {return {clearRect() {}, drawImage() {}, getImageData() {return {data: Array(16).fill(brightness)};}};},
        toDataURL() {return 'image';}, focus() {}, async play() {if (options.playError) throw Error('play');}});
    for (const id of ['videoVerify', 'stepVerify', 'stepDone', 'stepRoomCode', 'stepBleRoom']) elements.set(id, node());
    elements.get('videoVerify').parentElement = node();
    elements.get('stepVerify').parentElement = node();
    Object.assign(elements.get('videoVerify'), {videoWidth: 480, videoHeight: 640, readyState: options.sendError ? 2 : 0});
    const document = {visibilityState: 'visible', addEventListener() {},
        getElementById(id) {if (!elements.has(id)) elements.set(id, node()); return elements.get(id);},
        createElement() {created++; return node();}};
    const context = vm.createContext({document, window: {addEventListener() {}}, console,
        performance: {now: () => now}, Date, StepGuide: {show: async () => {}, dismiss() {}},
        setInterval() {return 1;}, clearInterval() {},
        setTimeout(fn, delay) {const id = ++timerId; timers.set(id, {fn, delay}); return id;},
        clearTimeout(id) {timers.delete(id);},
        navigator: {bluetooth: {}, mediaDevices: {async getUserMedia() {streams++; return {getTracks: () => [{stop() {}}]};}}},
        detectVirtualCamera: async () => ({blocked: false}),
        FaceMesh: class {
            setOptions() {} onResults(fn) {callback = fn;} async initialize() {}
            async send() {if (options.sendError) throw Error('send');}
            async close() {closes++; if (options.closeError) throw Error('close');}
        },
    });
    vm.runInContext((debug ? read('checkin_debug.js') : '') + read('checkin_flow.js') +
        `\nglobalThis.flow = new CheckinFlow({baselineEAR: 0.392436, debug: ${debug}, proximityMethod: 'ble'});`, context);
    const flow = context.flow;
    flow._sleep = async () => {};
    flow._drawFaceFeatures = () => {};
    flow._submitCheckin = async frame => {submits++; flow.submittedFrame = frame;};
    flow._showDone = (kind, message) => {flow.result = {kind, message};};
    flow._proximity = {deadline: 90000};
    if (flow._debug) flow._debug.receiptDue = 90000;
    return {flow, context, elements, timers, created: () => created, streams: () => streams,
        closes: () => closes, submits: () => submits, advance: t => now = t,
        dark: () => brightness = 0, frame: lm => callback({multiFaceLandmarks: lm ? [lm] : []})};
}
function landmarks(ear = .28) {
    const lm = Array.from({length: 468}, () => ({x: .5, y: .5}));
    lm[10] = {x: .5, y: .2}; lm[152] = {x: .5, y: .8};
    lm[234] = {x: .3, y: .5}; lm[454] = {x: .7, y: .5};
    for (const [indices, x] of [[[33,160,158,133,153,144], .4], [[362,385,387,263,373,380], .54]]) {
        const gap = ear * .06 * 480 / 640 / 2;
        const points = [[x,.4],[x+.015,.4+gap],[x+.045,.4+gap],[x+.06,.4],[x+.045,.4-gap],[x+.015,.4-gap]];
        indices.forEach((i, n) => lm[i] = {x: points[n][0], y: points[n][1]});
    }
    return lm;
}
(async () => {
    const off = harness(false);
    assert.equal(off.created(), 0);
    await off.flow._startVerify(); await off.frame(landmarks());
    assert.equal(off.flow._debug, null);
    assert.equal(off.streams(), 1, 'one stream per attempt');
    const h = harness(true);
    await h.flow._startVerify();
    const d = h.flow._debug;
    assert.match(d.panel.textContent, /v2 — pixel-v1/);
    assert.match(d.panel.textContent, /legacy baseline \(unused\)=0.392436/);
    await h.frame(landmarks(.12));
    assert.doesNotMatch(d.panel.textContent, /failed=eyesOk/);
    assert.match(d.panel.textContent, /ready=1/);
    assert.match(d.panel.textContent, /faceW=0.400000/);
    for (let i = 0; i < 23; i++) await h.frame(landmarks());
    assert.equal(h.submits(), 0, '24 ready frames must not submit');
    await h.frame(landmarks());
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(h.submits(), 1, 'submit after 25 consecutive ready frames');
    assert.equal(h.closes(), 1, 'close model exactly once');
    await h.frame(landmarks()); assert.equal(h.submits(), 1, 'ignore stale results');
    const reset = harness(false);
    await reset.flow._startVerify();
    for (let i = 0; i < 24; i++) await reset.frame(landmarks());
    await reset.frame(null);
    await reset.frame(landmarks());
    assert.equal(reset.submits(), 0, 'lost face resets the ready frame count');
    for (let i = 0; i < 24; i++) await reset.frame(landmarks());
    assert.equal(reset.submits(), 1);
    const timeout = harness(true);
    await timeout.flow._startVerify(); await timeout.frame(landmarks());
    const timer = [...timeout.timers.values()].find(t => t.delay === 40000);
    const blocking = timeout.elements.get('verifyStatus').textContent;
    timeout.advance(45000); timer.fn();
    assert.ok(timeout.flow.result.message.includes(blocking), 'timeout preserves actual reason');
    assert.match(timeout.flow._debug.history.textContent, /late 5000ms/);
    await timeout.flow._startVerify();
    assert.equal(timeout.flow._debug.attempts, 2);
    assert.equal(timeout.flow._debug.receiptDue, 90000);
    await timeout.frame(null); assert.match(timeout.flow._debug.panel.textContent, /earNow=n\/a/);
    timeout.dark(); await timeout.frame(landmarks());
    assert.match(timeout.flow._debug.panel.textContent, /failed=brightness/);
    timeout.advance(91000); timeout.flow._expireProximity();
    assert.equal(timeout.flow._proximity, null);
    for (const options of [{playError: true}, {sendError: true}, {playError: true, closeError: true}]) {
        const e = harness(true, options); await e.flow._startVerify();
        await new Promise(resolve => setImmediate(resolve));
        assert.equal(e.flow.result.kind, 'error'); assert.equal(e.closes(), 1);
    }
    console.log('PASS: capture without eye calibration, one stream/close, timeout, expiry, stale callbacks and promise failures');
})().catch(error => {console.error(error); process.exitCode = 1;});
