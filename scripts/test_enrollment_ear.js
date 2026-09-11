const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const read = name => fs.readFileSync(path.join(__dirname, '../app/static/js', name), 'utf8');
async function run(samples) {
    const payloads = [];
    const element = {style: {}, classList: {add() {}, remove() {}}, addEventListener() {}, checked: false};
    const context = vm.createContext({console, Date,
        document: {createElement: () => ({}), head: {appendChild() {}},
            getElementById: () => element, addEventListener() {}},
        window: {addEventListener() {}}, localStorage: {getItem: () => null},
        setTimeout() {}, clearTimeout() {}, clearInterval() {},
        ENROLL_CONFIG: {enrollUrl: '/enroll'},
        fetch: async (url, options) => {payloads.push(JSON.parse(options.body)); return {status: 200, json: async () => ({status: 'test'})};},
    });
    vm.runInContext(['ear_metric.js','step4_failure_tracker.js','enrollment_flow.js','enrollment_circular.js'].map(read).join('\n'), context);
    context.samples = samples;
    vm.runInContext(`
        _circFrontEarSamples.push(...samples);
        window.circularCapturedFrames = {FRONT:'a', RIGHT:'b', LEFT:'c', UP:'d', DOWN:'e'};
        _circSetCheckingMsg = () => {};
        _circShowEnrollError = () => {};
        _circSpoofCheck = async () => ({is_real:true});
        _csrfToken = () => 'test';
        globalThis.complete = _circHandleAllComplete();
    `, context);
    await context.complete;
    assert.equal(payloads.length,1);
    assert.equal(vm.runInContext('_circFrontEarSamples.length', context),0,'stop still clears live samples');
    return payloads[0];
}
(async () => {
    const measured = await run([.31,.33]);
    assert.ok(Math.abs(measured.baseline_ear-.32)<1e-12,'preserve measurements before stop');
    assert.ok(Math.abs(measured.ear_std-.01)<1e-12);
    assert.equal(measured.baseline_ear_metric,'pixel-v1');
    const missing = await run([NaN]);
    assert.equal(missing.baseline_ear,null,'missing measurements are unknown, not 0.25');
    assert.equal(missing.baseline_ear_metric,null);
    console.log('PASS: actual circular completion preserves EAR samples and versions only measured baselines');
})().catch(error => {console.error(error); process.exitCode = 1;});
