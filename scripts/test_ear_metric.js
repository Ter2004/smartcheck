const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.join(__dirname, '../app/static/js');
const context = vm.createContext({console});
vm.runInContext(fs.readFileSync(path.join(root, 'ear_metric.js'), 'utf8') +
    fs.readFileSync(path.join(root, 'mediapipe_liveness.js'), 'utf8') +
    '\nglobalThis.metric = EARMetric; globalThis.Calibration = EARCalibration;', context);
const {metric, Calibration} = context;
const indices = [33,160,158,133,153,144];
function eye(width, height, angle = 0, gap = 18) {
    const points = [[0,0],[15,gap/2],[45,gap/2],[60,0],[45,-gap/2],[15,-gap/2]];
    const lm = Array.from({length: 468}, () => ({x: .5, y: .5}));
    for (const idx of [indices, [362,385,387,263,373,380]]) {
        points.forEach(([x,y], n) => lm[idx[n]] = {
            x: (150 + x*Math.cos(angle)-y*Math.sin(angle))/width,
            y: (150 + x*Math.sin(angle)+y*Math.cos(angle))/height,
        });
    }
    return lm;
}
for (const [width, height] of [[640,480],[480,640],[1080,1920],[500,500]]) {
    for (const angle of [0, .4, -.7]) {
        const lm = eye(width,height,angle);
        assert.ok(Math.abs(metric.eye(lm,indices,width,height)-.3)<1e-12);
        assert.ok(Math.abs(context.calcEARFromLM(lm,indices,{videoWidth:width,videoHeight:height})-.3)<1e-12);
    }
    const video = {videoWidth:width,videoHeight:height};
    const checker = context.buildActionChecker('blink', .3, video);
    assert.equal(checker.check(eye(width,height)).done,false);
    assert.equal(checker.check(eye(width,height,0,6)).done,false);
    assert.equal(checker.check(eye(width,height)).done,true);
}
for (const bad of [0, -1, NaN, Infinity, undefined]) assert.ok(Number.isNaN(metric.eye(eye(640,480),indices,bad,480)));
assert.ok(Number.isNaN(metric.eye([],indices,640,480)));
const degenerate = eye(640,480); degenerate[133] = degenerate[33];
assert.ok(Number.isNaN(metric.eye(degenerate,indices,640,480)));
const nonfinite = eye(640,480); nonfinite[160].x = Infinity;
assert.ok(Number.isNaN(metric.eye(nonfinite,indices,640,480)));
const c = new Calibration();
for (let i=0;i<16;i++) c.update(.3,true,480,640,i*90);
assert.equal(c.phase,'blink');
c.update(.1,true,480,640,1500);
for (let i=0;i<5;i++) c.update(.3,true,480,640,1600+i*90);
assert.equal(c.phase,'ready');
assert.equal(c.update(.3,true,640,480,2000),false,'orientation requires new calibration');
assert.equal(c.phase,'open');
c.update(.3,false,640,480,2100);
assert.equal(c.baseline,null,'bad quality clears calibration');
const stillClosed = new Calibration();
for (let i=0;i<100;i++) assert.equal(stillClosed.update(.08,true,480,640,i*90),false,'steady eyes cannot complete blink');
const gap = new Calibration();
for (let i=0;i<16;i++) gap.update(.3,true,480,640,i*90);
gap.update(.1,true,480,640,5000);
assert.equal(gap.phase,'open','suspension invalidates old calibration');
for (const name of ['ear_metric.js','checkin_flow.js','checkin_debug.js','enrollment_flow.js','enrollment_circular.js','mediapipe_liveness.js']) {
    new vm.Script(fs.readFileSync(path.join(root,name),'utf8'), {filename:name});
}
console.log('PASS: portrait/landscape/tilted EAR, invalid geometry, corrected blink, calibration continuity and JS syntax');
