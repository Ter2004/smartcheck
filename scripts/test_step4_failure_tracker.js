'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');

// Import the exact browser asset shipped by Flask; this harness has no copy of
// the counter transition logic.
const shippedFile = path.resolve(__dirname, '../app/static/js/step4_failure_tracker.js');
const { Step4FailureTracker } = require(shippedFile);

const failure503 = { is_real: false, _networkError: true, _httpStatus: 503 };
const rate429 = { is_real: false, _networkError: true, _rateLimited: true, _httpStatus: 429 };
const refused = { is_real: false, _networkError: true };
const spoof = { is_real: false };
const pass = { is_real: true };

function caseRun(label, fn) {
    fn();
    console.log(`PASS ${label}`);
}

caseRun('(a) 503 response -> INCREMENTS', () => {
    const tracker = new Step4FailureTracker();
    assert.equal(tracker.observe(failure503).consecutive, 1);
});

caseRun('(b) 429 -> RESETS', () => {
    const tracker = new Step4FailureTracker();
    tracker.observe(refused);
    assert.equal(tracker.observe(rate429).consecutive, 0);
});

caseRun('(c) genuine spoof -> RESETS', () => {
    const tracker = new Step4FailureTracker();
    tracker.observe(refused);
    assert.equal(tracker.observe(spoof).consecutive, 0);
});

caseRun('(d) genuine pass -> RESETS', () => {
    const tracker = new Step4FailureTracker();
    tracker.observe(refused);
    assert.equal(tracker.observe(pass).consecutive, 0);
});

caseRun('(e) connection-refused -> INCREMENTS', () => {
    const tracker = new Step4FailureTracker();
    assert.equal(tracker.observe(refused).consecutive, 1);
});

caseRun('(f) 5 strikes, 429, 5 strikes -> does NOT trip', () => {
    const tracker = new Step4FailureTracker();
    for (let i = 0; i < 5; i++) assert.equal(tracker.observe(failure503).tripNow, false);
    assert.equal(tracker.observe(rate429).consecutive, 0);
    for (let i = 0; i < 5; i++) assert.equal(tracker.observe(failure503).tripNow, false);
    assert.equal(tracker.tripped, false);
});

caseRun('(g) exactly 6 consecutive -> trips once, does not re-arm', () => {
    const tracker = new Step4FailureTracker();
    let trips = 0;
    for (let i = 0; i < 10; i++) trips += tracker.observe(failure503).tripNow ? 1 : 0;
    assert.equal(trips, 1);
    assert.equal(tracker.tripped, true);
    assert.equal(tracker.consecutive, 6);
});

console.log(`Loaded shipped file: ${shippedFile}`);
