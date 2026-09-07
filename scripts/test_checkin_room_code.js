const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../app/static/js/checkin_flow.js'), 'utf8');
function harness(method) {
    const elements = new Map();
    let cameraCalls = 0, bleCalls = 0, now = 0;
    const requests = [];
    let accepted = false;
    const document = {
        getElementById(id) {
            if (!elements.has(id)) elements.set(id, {style: {}, value: '', classList: {toggle() {}}, focus() {},
                reportValidity() {return /^[0-9]{6}$/.test(this.value);}});
            return elements.get(id);
        },
        querySelector() {return {content: 'csrf'};},
    };
    const context = vm.createContext({document, performance: {now: () => now}, console,
        setInterval: () => 1, clearInterval() {}, clearTimeout() {},
        navigator: {bluetooth: {}, mediaDevices: {async getUserMedia() {cameraCalls++; throw new Error('test camera');}}},
        localStorage: {getItem() {return '';}},
        BLERoomScanner: class {async findRoom() {bleCalls++; return {ok: true, room: 'TEST-101'};}},
        fetch: async (url, options) => {
            requests.push({url, body: JSON.parse(options.body), headers: options.headers});
            const result = url.endsWith('/proximity') ?
                (accepted ? {ok: true, proximity_receipt: 'signed', expires_in: 90} : {ok: false, error: 'specific rejection'}) :
                {ok: true, message: 'success'};
            return {ok: result.ok, status: result.ok ? 200 : 400, json: async () => result};
        },
    });
    vm.runInContext(source + `\nglobalThis.flow = new CheckinFlow({sessionId: 's', proximityMethod: '${method}'});`, context);
    return {flow: context.flow, document, requests, context,
        camera: () => cameraCalls, ble: () => bleCalls,
        accept: () => accepted = true, expire: () => now = 91000};
}
(async () => {
    for (const method of ['ble', 'totp']) {
        const h = harness(method);
        h.flow.start();
        assert.equal(h.camera(), 0, 'opening page must not open camera');
        await h.flow._startVerify();
        assert.equal(h.camera(), 0, 'direct camera entry must require receipt');
        const enter = async () => {
            if (method === 'ble') await h.flow.startBleRoomScan();
            else {h.document.getElementById('roomCode').value = '012345'; await h.flow.submitRoomCode({preventDefault() {}});}
        };
        await enter();
        assert.equal(h.camera(), 0, 'failed preflight must not open camera');
        assert.equal(h.document.getElementById('proximityStatus').textContent, 'specific rejection');
        h.accept();
        await enter();
        assert.equal(h.camera(), 1, 'only successful preflight opens camera');
        const reads = h.ble();
        await h.flow._submitCheckin('frame', 'passive');
        assert.equal(h.ble(), reads, 'submission must never reconnect');
        const body = h.requests.at(-1).body;
        assert.equal(Object.hasOwn(h.requests.at(-1).headers, 'Authorization'), false);
        assert.equal(body.room_code, method === 'ble' ? 'TEST-101' : '012345');
        assert.equal(body.proximity_receipt, 'signed');
        h.context.localStorage.getItem = () => 'stored-token';
        await h.flow._submitCheckin('frame', 'passive');
        assert.equal(h.requests.at(-1).headers.Authorization, 'DeviceToken stored-token');
        const sent = h.requests.length;
        h.expire();
        await h.flow._submitCheckin('frame', 'passive');
        assert.equal(h.requests.length, sent, 'expired receipt requires explicit fresh verification');
        assert.equal(h.flow._proximity, null);
    }
    const unsupported = harness('ble');
    delete unsupported.context.navigator.bluetooth;
    unsupported.flow.start();
    assert.equal(unsupported.document.getElementById('bleUnsupportedWarning').style.display, 'block');
    assert.equal(unsupported.document.getElementById('bleRoomBtn').disabled, true);
    assert.equal(unsupported.camera(), 0);
    console.log('PASS: BLE/TOTP preflight gates camera, preserves values, no second read, expiry and unsupported browser');
})().catch(error => {console.error(error); process.exitCode = 1;});
