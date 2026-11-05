// Broadcast room board: payload parsing, the app's scanner (fake BluetoothLe plugin) and the check-in flow.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const load = name => fs.readFileSync(path.join(__dirname, '../app/static/js/', name), 'utf8');

// Same bytes as the firmware self-test and tests/test_broadcast_beacon.py (key 00..1f, counter 0x01020304).
const VECTOR = '534301544553542d313031010203047b6919855429e731';
const OTHER_ROOM = '53430152' + '4f4f4d2d3230' + '32' + '01020304' + '00'.repeat(8);   // "ROOM-202"

function fakeBle({adverts = [], enabled = true, location = true, denied = false} = {}) {
    const calls = [];
    let listener = null;
    return {
        calls,
        plugin: {
            async initialize(options) { calls.push(['initialize', options]); if (denied) throw new Error('Permission denied.'); },
            async isEnabled() { return {value: enabled}; },
            async isLocationEnabled() { return {value: location}; },
            addListener(event, callback) {
                calls.push(['addListener', event]);
                listener = callback;
                return {remove: async () => { calls.push(['remove']); listener = null; }};
            },
            async requestLEScan(options) {
                calls.push(['requestLEScan', options]);
                // Results arrive after the scan starts, as on a phone.
                setTimeout(() => adverts.forEach(hex => listener && listener({
                    device: {deviceId: 'AA'}, rssi: -60,
                    manufacturerData: hex === null ? {} : {'65535': hex.toUpperCase()}})), 0);
            },
            async stopLEScan() { calls.push(['stopLEScan']); },
        },
    };
}

function scannerContext(ble) {
    const context = vm.createContext({console, setTimeout, clearTimeout,
        Capacitor: ble ? {isNativePlatform: () => true, Plugins: {BluetoothLe: ble.plugin}} : undefined});
    vm.runInContext(load('broadcast_room_scanner.js') + '\nglobalThis.Scanner = BroadcastRoomScanner;', context);
    return context.Scanner;
}

function flowHarness({ble, proximityOk = true, gate = null}) {
    const elements = new Map();
    const requests = [];
    let guides = 0, cameras = 0;
    const document = {
        getElementById(id) {
            if (!elements.has(id)) elements.set(id, {style: {}, value: '', textContent: '', classList: {toggle() {}}, focus() {}});
            return elements.get(id);
        },
        querySelector() { return {content: 'csrf'}; },
    };
    const context = vm.createContext({document, console, setTimeout, clearTimeout,
        performance: {now: () => 0}, setInterval: () => 1, clearInterval() {},
        navigator: {mediaDevices: {async getUserMedia() { throw new Error('test camera'); }}},
        localStorage: {getItem() { return ''; }},
        StepGuide: {async show() { guides++; }, dismiss() {}},
        Capacitor: ble ? {isNativePlatform: () => true, Plugins: {BluetoothLe: ble.plugin}} : undefined,
        fetch: async (url, options) => {
            requests.push({url, body: JSON.parse(options.body)});
            if (gate && url.endsWith('/proximity')) await gate;
            const result = url.endsWith('/proximity')
                ? (proximityOk ? {ok: true, proximity_receipt: 'signed', expires_in: 90} : {ok: false, error: 'stale code'})
                : {ok: false, error: 'not in this test'};
            return {ok: result.ok, status: result.ok ? 200 : 400, json: async () => result};
        },
    });
    vm.runInContext(load('broadcast_room_scanner.js') + '\n' + load('checkin_flow.js') +
        '\nBroadcastRoomScanner.TIMEOUT_MS = 50;' +
        `\nglobalThis.flow = new CheckinFlow({sessionId: 's', proximityMethod: 'broadcast', expectedRoom: 'TEST-101', sessionCodeEnabled: true});`,
        context);
    context.flow._startVerify = async () => { cameras++; };      // camera is covered by other tests
    return {flow: context.flow, el: id => document.getElementById(id), requests, guides: () => guides,
            cameras: () => cameras};
}

const settle = () => new Promise(resolve => setTimeout(resolve, 20));
// Objects made inside a vm context have that context's prototypes; compare as JSON.
const same = (actual, expected, message) => assert.equal(JSON.stringify(actual), JSON.stringify(expected), message);

(async () => {
    // ── payload parsing ──
    const Scanner = scannerContext(null);
    same(Scanner.parsePayload(VECTOR.toUpperCase()), {room: 'TEST-101', payload: VECTOR});
    assert.equal(Scanner.parsePayload(OTHER_ROOM).room, 'ROOM-202');
    for (const bad of [null, '', VECTOR.slice(2), '544301' + VECTOR.slice(6), '534302' + VECTOR.slice(6),
                       VECTOR.slice(0, 44) + 'zz', '534301' + '0a'.repeat(8) + VECTOR.slice(22), '534301' + '00'.repeat(20)]) {
        assert.equal(Scanner.parsePayload(bad), null, String(bad));
    }
    assert.equal(Scanner.available(), false, 'a browser has no native plugin');

    // ── scanner with a fake plugin ──
    let ble = fakeBle({adverts: [null, OTHER_ROOM, VECTOR]});
    let result = await new (scannerContext(ble))().findRoom('TEST-101', 1000);
    same(result, {ok: true, room: 'TEST-101', payload: VECTOR}, 'skips other devices and the room next door');
    same(ble.calls[0], ['initialize', {androidNeverForLocation: false}]);
    assert.equal(ble.calls.find(c => c[0] === 'requestLEScan')[1].allowDuplicates, true);
    assert.ok(ble.calls.some(c => c[0] === 'stopLEScan') && ble.calls.some(c => c[0] === 'remove'), 'scan stopped and listener removed');

    result = await new (scannerContext(fakeBle({adverts: [OTHER_ROOM]})))().findRoom('TEST-101', 50);
    assert.equal(result.code, 'wrong_room');
    result = await new (scannerContext(fakeBle({location: false})))().findRoom('TEST-101', 50);
    assert.equal(result.code, 'location_off');
    result = await new (scannerContext(fakeBle()))().findRoom('TEST-101', 50);
    assert.equal(result.code, 'not_found');
    result = await new (scannerContext(fakeBle({enabled: false})))().findRoom('TEST-101', 50);
    assert.equal(result.code, 'bluetooth_off');
    result = await new (scannerContext(fakeBle({denied: true})))().findRoom('TEST-101', 50);
    assert.equal(result.code, 'permission_denied');

    // ── flow in the app: scans by itself, no tap-to-start guide ──
    let h = flowHarness({ble: fakeBle({adverts: [VECTOR]})});
    await h.flow.start();
    assert.equal(h.cameras(), 0, 'no camera before the board is found');
    await settle();
    assert.equal(h.cameras(), 1, 'camera opens once the board is found');
    assert.equal(h.guides(), 0, 'broadcast mode must not wait for a tap');
    assert.equal(h.el('stepBroadcastRoom').style.display, 'block');
    const proximity = h.requests.find(r => r.url === '/api/checkin/proximity');
    assert.ok(proximity, 'board code sent to the server');
    same(proximity.body, {session_id: 's', room_code: 'TEST-101', broadcast_payload: VECTOR});
    assert.equal(h.flow._proximity.receipt, 'signed');
    assert.notEqual(h.el('broadcastRetryBtn').style.display, 'block');

    // ── server refuses (e.g. stale code): retry button, no receipt ──
    h = flowHarness({ble: fakeBle({adverts: [VECTOR]}), proximityOk: false});
    await h.flow.start();
    await settle();
    assert.equal(h.flow._proximity, null);
    assert.equal(h.el('broadcastRetryBtn').style.display, 'block');
    assert.equal(h.el('proximityStatus').textContent, 'stale code');

    // ── camera opens while the server checks the board; the check-in waits for the receipt ──
    let release;
    h = flowHarness({ble: fakeBle({adverts: [VECTOR]}), gate: new Promise(resolve => { release = resolve; })});
    await h.flow.start();
    await settle();
    assert.equal(h.cameras(), 1, 'camera started before the server answered');
    assert.equal(h.flow._proximity.receipt, null);
    const submitted = h.flow._submitCheckin('face', 'passive');
    await settle();
    assert.ok(!h.requests.some(r => r.url === '/api/checkin'), 'no check-in before the receipt');
    release();
    await submitted;
    const checkin = h.requests.find(r => r.url === '/api/checkin');
    assert.ok(checkin, 'check-in sent once the receipt arrived');
    assert.equal(checkin.body.proximity_receipt, 'signed');

    // ── refused while the camera was open: back to the room step, nothing submitted ──
    h = flowHarness({ble: fakeBle({adverts: [VECTOR]}), proximityOk: false,
                     gate: new Promise(resolve => { release = resolve; })});
    await h.flow.start();
    await settle();
    const refused = h.flow._submitCheckin('face', 'passive');
    release();
    await refused;
    await settle();
    assert.equal(h.flow._proximity, null);
    assert.ok(!h.requests.some(r => r.url === '/api/checkin'));
    assert.equal(h.el('stepBroadcastRoom').style.display, 'block');
    assert.equal(h.el('broadcastRetryBtn').style.display, 'block');
    assert.equal(h.el('proximityStatus').textContent, 'stale code');

    // ── not in the room (nothing found): never opens the camera, asks to be in range ──
    h = flowHarness({ble: fakeBle()});
    await h.flow.start();
    await new Promise(resolve => setTimeout(resolve, 150));
    assert.equal(h.cameras(), 0, 'no face scan outside the room');
    assert.equal(h.flow._proximity, null);
    assert.ok(!h.requests.length, 'nothing sent without the board');
    assert.equal(h.el('broadcastRetryBtn').style.display, 'block');
    assert.match(h.el('broadcastStatus').textContent, /กรุณาอยู่ภายในระยะสัญญาณของห้องเรียน/);

    // ── other scan failures keep the scanner's own message ──
    h = flowHarness({ble: fakeBle({enabled: false})});
    await h.flow.start();
    await settle();
    assert.match(h.el('broadcastStatus').textContent, /Bluetooth ปิดอยู่/);

    // ── plain browser: no scan, points to the teacher's code ──
    h = flowHarness({ble: null});
    await h.flow.start();
    await settle();
    assert.equal(h.requests.length, 0);
    assert.equal(h.el('broadcastUnsupportedWarning').style.display, 'block');
    assert.match(h.el('broadcastUnsupportedWarning').textContent, /รหัสจากจอของอาจารย์/);

    console.log('broadcast check-in tests passed');
})().catch(error => { console.error(error); process.exit(1); });
