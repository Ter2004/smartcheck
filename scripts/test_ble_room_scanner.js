const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../app/static/js/ble_room_scanner.js'), 'utf8');

const NONCE = '000102030405060708090a0b0c0d0e0f';
const SIGNED = new Uint8Array(32).map((_, i) => 255 - i);
const SIGNED_HEX = Array.from(SIGNED, b => b.toString(16).padStart(2, '0')).join('');

async function scenario({ delayed = false, drop = false, empty = false, readError = false,
                          cleanupError = false, refuse = false, outdated = false, short = false } = {}) {
    const listeners = new Set();
    let disconnects = 0, written = null;
    const emitDrop = () => { for (const fn of [...listeners]) fn(); };
    const room = { async readValue() {
        if (drop) {
            device.gatt.connected = false;
            emitDrop();
        }
        if (readError) throw new Error('read failed');
        return new TextEncoder().encode(empty ? ' ' : 'TEST-101');
    } };
    const signer = {
        async writeValueWithResponse(bytes) { written = Array.from(bytes); },
        async readValue() {
            const bytes = short ? SIGNED.slice(0, 16) : SIGNED;
            return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        },
    };
    const server = { async getPrimaryService() {
        return { async getCharacteristic(uuid) {
            if (uuid.startsWith('7c6b1001')) return room;
            if (outdated) throw new Error('NotFoundError');
            return signer;
        } };
    } };
    const device = {
        addEventListener(_name, fn) { listeners.add(fn); },
        removeEventListener(_name, fn) { listeners.delete(fn); },
        gatt: {
            connected: false,
            async connect() { this.connected = true; return server; },
            disconnect() {
                disconnects++;
                assert.equal(listeners.size, 0, 'detach before deliberate disconnect');
                this.connected = false;
                if (cleanupError) throw new Error('cleanup failed');
                if (delayed) setTimeout(emitDrop, 0);
                else emitDrop();
            },
        },
    };
    const nonceRequests = [];
    const context = vm.createContext({ TextDecoder, DataView, Uint8Array, navigator: { bluetooth: {
        async requestDevice() { return device; },
    } } });
    vm.runInContext(source + '\nglobalThis.scanner = new BLERoomScanner();', context);
    const result = await context.scanner.proveRoom(async roomCode => {
        nonceRequests.push(roomCode);
        return refuse ? { ok: false, error: 'server says no' } : { ok: true, nonce: NONCE };
    });
    await new Promise(resolve => setTimeout(resolve, 5));
    assert.equal(listeners.size, 0);
    assert.equal(device.gatt.connected, false, 'board is always released');
    assert.equal(disconnects, drop ? 0 : 1);
    return { result, written, nonceRequests };
}

(async () => {
    for (const options of [{}, { delayed: true }, { cleanupError: true }]) {
        const { result, written, nonceRequests } = await scenario(options);
        assert.equal(result.ok, true);
        assert.equal(result.room, 'TEST-101');
        assert.equal(result.response, SIGNED_HEX, 'board HMAC is returned as hex');
        assert.deepEqual(written, Array.from({ length: 16 }, (_, i) => i), 'nonce bytes written');
        assert.deepEqual(nonceRequests, ['TEST-101'], 'nonce requested for the room read');
    }
    assert.equal((await scenario({ drop: true })).result.code, 'connection_dropped');
    assert.equal((await scenario({ empty: true })).result.code, 'empty_value');
    assert.equal((await scenario({ readError: true })).result.code, 'connection_dropped');
    const refused = await scenario({ refuse: true });
    assert.equal(refused.result.code, 'challenge_refused');
    assert.equal(refused.result.error, 'server says no');
    assert.equal(refused.written, null, 'nothing written without a nonce');
    assert.equal((await scenario({ outdated: true })).result.code, 'board_outdated');
    assert.equal((await scenario({ short: true })).result.code, 'bad_signature');
    console.log('PASS: 9 BLE challenge/read/disconnect scenarios');
})().catch(error => { console.error(error); process.exitCode = 1; });
