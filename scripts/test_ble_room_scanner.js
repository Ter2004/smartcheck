const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../app/static/js/ble_room_scanner.js'), 'utf8');

async function scenario({ delayed = false, drop = false, empty = false, readError = false, cleanupError = false } = {}) {
    const listeners = new Set();
    let disconnects = 0;
    const emitDrop = () => { for (const fn of [...listeners]) fn(); };
    const characteristic = { async readValue() {
        if (drop) {
            device.gatt.connected = false;
            emitDrop();
        }
        if (readError) throw new Error('read failed');
        return new TextEncoder().encode(empty ? ' ' : 'TEST-101');
    } };
    const server = { async getPrimaryService() {
        return { async getCharacteristic() { return characteristic; } };
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
    const context = vm.createContext({ TextDecoder, navigator: { bluetooth: {
        async requestDevice() { return device; },
    } } });
    vm.runInContext(source + '\nglobalThis.scanner = new BLERoomScanner();', context);
    const result = await context.scanner.findRoom();
    await new Promise(resolve => setTimeout(resolve, 5));
    assert.equal(listeners.size, 0);
    assert.equal(device.gatt.connected, false);
    assert.equal(disconnects, drop ? 0 : 1);
    return result;
}

(async () => {
    for (const options of [{}, { delayed: true }, { cleanupError: true }]) {
        const result = await scenario(options);
        assert.equal(result.ok, true);
        assert.equal(result.room, 'TEST-101');
    }
    assert.equal((await scenario({ drop: true })).code, 'connection_dropped');
    assert.equal((await scenario({ empty: true })).code, 'empty_value');
    assert.equal((await scenario({ readError: true })).code, 'connection_dropped');
    console.log('PASS: 6 BLE read/disconnect regression scenarios');
})().catch(error => { console.error(error); process.exitCode = 1; });
