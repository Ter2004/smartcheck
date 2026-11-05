/**
 * broadcast_room_scanner.js — read the broadcast room board inside the SmartCheck app
 *
 * The board (firmware/smartcheck_broadcast) accepts no connections. It advertises
 * manufacturer data for company 0xFFFF:
 *   "SC" | version 0x01 | room code (8 bytes, 0x00 padded) | counter (4) | signature (8)
 * Every phone in the room reads it at the same time, so there is no queue. The
 * server checks the signature and freshness (app/services/broadcast_beacon.py).
 *
 * Browsers cannot scan without a tap and a permission prompt, so this works only
 * inside the Android app (android-app/), which injects window.Capacitor and the
 * BluetoothLe plugin (@capacitor-community/bluetooth-le). Plugin calls here use the
 * native API directly: the site is remote, so the plugin's JS wrapper is not loaded.
 *
 * Usage:
 *   if (BroadcastRoomScanner.available()) {
 *       const result = await new BroadcastRoomScanner().findRoom('TEST-101');
 *       // { ok: true, room: 'TEST-101', payload: '<46 hex>' }
 *       // { ok: false, code: '...', error: 'Thai message' }
 *   }
 */
class BroadcastRoomScanner {
    static COMPANY_ID = '65535';           // 0xFFFF, the plugin's decimal key
    static PAYLOAD_HEX_LEN = 46;           // 23 bytes
    static TIMEOUT_MS = 8000;      // in a room the board is found in 0.1–0.5 s (Redmi, 2026-11-02)
    static SCAN_MODE_LOW_LATENCY = 2;      // Android ScanSettings.SCAN_MODE_LOW_LATENCY

    static plugin() {
        const cap = globalThis.Capacitor;
        if (!cap || typeof cap.isNativePlatform !== 'function' || !cap.isNativePlatform()) return null;
        return (cap.Plugins && cap.Plugins.BluetoothLe) || null;
    }

    static available() {
        return BroadcastRoomScanner.plugin() !== null;
    }

    /** { room, payload } from the board's manufacturer data hex, or null. */
    static parsePayload(hex) {
        if (typeof hex !== 'string') return null;
        const value = hex.toLowerCase();
        if (value.length !== BroadcastRoomScanner.PAYLOAD_HEX_LEN || !/^[0-9a-f]+$/.test(value)) return null;
        if (!value.startsWith('534301')) return null;                      // "SC", version 1
        let room = '';
        for (let i = 6; i < 22; i += 2) {
            const code = parseInt(value.slice(i, i + 2), 16);
            if (code === 0) break;
            if (code < 0x21 || code > 0x7e) return null;
            room += String.fromCharCode(code);
        }
        return room ? {room, payload: value} : null;
    }

    async findRoom(expectedRoom, timeoutMs = BroadcastRoomScanner.TIMEOUT_MS) {
        const ble = BroadcastRoomScanner.plugin();
        if (!ble) {
            return {ok: false, code: 'unsupported', error: 'การสแกนอัตโนมัติใช้ได้ในแอป SmartCheck บน Android เท่านั้น'};
        }
        try {
            await ble.initialize({androidNeverForLocation: false});
        } catch {
            return {ok: false, code: 'permission_denied',
                error: 'แอปยังไม่ได้รับสิทธิ์ Bluetooth/ตำแหน่ง — กดอนุญาตในหน้าต่างที่ขึ้นมา หรือเปิดสิทธิ์ในการตั้งค่าแอป แล้วลองใหม่'};
        }
        try {
            const enabled = await ble.isEnabled();
            if (enabled && enabled.value === false) {
                return {ok: false, code: 'bluetooth_off', error: 'Bluetooth ปิดอยู่ — เปิด Bluetooth แล้วลองใหม่'};
            }
        } catch { /* unknown: try scanning anyway */ }

        const wanted = typeof expectedRoom === 'string' ? expectedRoom.trim() : '';
        let listener = null, timer = null, settle;
        const done = new Promise(resolve => { settle = resolve; });
        let otherRoom = '';
        try {
            listener = await ble.addListener('onScanResult', result => {
                const hex = result && result.manufacturerData && result.manufacturerData[BroadcastRoomScanner.COMPANY_ID];
                const found = BroadcastRoomScanner.parsePayload(hex);
                if (!found) return;
                if (wanted && found.room !== wanted) { otherRoom = found.room; return; }   // the room next door
                settle({ok: true, room: found.room, payload: found.payload});
            });
            timer = setTimeout(() => settle(null), timeoutMs);
            // allowDuplicates: the board changes its code every 5 s, keep the newest.
            await ble.requestLEScan({allowDuplicates: true, scanMode: BroadcastRoomScanner.SCAN_MODE_LOW_LATENCY});
            const result = await done;
            if (result) return result;
        } catch {
            return {ok: false, code: 'scan_error', error: 'เริ่มสแกน Bluetooth ไม่สำเร็จ — ปิดแล้วเปิด Bluetooth ใหม่ แล้วลองอีกครั้ง'};
        } finally {
            clearTimeout(timer);
            try { await ble.stopLEScan(); } catch { /* already stopped */ }
            try { if (listener) await listener.remove(); } catch { /* ignore */ }
        }
        if (otherRoom) {
            return {ok: false, code: 'wrong_room',
                error: 'พบอุปกรณ์ของห้องอื่น (' + otherRoom + ') แต่ไม่พบอุปกรณ์ของห้องนี้ — ตรวจสอบว่าอยู่ในห้องที่ถูกต้อง'};
        }
        let locationOff = false;
        try {
            const location = await ble.isLocationEnabled();
            locationOff = location && location.value === false;
        } catch { /* not reported */ }
        return locationOff
            ? {ok: false, code: 'location_off', error: 'ไม่พบอุปกรณ์ในห้อง — Android ต้องเปิด "ตำแหน่ง (Location)" จึงจะสแกน Bluetooth ได้ เปิดแล้วลองใหม่'}
            : {ok: false, code: 'not_found', error: 'ไม่พบอุปกรณ์ในห้องเรียน — ตรวจสอบว่าอยู่ในห้องและอุปกรณ์เปิดอยู่ แล้วลองใหม่'};
    }
}
