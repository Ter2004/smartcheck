/**
 * ble_room_scanner.js — Web Bluetooth GATT room-identifier reader
 *
 * Talks to the SmartCheck ESP32 classroom peripheral (firmware/README.md):
 * a connectable GATT device, not an RSSI/iBeacon broadcaster. Proximity proof
 * is "we connected to this specific board and read its room value" — there
 * is no RSSI in this protocol, so BLE-mode check-ins never populate ble_rssi.
 *
 * Usage:
 *   const result = await new BLERoomScanner().findRoom();
 *   // { ok: true, room: 'TEST-101' }
 *   // { ok: false, code: '...', error: 'Thai message' }
 */
class BLERoomScanner {
    static SERVICE_UUID        = '7c6b1000-9f3a-4b27-8d15-6e2a90c4f801';
    static CHARACTERISTIC_UUID = '7c6b1001-9f3a-4b27-8d15-6e2a90c4f801';

    async findRoom() {
        if (!navigator.bluetooth) {
            return {
                ok: false, code: 'unsupported',
                error: 'เบราว์เซอร์นี้ไม่รองรับ Web Bluetooth — กรุณาใช้ Chrome หรือ Edge บนคอมพิวเตอร์ Windows',
            };
        }

        let device;
        try {
            device = await navigator.bluetooth.requestDevice({
                filters: [{ services: [BLERoomScanner.SERVICE_UUID] }],
            });
        } catch (err) {
            // Chrome reports both "user cancelled the picker" and "no matching
            // device found" as NotFoundError with different message text.
            // NOT VERIFIED against the actual Chrome/Edge build in use — check
            // this split during testing; message text is not a stable contract.
            const msg = (err.message || '').toLowerCase();
            if (err.name === 'NotFoundError' && msg.includes('cancel')) {
                return {
                    ok: false, code: 'user_cancelled',
                    error: 'ยกเลิกการเลือกอุปกรณ์ — กดปุ่ม "หาอุปกรณ์ในห้อง" อีกครั้งเพื่อลองใหม่',
                };
            }
            if (err.name === 'NotFoundError') {
                return {
                    ok: false, code: 'device_not_found',
                    error: 'ไม่พบอุปกรณ์ในห้องเรียน — ตรวจสอบว่าอุปกรณ์เปิดอยู่และคุณอยู่ในห้อง แล้วลองใหม่',
                };
            }
            return {
                ok: false, code: 'picker_error',
                error: `ไม่สามารถเปิดตัวเลือกอุปกรณ์ Bluetooth ได้: ${err.message}`,
            };
        }

        return await new Promise((resolve) => {
            let settled = false;
            const finish = (result) => {
                if (settled) return;
                settled = true;
                device.removeEventListener('gattserverdisconnected', onDrop);
                // Fix the outcome before releasing the single-client board.
                try {
                    if (device.gatt.connected) device.gatt.disconnect();
                } catch {
                    // Cleanup must not replace the completed read/original error.
                }
                resolve(result);
            };
            const onDrop = () => {
                if (settled) return;
                finish({
                ok: false, code: 'connection_dropped',
                error: 'การเชื่อมต่อกับอุปกรณ์ในห้องเรียนขาดหาย — กรุณาลองใหม่อีกครั้ง',
                });
            };
            device.addEventListener('gattserverdisconnected', onDrop);

            (async () => {
                try {
                    const server         = await device.gatt.connect();
                    const service        = await server.getPrimaryService(BLERoomScanner.SERVICE_UUID);
                    const characteristic = await service.getCharacteristic(BLERoomScanner.CHARACTERISTIC_UUID);
                    const raw            = await characteristic.readValue();
                    const room           = new TextDecoder().decode(raw).trim();
                    if (!room) {
                        finish({ ok: false, code: 'empty_value',
                                 error: 'อ่านค่าจากอุปกรณ์ไม่สำเร็จ (ค่าว่าง) — กรุณาลองใหม่' });
                        return;
                    }
                    finish({ ok: true, room });
                } catch (err) {
                    finish({ ok: false, code: 'connection_dropped',
                             error: 'ไม่สามารถเชื่อมต่อหรืออ่านค่าจากอุปกรณ์ได้ — กรุณาลองใหม่อีกครั้ง' });
                }
            })();
        });
    }
}
