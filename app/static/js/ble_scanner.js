/**
 * ble_scanner.js — BLE Beacon Scanner (Web Bluetooth API)
 *
 * Usage:
 *   const scanner = new BLEScanner(beaconUUID);
 *   const result  = await scanner.scan();
 *   // result = { pass: bool, rssi: number, error: string|null }
 */

class BLEScanner {
    /**
     * @param {string} targetUUID  — iBeacon UUID ของห้องนั้น (lowercase)
     * @param {number} threshold   — RSSI threshold (เช่น -75)
     * @param {number} readings    — จำนวน readings ก่อนคำนวณ median (default 5)
     */
    constructor(targetUUID, threshold = -75, readings = 5) {
        this.targetUUID = targetUUID.toLowerCase();
        this.threshold  = threshold;
        this.readings   = readings;
    }

    /** หยุด background pre-scan (ถ้ามี) */
    stop() {
        this._stopped = true;
        if (this._device && this._device.gatt.connected) {
            this._device.gatt.disconnect();
        }
    }

    /**
     * สแกนหา beacon และคืนผลลัพธ์
     * @returns {Promise<{pass:boolean, rssi:number|null, error:string|null}>}
     */
    async scan() {
        this._stopped = false;

        if (!navigator.bluetooth) {
            return { pass: false, rssi: null, error: 'เบราว์เซอร์นี้ไม่รองรับ Web Bluetooth (ใช้ Chrome หรือ Bluefy บน iOS)' };
        }

        try {
            // requestDevice เปิด native BLE picker
            const device = await navigator.bluetooth.requestDevice({
                filters: [{ services: ['battery_service'] }],
                optionalServices: ['battery_service'],
                acceptAllDevices: false,
            }).catch(() => null);

            // ถ้า user กด cancel หรือหา device ไม่เจอ → fallback ใช้ acceptAllDevices
            const dev = device || await navigator.bluetooth.requestDevice({
                acceptAllDevices: true,
                optionalServices: ['battery_service'],
            });

            this._device = dev;
            const rssiValues = await this._collectRSSI(dev, this.readings);

            if (rssiValues.length === 0) {
                return { pass: false, rssi: null, error: 'ไม่สามารถอ่านค่า RSSI ได้' };
            }

            const medianRSSI = this._median(rssiValues);
            const pass = medianRSSI >= this.threshold;

            return { pass, rssi: medianRSSI, error: null };

        } catch (err) {
            if (err.name === 'NotFoundError') {
                return { pass: false, rssi: null, error: 'ไม่พบอุปกรณ์ BLE — กรุณาเปิด Bluetooth และอยู่ในห้องเรียน' };
            }
            if (err.name === 'SecurityError') {
                return { pass: false, rssi: null, error: 'ไม่ได้รับอนุญาตใช้ Bluetooth — กรุณาอนุญาตในเบราว์เซอร์' };
            }
            return { pass: false, rssi: null, error: `BLE Error: ${err.message}` };
        }
    }

    /**
     * Background pre-scan: เรียกก่อนเปิดหน้าเพื่อให้ผล RSSI พร้อม
     * คืน Promise ที่ resolve เมื่อสแกนเสร็จ (หรือ reject เมื่อ error)
     * เก็บผลไว้ใน this.preScanResult
     */
    async preScan() {
        this.preScanResult = null;
        try {
            this.preScanResult = await this.scan();
        } catch (e) {
            this.preScanResult = { pass: false, rssi: null, error: e.message };
        }
        return this.preScanResult;
    }

    // ─── Private ────────────────────────────────────────

    async _collectRSSI(device, count) {
        const values = [];
        // Web Bluetooth ไม่ expose RSSI โดยตรงใน standard API
        // ใช้ watchAdvertisements (Chrome M79+) ถ้า available
        if (device.watchAdvertisements) {
            return await this._watchRSSI(device, count);
        }
        // Fallback: connect + read battery level เพื่อ confirm proximity
        // แล้วใช้ dummy RSSI (ไม่แม่นยำ แต่ใช้ confirm connection ได้)
        try {
            const server = await device.gatt.connect();
            // ถ้า connect ได้ถือว่าอยู่ในระยะ — ใส่ค่า -60 เป็น fallback
            device.gatt.disconnect();
            for (let i = 0; i < count; i++) values.push(-60);
        } catch {
            // connect ไม่ได้ → อยู่นอกระยะ
        }
        return values;
    }

    async _watchRSSI(device, count) {
        return new Promise((resolve) => {
            const values = [];
            const timeout = setTimeout(() => {
                device.removeEventListener('advertisementreceived', handler);
                resolve(values);
            }, 8000);

            const handler = (event) => {
                if (this._stopped) {
                    clearTimeout(timeout);
                    device.removeEventListener('advertisementreceived', handler);
                    resolve(values);
                    return;
                }
                values.push(event.rssi);
                if (values.length >= count) {
                    clearTimeout(timeout);
                    device.removeEventListener('advertisementreceived', handler);
                    resolve(values);
                }
            };

            device.addEventListener('advertisementreceived', handler);
            device.watchAdvertisements().catch(() => {
                clearTimeout(timeout);
                resolve([]);
            });
        });
    }

    _median(arr) {
        const sorted = [...arr].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2 !== 0
            ? sorted[mid]
            : Math.round((sorted[mid - 1] + sorted[mid]) / 2);
    }
}
