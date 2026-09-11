// All EAR consumers use actual input pixels, never CSS preview dimensions.
const EARMetric = Object.freeze({
    version: 'pixel-v1',
    eye(lm, indices, width, height) {
        if (!(Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0)) return NaN;
        const points = indices.map(i => lm?.[i]);
        if (points.some(p => !p || !Number.isFinite(p.x) || !Number.isFinite(p.y))) return NaN;
        const d = (a, b) => Math.hypot((a.x - b.x) * width, (a.y - b.y) * height);
        const [a, b, c, e, f, g] = points;
        const span = d(a, e);
        if (span <= 1e-6) return NaN;
        const value = (d(b, g) + d(c, f)) / (2 * span);
        return Number.isFinite(value) && value > 0 && value < 1 ? value : NaN;
    },
    mean(lm, video) {
        return (this.eye(lm, [33,160,158,133,153,144], video?.videoWidth, video?.videoHeight) +
            this.eye(lm, [362,385,387,263,373,380], video?.videoWidth, video?.videoHeight)) / 2;
    },
});

// Attempt-local calibration: never compare corrected EAR with an unversioned
// database scalar. Open hold -> blink -> open hold guards against calibrating
// a single arbitrary frame. This is a UI quality check, not server liveness proof.
class EARCalibration {
    constructor() { this.reset(); }
    reset() {
        this.phase = 'open'; this.samples = []; this.baseline = null;
        this.since = null; this.lastAt = null; this.dimensions = null;
    }
    update(ear, quality, width, height, now) {
        const dims = `${width}x${height}`;
        if (dims !== this.dimensions || (this.lastAt !== null && now - this.lastAt > 1000)) this.reset();
        this.dimensions = dims; this.lastAt = now;
        if (!quality || !Number.isFinite(ear) || ear <= 0 || ear >= 1) {
            this.reset();
            return false;
        }
        if (this.phase === 'ready') return true;
        if (this.phase === 'blink') {
            if (now - this.since > 8000) { this.reset(); return false; }
            if (ear < this.baseline * 0.70) { this.phase = 'reopen'; this.since = now; this.samples = []; }
            return false;
        }
        if (this.phase === 'reopen') {
            if (now - this.since > 3000) { this.reset(); return false; }
            if (ear >= this.baseline * 0.85) this.samples.push(ear);
            else this.samples = [];
            if (this.samples.length >= 5) { this.phase = 'ready'; return true; }
            return false;
        }
        if (this.since === null) this.since = now;
        this.samples.push(ear);
        this.samples = this.samples.slice(-60);
        if (this.samples.length >= 15 && now - this.since >= 1200) {
            const sorted = [...this.samples].sort((a, b) => a - b);
            const median = sorted[Math.floor(sorted.length / 2)];
            const spread = sorted[Math.floor(sorted.length * .9)] - sorted[Math.floor(sorted.length * .1)];
            if (spread > median * .15) { this.reset(); return false; }
            this.baseline = median; this.phase = 'blink'; this.since = now;
        }
        return false;
    }
    get instruction() {
        if (this.phase === 'blink') return 'กะพริบตาหนึ่งครั้งเพื่อยืนยันการวัด';
        if (this.phase === 'reopen') return 'ลืมตาตามปกติและมองตรง';
        return 'กำลังวัดค่าดวงตา — ลืมตาตามปกติ มองตรง และอยู่นิ่ง';
    }
}
