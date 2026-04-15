/**
 * rt_analyze.js — Shared real-time Moiré + edge spoof detection
 *
 * Included by BOTH enrollment (enroll_face.html) and check-in (checkin.html)
 * so that _rtAnalyzeFrame is available in mediapipe_liveness.js during check-in.
 *
 * H5 fix: previously defined only in enrollment_flow.js, so check-in liveness
 * challenge silently skipped all Moiré/edge checks.
 */

// ── Canvas shared across all checks (avoids repeated createElement) ──────────
const _RT_CANVAS = document.createElement('canvas');
_RT_CANVAS.width = 64; _RT_CANVAS.height = 64;
const _RT_CTX = _RT_CANVAS.getContext('2d', { willReadFrequently: true });

const RT_THROTTLE_MS   = 250;   // max analysis rate (ms between runs)
const RT_CONSEC_THRESH = 4;     // consecutive alert frames before blocking
const RT_MOIRE_CV_MAX  = 0.20;  // CV below this → screen (uniform texture)
const RT_MOIRE_SAD_MIN = 2.5;   // min mean SAD — skip near-blank/dark frames
const RT_EDGE_RATIO    = 4.5;   // Sobel peak/average ratio to flag a straight edge

let _rtLastCheck   = 0;
let _rtMoireConsec = 0;
let _rtEdgeConsec  = 0;

function _rtResetCounters() { _rtMoireConsec = 0; _rtEdgeConsec = 0; }

/**
 * Analyse one video frame for Moiré + straight-edge spoof indicators.
 * @param {HTMLVideoElement} video
 * @param {Array} landmarks  — MediaPipe face landmark array
 * @returns {null | { moireAlert, edgeAlert, blocked, reason, sadCV, sadMean }}
 */
function _rtAnalyzeFrame(video, landmarks) {
    const now = Date.now();
    if (now - _rtLastCheck < RT_THROTTLE_MS) return null;
    _rtLastCheck = now;

    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return null;

    // ── Face bounding box ──────────────────────────────────────────────────
    let minX = 1, maxX = 0, minY = 1, maxY = 0;
    for (const p of landmarks) {
        if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
    }
    const faceX = minX * vw, faceY = minY * vh;
    const faceW = (maxX - minX) * vw, faceH = (maxY - minY) * vh;
    if (faceW < 30 || faceH < 30) return null;

    // ── 1. Moiré: row-SAD coefficient of variation ─────────────────────────
    _RT_CTX.drawImage(video, faceX, faceY, faceW, faceH, 0, 0, 64, 64);
    const fp = _RT_CTX.getImageData(0, 0, 64, 64).data;
    const gray = new Uint8Array(4096);
    for (let i = 0; i < 4096; i++) {
        gray[i] = (77 * fp[i*4] + 150 * fp[i*4+1] + 29 * fp[i*4+2]) >> 8;
    }

    const rowSAD = new Float32Array(64);
    for (let r = 0; r < 64; r++) {
        let s = 0;
        for (let c = 0; c < 63; c++) s += Math.abs(gray[r*64+c+1] - gray[r*64+c]);
        rowSAD[r] = s / 63;
    }
    const sadMean = rowSAD.reduce((a, b) => a + b, 0) / 64;
    const sadStd  = Math.sqrt(rowSAD.reduce((s, v) => s + (v - sadMean) ** 2, 0) / 64);
    const sadCV   = sadMean > RT_MOIRE_SAD_MIN ? sadStd / sadMean : 1.0;
    const moireAlert = sadCV < RT_MOIRE_CV_MAX && sadMean > RT_MOIRE_SAD_MIN;

    // ── 2. Edge detection: straight line at face-border zone ──────────────
    const pad  = 0.18;
    const expX = Math.max(0, faceX - faceW * pad);
    const expY = Math.max(0, faceY - faceH * pad);
    const expW = Math.min(vw - expX, faceW * (1 + 2 * pad));
    const expH = Math.min(vh - expY, faceH * (1 + 2 * pad));

    _RT_CTX.drawImage(video, expX, expY, expW, expH, 0, 0, 64, 64);
    const ep = _RT_CTX.getImageData(0, 0, 64, 64).data;
    const ge = new Int16Array(4096);
    for (let i = 0; i < 4096; i++) {
        ge[i] = (77 * ep[i*4] + 150 * ep[i*4+1] + 29 * ep[i*4+2]) >> 8;
    }

    const colSum = new Float32Array(64);
    const rowSum = new Float32Array(64);
    for (let r = 1; r < 63; r++) {
        for (let c = 1; c < 63; c++) {
            const sx = Math.abs(
                ge[(r-1)*64+c+1] + 2*ge[r*64+c+1] + ge[(r+1)*64+c+1] -
                ge[(r-1)*64+c-1] - 2*ge[r*64+c-1] - ge[(r+1)*64+c-1]
            );
            const sy = Math.abs(
                ge[(r+1)*64+c-1] + 2*ge[(r+1)*64+c] + ge[(r+1)*64+c+1] -
                ge[(r-1)*64+c-1] - 2*ge[(r-1)*64+c] - ge[(r-1)*64+c+1]
            );
            colSum[c] += sy;
            rowSum[r] += sx;
        }
    }

    const avgCol = colSum.reduce((a, b) => a + b, 0) / 64;
    const avgRow = rowSum.reduce((a, b) => a + b, 0) / 64;
    let edgeAlert = false;
    if (avgCol > 10 || avgRow > 10) {
        let maxColVal = 0, maxColIdx = 0, maxRowVal = 0, maxRowIdx = 0;
        for (let i = 0; i < 64; i++) {
            if (colSum[i] > maxColVal) { maxColVal = colSum[i]; maxColIdx = i; }
            if (rowSum[i] > maxRowVal) { maxRowVal = rowSum[i]; maxRowIdx = i; }
        }
        const colAtBorder = maxColIdx < 20 || maxColIdx > 43;
        const rowAtBorder = maxRowIdx < 20 || maxRowIdx > 43;
        edgeAlert = (avgCol > 10 && maxColVal > RT_EDGE_RATIO * avgCol && colAtBorder) ||
                    (avgRow > 10 && maxRowVal > RT_EDGE_RATIO * avgRow && rowAtBorder);
    }

    // ── Consecutive counters ──────────────────────────────────────────────
    _rtMoireConsec = moireAlert ? _rtMoireConsec + 1 : 0;
    _rtEdgeConsec  = edgeAlert  ? _rtEdgeConsec  + 1 : 0;

    const moireBlocked = _rtMoireConsec >= RT_CONSEC_THRESH;
    const edgeBlocked  = _rtEdgeConsec  >= RT_CONSEC_THRESH;
    const blocked = moireBlocked || edgeBlocked;

    return {
        moireAlert, edgeAlert, blocked,
        reason: moireBlocked
            ? 'ตรวจพบหน้าจอ (Moiré) — กรุณาใช้ใบหน้าจริงต่อหน้ากล้อง'
            : edgeBlocked
            ? 'ตรวจพบขอบวัตถุแปลกปลอม — กรุณานำโทรศัพท์/กระดาษออก'
            : null,
        sadCV: +sadCV.toFixed(3),
        sadMean: +sadMean.toFixed(1),
    };
}
