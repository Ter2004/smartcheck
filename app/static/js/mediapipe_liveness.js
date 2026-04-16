/**
 * mediapipe_liveness.js — Liveness & Interactive Challenge Detection
 *
 * Classes:
 *   LivenessDetector       — single-action liveness (backward-compat, used in check-in)
 *   InteractiveChallengeDetector — 2-action sequential challenge (enrollment step 3)
 *
 * Supported actions:
 *   'blink'         — กระพริบตา (EAR detection)
 *   'smile'         — ยิ้ม (mouth-width ratio)
 *   'turn_left'     — หันซ้าย (nose yaw < 0.38)
 *   'turn_right'    — หันขวา (nose yaw > 0.62)
 *   'nod'           — พยักหน้า (pitch swing ≥15°)
 *   'raise_eyebrows'— ยกคิ้ว (brow-to-eye distance +20%)
 */

// ─── Shared landmark indices ──────────────────────────────────────────────────
const LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380];

// ─── Shared helpers ───────────────────────────────────────────────────────────
function dist2D(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
}

function calcEARFromLM(lm, indices) {
    if (!lm || lm.length < 468) return 0.25;   // R4: safe default open-eye EAR
    const [p1, p2, p3, p4, p5, p6] = indices.map(i => lm[i]);
    return (dist2D(p2, p6) + dist2D(p3, p5)) / (2.0 * dist2D(p1, p4));
}

/** Yaw: nose x position relative to face width (0=hard left, 0.5=centre, 1=hard right) */
function noseRelX(lm) {
    if (!lm || lm.length < 468) return 0.5;
    const faceW = lm[454].x - lm[234].x;
    return faceW > 0 ? (lm[1].x - lm[234].x) / faceW : 0.5;
}

/** Pitch proxy: nose y relative to forehead-chin span */
function nosePitch(lm) {
    if (!lm || lm.length < 468) return 0.5;
    const faceH = lm[152].y - lm[10].y;
    return faceH > 0 ? (lm[1].y - lm[10].y) / faceH : 0.5;
}

// ─── Action labels (Thai) ─────────────────────────────────────────────────────
const ACTION_LABELS = {
    blink:          'กะพริบตา',
    smile:          'ยิ้มให้กว้าง',
    turn_left:      'หันหน้าไปทางซ้าย',
    turn_right:     'หันหน้าไปทางขวา',
    nod:            'พยักหน้าขึ้น-ลง',
    raise_eyebrows: 'ยกคิ้วขึ้น',
};

/** Build a stateful checker for one action. Returns { check(lm) → { done, statusText } } */
function buildActionChecker(action, baselineEAR) {
    const EAR_THRESHOLD = (baselineEAR || 0.25) * 0.70;

    if (action === 'blink') {
        let blinkDown = false;
        return {
            check(lm) {
                const ear = (calcEARFromLM(lm, LEFT_EYE_IDX) + calcEARFromLM(lm, RIGHT_EYE_IDX)) / 2;
                if (!blinkDown && ear < EAR_THRESHOLD) {
                    blinkDown = true;
                    return { done: false, statusText: 'กะพริบตา...' };
                }
                if (blinkDown && ear >= EAR_THRESHOLD) {
                    return { done: true, statusText: '✓ กะพริบตาสำเร็จ!' };
                }
                return { done: false, statusText: `กรุณากะพริบตา (EAR: ${ear.toFixed(3)})` };
            }
        };
    }

    if (action === 'smile') {
        // Detect sustained smile (no neutral-return requirement — too hard in practice)
        const SMILE_ON    = 0.40;   // ratio above this = smiling
        const SMILE_FRAMES = 2;    // 2 frames sufficient (was 4 — too slow)
        let smileFrames = 0;
        return {
            check(lm) {
                const mouthW = dist2D(lm[61], lm[291]);
                const faceW  = dist2D(lm[234], lm[454]);
                const ratio  = faceW > 0 ? mouthW / faceW : 0;
                if (ratio > SMILE_ON) {
                    smileFrames++;
                    if (smileFrames >= SMILE_FRAMES) return { done: true, statusText: '✓ ยิ้มสำเร็จ!' };
                    return { done: false, statusText: `😊 ยิ้มค้างไว้... (${smileFrames}/${SMILE_FRAMES})` };
                }
                smileFrames = 0;
                return { done: false, statusText: `กรุณายิ้มให้กว้าง (${ratio.toFixed(3)})` };
            }
        };
    }

    if (action === 'turn_left') {
        // In mirrored selfie view: user turns left → nose moves to screen-right → relX increases
        const TURN_THRESHOLD = 0.62;
        const TURN_FRAMES    = 3;   // 3 frames (was 5 — reduced for faster response)
        let turnFrames = 0;
        return {
            check(lm) {
                const rel = noseRelX(lm);
                if (rel > TURN_THRESHOLD) {
                    turnFrames++;
                    if (turnFrames >= TURN_FRAMES) return { done: true, statusText: '✓ หันซ้ายสำเร็จ!' };
                    return { done: false, statusText: `กำลังหัน... (${turnFrames}/${TURN_FRAMES})` };
                }
                turnFrames = 0;
                return { done: false, statusText: `กรุณาหันหน้าไปทางซ้าย (${rel.toFixed(2)})` };
            }
        };
    }

    if (action === 'turn_right') {
        // user turns right → nose moves to screen-left → relX decreases
        const TURN_THRESHOLD = 0.38;
        const TURN_FRAMES    = 3;   // 3 frames (was 5 — reduced for faster response)
        let turnFrames = 0;
        return {
            check(lm) {
                const rel = noseRelX(lm);
                if (rel < TURN_THRESHOLD) {
                    turnFrames++;
                    if (turnFrames >= TURN_FRAMES) return { done: true, statusText: '✓ หันขวาสำเร็จ!' };
                    return { done: false, statusText: `กำลังหัน... (${turnFrames}/${TURN_FRAMES})` };
                }
                turnFrames = 0;
                return { done: false, statusText: `กรุณาหันหน้าไปทางขวา (${rel.toFixed(2)})` };
            }
        };
    }

    if (action === 'nod') {
        // Detect pitch swing: neutral → down (pitchRel > 0.57) → back (< 0.51)
        // Lowered thresholds — typical neutral pitch is 0.50–0.55, so 0.60 was too far
        let nodDown = false;
        return {
            check(lm) {
                const pitch = nosePitch(lm);
                if (!nodDown && pitch > 0.57) {
                    nodDown = true;
                    return { done: false, statusText: 'กำลังก้มหน้า...' };
                }
                if (nodDown && pitch < 0.51) {
                    return { done: true, statusText: '✓ พยักหน้าสำเร็จ!' };
                }
                return { done: false, statusText: `กรุณาพยักหน้าขึ้น-ลง (pitch: ${pitch.toFixed(2)})` };
            }
        };
    }

    if (action === 'raise_eyebrows') {
        // Brow-to-eye distance using both inner (65,295) and outer (107,336) brow landmarks
        // vs eye-top landmarks (159,386). Calibrate from first CALIB_FRAMES frames (mean).
        // Detect ≥8% increase sustained 3 frames.
        const CALIB_FRAMES    = 6;
        const RAISE_THRESHOLD = 0.08;   // 8% (was 12%)
        const RAISE_FRAMES    = 3;
        const calibSamples    = [];
        let baselineDist      = null;
        let raisedFrames      = 0;
        return {
            check(lm) {
                const faceH = dist2D(lm[10], lm[152]);
                if (faceH === 0) return { done: false, statusText: 'กรุณายกคิ้ว' };

                // Average inner + outer brow landmarks for more robust measurement
                const leftBrowDist  = (dist2D(lm[65],  lm[159]) + dist2D(lm[107], lm[159])) / 2 / faceH;
                const rightBrowDist = (dist2D(lm[295], lm[386]) + dist2D(lm[336], lm[386])) / 2 / faceH;
                const currentDist   = (leftBrowDist + rightBrowDist) / 2;

                // Collect baseline from first CALIB_FRAMES frames (mean = stable neutral)
                if (baselineDist === null) {
                    calibSamples.push(currentDist);
                    if (calibSamples.length < CALIB_FRAMES) {
                        return { done: false, statusText: 'กรุณาทำหน้าปกติสักครู่...' };
                    }
                    baselineDist = calibSamples.reduce((a, b) => a + b, 0) / calibSamples.length;
                    return { done: false, statusText: 'กรุณายกคิ้วขึ้น' };
                }

                const increase = (currentDist - baselineDist) / (baselineDist + 1e-6);
                if (increase > RAISE_THRESHOLD) {
                    raisedFrames++;
                    if (raisedFrames >= RAISE_FRAMES) return { done: true, statusText: '✓ ยกคิ้วสำเร็จ!' };
                    return { done: false, statusText: `กำลังยกคิ้ว... (${raisedFrames}/${RAISE_FRAMES})` };
                }
                raisedFrames = 0;
                return { done: false, statusText: `กรุณายกคิ้วขึ้น (+${(increase * 100).toFixed(0)}%)` };
            }
        };
    }

    return { check: () => ({ done: false, statusText: 'ไม่รู้จัก action' }) };
}

// ─── RigidBodyGuard ───────────────────────────────────────────────────────────
/**
 * Detect photo replay by measuring rigid-body translation:
 * if all landmarks move together with very low variance → flat image being moved.
 */
class RigidBodyGuard {
    constructor() {
        this._prevPos    = null;
        this._rigidCount = 0;
        this._IDX = [1, 10, 33, 133, 152, 234, 263, 362, 454, 61, 291];
    }

    check(lm) {
        const pts = this._IDX.map(i => ({ x: lm[i].x, y: lm[i].y }));
        if (!this._prevPos) { this._prevPos = pts; return { spoof: false, reason: null }; }

        const diffs  = pts.map((p, j) => ({ dx: p.x - this._prevPos[j].x, dy: p.y - this._prevPos[j].y }));
        this._prevPos = pts;

        const n      = diffs.length;
        const meanDx = diffs.reduce((s, d) => s + d.dx, 0) / n;
        const meanDy = diffs.reduce((s, d) => s + d.dy, 0) / n;
        const movement = Math.abs(meanDx) + Math.abs(meanDy);
        const varDx  = diffs.reduce((s, d) => s + (d.dx - meanDx) ** 2, 0) / n;
        const varDy  = diffs.reduce((s, d) => s + (d.dy - meanDy) ** 2, 0) / n;
        const rigidity = Math.sqrt(varDx + varDy);

        if (movement > 0.010 && rigidity < 0.0018) {
            this._rigidCount++;
        } else {
            this._rigidCount = Math.max(0, this._rigidCount - 1);
        }
        if (this._rigidCount >= 6) {
            return { spoof: true, reason: 'ตรวจพบการโยกรูปถ่าย — กรุณาใช้ใบหน้าจริง' };
        }
        return { spoof: false, reason: null };
    }
}

// ─── LivenessDetector (single action — backward compat, used in check-in) ────
class LivenessDetector {
    constructor(video, canvas, baselineEAR) {
        this.video       = video;
        this.canvas      = canvas;
        this.baselineEAR = baselineEAR;
        this._camera     = null;
        this._faceMesh   = null;
    }

    stop() {
        if (this._camera)   { try { this._camera.stop();   } catch(e) {} }
        if (this._faceMesh) { try { this._faceMesh.close(); } catch(e) {} }
    }

    run(action, onStatus = () => {}) {
        return new Promise((resolve) => {
            let resolved = false;
            const TIMEOUT_MS = 20000;

            const faceMesh = new FaceMesh({ locateFile: f =>
                `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}` });
            faceMesh.setOptions({
                maxNumFaces: 1, refineLandmarks: true,
                minDetectionConfidence: 0.7, minTrackingConfidence: 0.7,
            });
            this._faceMesh = faceMesh;

            const checker = buildActionChecker(action, this.baselineEAR);
            const guard   = new RigidBodyGuard();
            const ctx     = this.canvas.getContext('2d');

            faceMesh.onResults((results) => {
                this.canvas.width  = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
                ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

                if (!results.multiFaceLandmarks?.length) {
                    onStatus('ไม่พบใบหน้า — จัดหน้าให้อยู่ในกรอบ');
                    return;
                }

                const lm = results.multiFaceLandmarks[0];

                // Real-time Moiré + edge check (defined in rt_analyze.js)
                if (typeof _rtAnalyzeFrame === 'function') {
                    const rt = _rtAnalyzeFrame(this.video, lm);
                    if (rt && rt.blocked && !resolved) {
                        resolved = true;
                        this.stop();
                        resolve({ pass: false, action, error: rt.reason });
                        return;
                    }
                    // non-blocking moireAlert/edgeAlert: don't skip — let action checker run
                }

                const guardResult = guard.check(lm);
                if (guardResult.spoof && !resolved) {
                    resolved = true;
                    this.stop();
                    resolve({ pass: false, action, error: guardResult.reason });
                    return;
                }

                const { done, statusText } = checker.check(lm);
                onStatus(statusText);

                if (done && !resolved) {
                    resolved = true;
                    const snap = document.createElement('canvas');
                    snap.width  = this.video.videoWidth  || 640;
                    snap.height = this.video.videoHeight || 480;
                    snap.getContext('2d').drawImage(this.video, 0, 0);
                    const capturedFrame = snap.toDataURL('image/jpeg', 0.85);
                    this.stop();
                    resolve({ pass: true, action, error: null, capturedFrame });
                }
            });

            const camera = new Camera(this.video, {
                onFrame: async () => { await faceMesh.send({ image: this.video }); },
                width: 640, height: 480,
            });
            this._camera = camera;
            camera.start().catch(() => {
                if (!resolved) {
                    resolved = true;
                    this.stop();
                    resolve({ pass: false, action, error: 'ไม่สามารถเปิดกล้องได้ กรุณาอนุญาตการใช้กล้อง' });
                }
            });

            setTimeout(() => {
                if (!resolved) {
                    this.stop();
                    resolve({ pass: false, action, error: `หมดเวลา — กรุณา${ACTION_LABELS[action] || action}อีกครั้ง` });
                }
            }, TIMEOUT_MS);
        });
    }
}

// ─── InteractiveChallengeDetector (2-action sequential — used in enrollment) ─
/**
 * Runs N actions in strict sequence inside a SINGLE camera session.
 * Sequence validation is enforced: action[0] must complete before action[1] starts.
 *
 * Pass options.faceMesh to share an existing FaceMesh instance (singleton pattern).
 * If omitted, a new FaceMesh is created and owned by this detector.
 */
class InteractiveChallengeDetector {
    /**
     * @param {HTMLVideoElement} video
     * @param {HTMLCanvasElement} canvas
     * @param {number} baselineEAR
     * @param {Function} onProgress  — callback(actionIndex, totalActions, statusText)
     * @param {Object}   options     — { faceMesh?: FaceMesh }
     */
    constructor(video, canvas, baselineEAR, onProgress = () => {}, options = {}) {
        this.video       = video;
        this.canvas      = canvas;
        this.baselineEAR = baselineEAR;
        this.onProgress  = onProgress;
        this._sharedFM   = options.faceMesh || null;  // external shared instance (not owned)
        this._camera     = null;
        this._faceMesh   = null;  // only set when we create our own instance
    }

    stop() {
        if (this._camera)   { try { this._camera.stop();   } catch(e) {} }
        if (this._faceMesh) { try { this._faceMesh.close(); } catch(e) {} }  // only close owned FM
    }

    /**
     * @param {string[]} actions — ordered list of actions to perform
     * @param {number} perActionTimeout — ms per action (default 8000)
     * @returns {Promise<{ pass: boolean, sequence: string[], failedAt: string|null, error: string|null }>}
     */
    run(actions, perActionTimeout = 8000) {
        return new Promise((resolve) => {
            let currentIdx = 0;
            let resolved   = false;
            let checker    = buildActionChecker(actions[0], this.baselineEAR);
            const guard    = new RigidBodyGuard();
            let actionTimer = null;

            const fail = (reason) => {
                if (resolved) return;
                resolved = true;
                clearTimeout(actionTimer);
                this.stop();
                resolve({ pass: false, sequence: actions, failedAt: actions[currentIdx], error: reason });
            };

            const startActionTimer = () => {
                clearTimeout(actionTimer);
                actionTimer = setTimeout(() => {
                    fail(`หมดเวลา — กรุณา${ACTION_LABELS[actions[currentIdx]] || actions[currentIdx]}ให้เร็วขึ้น`);
                }, perActionTimeout);
            };

            // Use shared FaceMesh if provided, otherwise create and own one
            // Confidence 0.5 (was 0.7): lower threshold reduces detection lag on mobile
            // without meaningful accuracy loss for the 6 supported actions.
            // refineLandmarks: false — iris model adds ~15 ms/frame and is unused here.
            let faceMesh;
            if (this._sharedFM) {
                faceMesh = this._sharedFM;
                faceMesh.setOptions({
                    maxNumFaces: 1, refineLandmarks: false,
                    minDetectionConfidence: 0.5, minTrackingConfidence: 0.5,
                });
                // _faceMesh stays null → stop() will not close the shared instance
            } else {
                faceMesh = new FaceMesh({ locateFile: f =>
                    `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}` });
                faceMesh.setOptions({
                    maxNumFaces: 1, refineLandmarks: false,
                    minDetectionConfidence: 0.5, minTrackingConfidence: 0.5,
                });
                this._faceMesh = faceMesh;  // owned → stop() will close it
            }

            const ctx = this.canvas.getContext('2d');

            faceMesh.onResults((results) => {
                if (resolved) return;
                this.canvas.width  = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
                ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

                if (!results.multiFaceLandmarks?.length) {
                    this.onProgress(currentIdx, actions.length, 'ไม่พบใบหน้า — จัดหน้าให้อยู่ในกรอบ');
                    return;
                }

                const lm = results.multiFaceLandmarks[0];

                // Real-time Moiré + edge check (defined in rt_analyze.js)
                if (typeof _rtAnalyzeFrame === 'function') {
                    const rt = _rtAnalyzeFrame(this.video, lm);
                    if (rt && rt.blocked) { fail(rt.reason); return; }
                    // non-blocking moireAlert/edgeAlert: don't skip — let action checker run
                }

                const guardResult = guard.check(lm);
                if (guardResult.spoof) { fail(guardResult.reason); return; }

                const { done, statusText } = checker.check(lm);
                this.onProgress(currentIdx, actions.length, statusText);

                if (!done) return;

                // Current action completed → advance to next
                currentIdx++;
                if (currentIdx >= actions.length) {
                    // All actions completed in correct sequence
                    resolved = true;
                    clearTimeout(actionTimer);
                    this.stop();
                    resolve({ pass: true, sequence: actions, failedAt: null, error: null });
                } else {
                    // Start next action
                    checker = buildActionChecker(actions[currentIdx], this.baselineEAR);
                    startActionTimer();
                    this.onProgress(currentIdx, actions.length, ACTION_LABELS[actions[currentIdx]] || actions[currentIdx]);
                }
            });

            const camera = new Camera(this.video, {
                onFrame: async () => { await faceMesh.send({ image: this.video }); },
                width: 640, height: 480,
            });
            this._camera = camera;
            camera.start()
                .then(() => { startActionTimer(); })
                .catch(() => { fail('ไม่สามารถเปิดกล้องได้ กรุณาอนุญาตการใช้กล้อง'); });
        });
    }
}

// ─── Random challenge helpers ─────────────────────────────────────────────────

const CHALLENGE_POOL = ['blink', 'smile', 'turn_left', 'turn_right', 'nod', 'raise_eyebrows'];

/** Pick N unique actions at random from the full pool */
function randomChallengeActions(count = 2) {
    const pool = [...CHALLENGE_POOL];
    const result = [];
    while (result.length < count && pool.length > 0) {
        const i = Math.floor(Math.random() * pool.length);
        result.push(pool.splice(i, 1)[0]);
    }
    return result;
}

/** Legacy: pick 1 action (used by check-in hybrid liveness) */
function randomLivenessAction() {
    return randomChallengeActions(1)[0];
}

/** Legacy: pick 2 actions (backward compat) */
function randomLivenessActions() {
    return randomChallengeActions(2);
}
