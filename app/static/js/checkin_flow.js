/**
 * checkin_flow.js — SmartCheck Check-in Flow
 *
 * Flow: BLE read / TOTP → server preflight → camera → receipt-backed submit.
 */

class CheckinFlow {
    // Gestures the server may request (app/services/liveness_challenge.py ACTIONS).
    static GESTURE_LABELS = {
        turn_left:  'หันหน้าไปทางซ้าย',
        turn_right: 'หันหน้าไปทางขวา',
        tilt_left:  'เอียงศีรษะไปทางซ้าย (หูซ้ายเข้าหาไหล่)',
        tilt_right: 'เอียงศีรษะไปทางขวา (หูขวาเข้าหาไหล่)',
    };
    // Asks for more than the server's TILT_MIN_DEG, so frames it accepts pass there.
    static TILT_DEG = 18;
    static LEVEL_DEG = 8;

    /** Eye-line angle (degrees) on the raw frame; positive = image-right eye lower = tilt_left. */
    static eyeRollDeg(lm, width, height) {
        let a = lm[33], b = lm[263];
        if (a.x > b.x) [a, b] = [b, a];
        return Math.atan2((b.y - a.y) * (height || 480), (b.x - a.x) * (width || 640)) * 180 / Math.PI;
    }

    // Turn measured like the server; same formula and threshold as eyeYaw/TURN_YAW in
    // mediapipe_liveness.js (noseRelX 0.65 read only ~0.11 server-side, LFW 2026-10-04).
    static TURN_YAW = 0.30;

    /** Nose offset from the eye midpoint along the eye line, in eye distances, in pixels.
     *  Positive = nose toward the image's right = turn_left. */
    static eyeYaw(lm, width, height) {
        const w = width || 640, h = height || 480;
        const c = (i, j) => ({x: (lm[i].x + lm[j].x) / 2 * w, y: (lm[i].y + lm[j].y) / 2 * h});
        let a = c(33, 133), b = c(362, 263);
        if (a.x > b.x) [a, b] = [b, a];
        const ax = b.x - a.x, ay = b.y - a.y, d = ax * ax + ay * ay;
        if (d < 1e-6) return 0;
        return ((lm[1].x * w - (a.x + b.x) / 2) * ax + (lm[1].y * h - (a.y + b.y) / 2) * ay) / d;
    }

    constructor(opts) {
        this.sessionId     = opts.sessionId;
        this.beaconUUID    = opts.beaconUUID;
        this.apiUrl        = opts.apiUrl || '/api/checkin';
        this.proximityMethod = opts.proximityMethod || 'ble';
        this.expectedRoom  = typeof opts.expectedRoom === 'string' ? opts.expectedRoom : '';
        // 'head_turn': one server-chosen turn, re-checked by the server; 'passive': legacy.
        this.livenessMode  = opts.livenessMode === 'passive' ? 'passive' : 'head_turn';
        this.livenessChallengeUrl = opts.livenessChallengeUrl || '/api/checkin/liveness/challenge';
        this.lineReceipt = typeof opts.lineReceipt === 'string' && opts.lineReceipt ? opts.lineReceipt : null;
        this.lineRoom    = typeof opts.lineRoom === 'string' ? opts.lineRoom : '';
        this.sessionCodeEnabled = opts.sessionCodeEnabled === true;
        this.sessionCodeFromUrl = this.sessionCodeEnabled && /^[0-9]{6}$/.test(opts.sessionCodeFromUrl || '')
            ? opts.sessionCodeFromUrl : null;
        this._liveness     = null;

        this._camStream  = null;
        this._proximity = null;
        this._debug = opts.debug === true ? new CheckinDebug() : null;
    }

    async start() {
        if (!document.getElementById('stepRoomCode')) return;
        this._preloadFaceMesh();
        if (this.lineReceipt) {
            // LINE Beacon already proved the room (server-issued receipt in the link):
            // skip the Bluetooth/room-code step and open the camera. 300 s matches
            // line_beacon.RECEIPT_TTL_SECONDS; the server enforces the real limit.
            this._proximity = {room: this.lineRoom, receipt: this.lineReceipt, via: 'line',
                deadline: performance.now() + 300 * 1000, captureStarted: performance.now()};
            await this._startVerify();
            return;
        }
        if (this.sessionCodeFromUrl) {
            // Opened from the QR on the teacher's screen: confirm that code right away.
            this._requestRoomCode();
            this._submitting = true;
            try {
                await this._verifyProximity('', null, 'code', {session_code: this.sessionCodeFromUrl});
            } finally {
                this._submitting = false;
            }
            return;
        }
        // Proximity (room code or BLE room read) always gates the camera.
        if (this.proximityMethod === 'broadcast') {
            // The app scans the room board by itself: no instructions to tap away.
            this._requestRoomCode();
            return;
        }
        await StepGuide.show('📍', 'ยืนยันห้องเรียนก่อนเช็คชื่อ',
            this.proximityMethod === 'ble'
                ? 'เปิด Bluetooth แล้วกดหาอุปกรณ์ในห้อง เลือกอุปกรณ์ของห้องเรียน เมื่อยืนยันสำเร็จระบบจะเปิดกล้อง'
                : 'ดูรหัส 6 หลักจากจอในห้องเรียน กรอกรหัสแล้วกดตรวจรหัสและเริ่มถ่ายภาพ');
        this._requestRoomCode();
    }

    // ─── Step dots ───────────────────────────────────────

    _goToStep(n) {
        this._debug?.mount(n);
        const title = document.getElementById('checkinStageTitle');
        const help = document.getElementById('checkinStageHelp');
        const scansBoard = this.proximityMethod === 'ble' || this.proximityMethod === 'broadcast';
        if (title) title.textContent = ['', scansBoard ? 'หาสัญญาณห้องเรียน' : 'ยืนยันรหัสห้องเรียน', 'ยืนยันตัวตนด้วยใบหน้า', 'กำลังตรวจสอบ'][n];
        if (help) help.textContent = ['', 'ตรวจสอบว่าคุณอยู่ในห้องเรียนของคาบที่เลือก', 'จัดใบหน้าในกรอบ ระบบจะถ่ายภาพและเช็คชื่อให้อัตโนมัติ', 'ระบบตรวจสอบใบหน้าและบันทึกการเข้าเรียน'][n];
        const band = document.getElementById('checkinBand');
        if (band) band.dataset.state = 'boarding';
        // Camera and checking: hide the tab bar and back link so a thumb cannot
        // leave the scan by accident (the result screen brings them back).
        document.body.classList.toggle('ci-focus', n >= 2);
        // 1=proximity, 2=camera, 3=result.
        const firstStep = {ble: 'stepBleRoom', broadcast: 'stepBroadcastRoom'}[this.proximityMethod] || 'stepRoomCode';
        const steps = [firstStep, 'stepVerify', 'stepDone'];
        steps.forEach((id, i) => {
            const el = document.getElementById(id);
            if (el) el.style.display = (i + 1 === n) ? 'block' : 'none';
        });
        // The teacher-code fallback belongs to step 1 only.
        const codeForm = document.getElementById('stepSessionCode');
        if (codeForm && n !== 1) codeForm.style.display = 'none';
        for (let i = 1; i <= 3; i++) {
            const dot = document.getElementById('dot' + i);
            if (dot) {
                dot.classList.toggle('active', i <= n);
                dot.classList.toggle('done',   i < n);
            }
        }
    }

    // Step 2: Detect a steady frontal face, then (head_turn mode) one server-chosen
    // head turn and a look back, and submit automatically.

    // Still frontal frames before the capture (~0.7 s at the 66 ms pump). The server
    // still runs the face match and anti-spoof on the frame; this only picks a steady one.
    static READY_FRAMES = 10;

    _newFaceMesh() {
        const mesh = new FaceMesh({ locateFile: f =>
            `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/${f}` });
        mesh.setOptions({
            maxNumFaces: 1, refineLandmarks: false,
            minDetectionConfidence: 0.7, minTrackingConfidence: 0.7,
        });
        return mesh;
    }

    /** Load the face model while the room is being confirmed, so the camera does not wait for it. */
    _preloadFaceMesh() {
        if (this._meshPreload || typeof FaceMesh === 'undefined') return;
        try {
            const mesh = this._newFaceMesh();
            // The first send() builds the WebGL graph (1.4 s on an OPPO, 2026-11-05).
            // Do it on a blank frame while the board is being found, so detection starts
            // with the camera's first frame. _startVerify sets the real onResults later.
            mesh.onResults(() => {});
            const ready = Promise.resolve(mesh.initialize()).then(() => {
                const blank = document.createElement('canvas');
                blank.width = blank.height = 64;
                return mesh.send({image: blank});
            });
            ready.catch(() => {});  // _startVerify sees the failure and shows it
            this._meshPreload = {mesh, ready};
        } catch (_) { /* _startVerify loads it again */ }
    }

    async _fetchLivenessChallenge() {
        const res = await fetch(this.livenessChallengeUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': document.querySelector('meta[name="csrf-token"]')?.content || '',
            },
            body: '{}',
        });
        if (!res.ok) throw new Error(`challenge HTTP ${res.status}`);
        const body = await res.json();
        if (!body || typeof body.nonce !== 'string' || !Array.isArray(body.actions) || body.actions.length !== 1) {
            throw new Error('challenge response invalid');
        }
        return body;
    }

    async _startVerify() {
        if (!this._proximity || performance.now() >= this._proximity.deadline) {
            this._requestRoomCode();
            return;
        }
        this._stopStream(this._camStream);
        const generation = this._captureGeneration;
        const active = () => generation === this._captureGeneration && !!this._proximity;
        let faceReadyFrames = 0, countingDown = false, verified = false;
        // head_turn phases: 'frontal' → 'turn' → 'return' → submit
        let phase = 'frontal', turnCount = 0, backCount = 0, beforeFrame = null, turnFrame = null;
        let baselineRoll = 0;
        this._liveness = null;
        this._debug?.begin();
        this._goToStep(2);
        document.getElementById("stepRoomCode").style.display = "none";
        this._capturedFrames = [];

        const video    = document.getElementById('videoVerify');
        const canvas   = document.getElementById('canvasVerify');
        const guide    = document.getElementById('faceGuideVerify');
        const statusEl = document.getElementById('verifyStatus');
        // Write only when the words change: the live region re-announces every write,
        // and the face loop sets the same hint on almost every frame.
        const status = {
            get textContent() { return statusEl.textContent; },
            set textContent(t) { if (statusEl.textContent !== t) statusEl.textContent = t; },
        };
        const ring     = document.getElementById('faceProgressVerify');
        // Ring around the oval: 0..100 = how far the still-face count has got.
        const setProgress = p => { if (ring) ring.style.strokeDashoffset = String(100 - p); };
        setProgress(0);

        let turnAction = null, challengeNonce = null;
        if (this.livenessMode === 'head_turn') {
            try {
                const challenge = await this._fetchLivenessChallenge();
                turnAction = challenge.actions[0];
                challengeNonce = challenge.nonce;
            } catch (e) {
                if (!active()) return;
                console.info('step=liveness_challenge result=error details={}');
                this._goToStep(3);
                document.getElementById('doneLoadingView').style.display = 'none';
                document.getElementById('doneResultView').style.display = 'block';
                this._showDone('error', 'เชื่อมต่อเซิร์ฟเวอร์ไม่สำเร็จ — กดลองใหม่', true);
                return;
            }
        }
        const turnLabel = CheckinFlow.GESTURE_LABELS[turnAction] || 'ทำตามคำสั่ง';
        const isTilt = !!turnAction && turnAction.startsWith('tilt_');
        if (!active()) return;
        if (performance.now() >= this._proximity.deadline) {
            this._expireProximity();
            return;
        }
        this._proximity.captureStarted = performance.now();

        // เปิดกล้อง
        try {
            if (this._meshCleanup) await this._meshCleanup;
            if (!active()) return;
            const acquired = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'user', width: 640, height: 480 }
            });
            if (!active()) { acquired.getTracks().forEach(t => t.stop()); return; }
            this._camStream = acquired;
            if (!this._proximity || performance.now() >= this._proximity.deadline) {
                this._stopStream(this._camStream);
                this._expireProximity();
                return;
            }
            const vcCheck = await detectVirtualCamera(this._camStream);
            if (!active()) return;
            if (vcCheck.blocked) {
                this._camStream.getTracks().forEach(t => t.stop());
                const vc = new Error('Virtual Camera Detected');
                vc.name = 'VirtualCameraError';
                throw vc;
            }
            video.srcObject = this._camStream;
        } catch (e) {
            if (!active()) return;
            this._stopStream(this._camStream);
            console.info('step=camera_open result=error details={}');
            status.textContent = CheckinFlow.cameraErrorText(e);
            this._goToStep(3);
            document.getElementById('doneLoadingView').style.display = 'none';
            document.getElementById('doneResultView').style.display = 'block';
            this._showDone('error', status.textContent, true);
            return;
        }

        status.textContent = 'เตรียมกล้อง — จัดใบหน้าให้อยู่ในกรอบวงรี';

        // Timeout: ถ้า 40 วินาทีแล้วยังไม่พบใบหน้า ให้ปิดกล้องและแสดงข้อผิดพลาด
        this._debug?.arm();
        const captureDeadline = performance.now() + 40000;
        const expireCapture = () => {
            if (!active()) return;
            this._debug?.fired();
            if (!verified && !countingDown) {
                const reason = status.textContent;
                this._stopStream(this._camStream);
                status.textContent = 'หมดเวลาถ่ายภาพ — ' + reason;
                guide.classList.remove('ok');
                guide.classList.add('fail');
                console.info('step=capture_timeout result=reject details={}');
                this._goToStep(3);
                document.getElementById('doneLoadingView').style.display = 'none';
                document.getElementById('doneResultView').style.display = 'block';
                this._showDone('error', status.textContent, true);
            }
        };
        this._streamTimeoutId = setTimeout(expireCapture, 40000);
        const snapFrame = () => {
            const snap = document.createElement('canvas');
            snap.width  = video.videoWidth  || 640;
            snap.height = video.videoHeight || 480;
            snap.getContext('2d').drawImage(video, 0, 0);
            return snap.toDataURL('image/jpeg', 0.85);
        };

        status.textContent = 'จัดใบหน้าให้อยู่ในกรอบวงรี';

        // FaceMesh ตรวจตำแหน่งหน้า (the one loaded during the room step, if any)
        const preload = this._meshPreload;
        this._meshPreload = null;
        const faceMesh = preload ? preload.mesh : this._newFaceMesh();
        this._activeMesh = faceMesh;

        const processResults = async results => {
            if (!active() || verified || countingDown) return;
            if (performance.now() >= captureDeadline) { expireCapture(); return; }
            if (!video.videoWidth || !video.videoHeight) return;

            canvas.width  = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // ─── ตรวจ brightness ─────────────────────────────
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
            let totalBrightness = 0;
            for (let i = 0; i < pixels.length; i += 16) {
                totalBrightness += 0.299 * pixels[i] + 0.587 * pixels[i+1] + 0.114 * pixels[i+2];
            }
            const avgBrightness = totalBrightness / (pixels.length / 16);
            if (avgBrightness < 60) {
                this._debug?.sample({width: video.videoWidth, height: video.videoHeight,
                    ready: 0, faces: results.multiFaceLandmarks?.length || 0,
                    failed: ['brightness (other gates not evaluated)']});
                guide.classList.remove('ok');
                faceReadyFrames = 0;
                setProgress(0);
                status.textContent = 'แสงน้อยเกินไป — กรุณาอยู่ในพื้นที่ที่มีแสงสว่างเพียงพอ';
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                return;
            }

            const hasFace = results.multiFaceLandmarks?.length > 0;
            if (!hasFace) {
                this._debug?.sample({width: video.videoWidth, height: video.videoHeight,
                    ready: 0, faces: 0, failed: ['hasFace (geometry not evaluated)']});
                guide.classList.remove('ok', 'fail');
                faceReadyFrames = 0;
                setProgress(0);
                status.textContent = 'ไม่พบใบหน้า — จัดหน้าให้อยู่ในกรอบ';
                return;
            }

            const lm = results.multiFaceLandmarks[0];
            let minX = 1, maxX = 0, minY = 1, maxY = 0;
            for (const p of lm) {
                if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
                if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
            }
            const faceW = maxX - minX, faceH = maxY - minY;
            const nose  = lm[1];

            const inCenter    = nose.x > 0.25 && nose.x < 0.75 && nose.y > 0.2 && nose.y < 0.8;
            const closeEnough = faceW > 0.28 && faceH > 0.36;
            const noseRatio   = faceH > 0 ? (nose.y - minY) / faceH : 0.5;
            const pitchOk     = noseRatio > 0.30 && noseRatio < 0.70;
            const eyeDiff     = Math.abs(lm[33].y - lm[362].y);
            const rollOk      = faceH > 0 && (eyeDiff / faceH) < 0.10;
            const faceSpanX   = lm[454].x - lm[234].x;
            const noseRelX    = faceSpanX > 0 ? (nose.x - lm[234].x) / faceSpanX : 0.5;
            const yawOk       = noseRelX > 0.38 && noseRelX < 0.62;

            // Preserve a short real camera burst for the existing temporal check.
            const frame = await this._captureFrame(video);
            if (!active()) return;
            this._capturedFrames.push(frame);
            if (this._capturedFrames.length > 3) this._capturedFrames.shift();

            // ปากต้องหุบ
            const mouthOpen = Math.hypot(lm[13].x - lm[14].x, lm[13].y - lm[14].y);
            const mouthOk   = faceH > 0 && (mouthOpen / faceH) < 0.10;

            const faceOk = inCenter && closeEnough && pitchOk && rollOk && yawOk;
            this._debug?.sample({width: video.videoWidth, height: video.videoHeight,
                faceW, faceH,
                ready: faceOk ? faceReadyFrames + 1 : 0,
                faces: results.multiFaceLandmarks.length,
                failed: Object.entries({inCenter, closeEnough, pitchOk, rollOk, yawOk, mouthOk})
                    .filter(([, passed]) => !passed).map(([name]) => name)});

            const rollDelta = CheckinFlow.eyeRollDeg(lm, video.videoWidth, video.videoHeight) - baselineRoll;
            if (phase === 'turn') {
                // Same thresholds as mediapipe_liveness.js (raw frame coordinates). A tilt
                // is measured from the frontal frame and must not also be a turn.
                const yaw = CheckinFlow.eyeYaw(lm, video.videoWidth, video.videoHeight);
                const turned = isTilt
                    ? noseRelX > 0.35 && noseRelX < 0.65 &&
                      (turnAction === 'tilt_left' ? rollDelta >= CheckinFlow.TILT_DEG : rollDelta <= -CheckinFlow.TILT_DEG)
                    : turnAction === 'turn_left' ? yaw > CheckinFlow.TURN_YAW : yaw < -CheckinFlow.TURN_YAW;
                turnCount = turned ? turnCount + 1 : 0;
                guide.classList.remove('fail');
                guide.classList.add('ok');
                status.textContent = turned ? '✓ ค้างไว้สักครู่...' : `${turnLabel} แล้วค้างไว้สักครู่`;
                if (turnCount >= 3) {
                    turnFrame = snapFrame();
                    phase = 'return';
                    status.textContent = 'มองตรงเข้ากล้องอีกครั้ง';
                }
                return;
            }
            if (phase === 'return') {
                const level = !isTilt || Math.abs(rollDelta) < CheckinFlow.LEVEL_DEG;
                backCount = yawOk && level ? backCount + 1 : 0;
                if (backCount < 5) {
                    status.textContent = 'มองตรงเข้ากล้องอีกครั้ง';
                    return;
                }
                this._debug?.event('head turn complete; submitting');
                countingDown = true;
                clearTimeout(this._streamTimeoutId);
                verified = true;
                status.textContent = '✓ ครบแล้ว — กำลังเช็คชื่อ...';
                const afterFrame = snapFrame();
                this._stopStream(this._camStream);
                this._liveness = {nonce: challengeNonce, turnFrames: [turnFrame], after: afterFrame};
                await this._submitCheckin(beforeFrame, 'head_turn');
                return;
            }

            if (faceOk) {
                guide.classList.remove('fail');
                guide.classList.add('ok');
                faceReadyFrames++;
                setProgress(Math.min(100, faceReadyFrames / CheckinFlow.READY_FRAMES * 100));
                status.textContent = '✓ พบใบหน้า — กรุณานิ่งสักครู่...';

                if (faceReadyFrames >= CheckinFlow.READY_FRAMES && this.livenessMode === 'head_turn') {
                    // The frontal frame the server matches; the turn follows.
                    this._debug?.event(`${faceReadyFrames} ready frames; head turn requested`);
                    beforeFrame = snapFrame();
                    baselineRoll = CheckinFlow.eyeRollDeg(lm, video.videoWidth, video.videoHeight);
                    phase = 'turn';
                    status.textContent = `${turnLabel} แล้วค้างไว้สักครู่`;
                } else if (faceReadyFrames >= CheckinFlow.READY_FRAMES) {
                    this._debug?.event(`${faceReadyFrames} ready frames; capture complete`);
                    countingDown = true;
                    clearTimeout(this._streamTimeoutId);
                    verified = true;
                    // Submit to the existing server face/anti-spoof checks.
                    status.textContent = '✓ พบใบหน้า — กำลังเช็คชื่อ...';
                    const snap = document.createElement('canvas');
                    snap.width  = video.videoWidth  || 640;
                    snap.height = video.videoHeight || 480;
                    snap.getContext('2d').drawImage(video, 0, 0);
                    const capturedFrame = snap.toDataURL('image/jpeg', 0.85);
                    this._stopStream(this._camStream);
                    await this._submitCheckin(capturedFrame, 'passive');
                }
            } else {
                guide.classList.remove('ok');
                guide.classList.add('fail');
                faceReadyFrames = 0;
                setProgress(0);
                if (!closeEnough)  status.textContent = 'เข้าใกล้กล้องอีกหน่อย';
                else if (!mouthOk) status.textContent = 'กรุณาหุบปาก';
                else if (!pitchOk) status.textContent = 'กรุณาอย่าก้มหรือเงยหน้า';
                else if (!rollOk)  status.textContent = 'กรุณาอย่าเอียงศีรษะ';
                else if (!yawOk)   status.textContent = 'กรุณามองตรงเข้ากล้อง';
                else               status.textContent = 'ขยับหน้าให้อยู่กลางกรอบ';
            }
        };

        // Reuse the acquired stream; Camera.start() would acquire a second one.
        // Serialize model sends and async result work, including error handling.
        let resultsWork = Promise.resolve();
        faceMesh.onResults(results => {
            resultsWork = processResults(results);
            resultsWork.catch(() => {});
            return resultsWork;
        });
        const fail = stage => {
            if (!active()) return;
            this._debug?.event(`camera failure: ${stage}`);
            this._stopStream(this._camStream);
            this._goToStep(3);
            document.getElementById('doneLoadingView').style.display = 'none';
            document.getElementById('doneResultView').style.display = 'block';
            this._showDone('error', 'กล้องประมวลผลไม่สำเร็จ — กรุณาลองถ่ายภาพใหม่', true);
        };
        const pump = async () => {
            if (!active() || verified) return;
            try {
                if (performance.now() >= this._proximity.deadline) { this._expireProximity(); return; }
                if (performance.now() >= captureDeadline) { expireCapture(); return; }
                if (video.readyState >= 2 && video.videoWidth && video.videoHeight) {
                    this._meshWork = Promise.resolve(faceMesh.send({image: video}));
                    await this._meshWork;
                    await resultsWork;
                }
            } catch (_) { fail('frame'); return; }
            if (active() && !verified) this._frameTimer = setTimeout(pump, 66);
        };
        try {
            await video.play();
            if (!active()) return;
            this._meshWork = preload ? preload.ready : Promise.resolve(faceMesh.initialize());
            await this._meshWork;
            if (active()) await pump();
        } catch (_) { fail('startup'); }
    }

    async _captureFrame(videoEl) {
        const c = document.createElement('canvas');
        c.width  = videoEl.videoWidth;
        c.height = videoEl.videoHeight;
        c.getContext('2d').drawImage(videoEl, 0, 0);
        return c.toDataURL('image/jpeg', 0.85);
    }

    // ─── Submit ──────────────────────────────────────────

    _requestRoomCode() {
        StepGuide.dismiss();
        clearInterval(this._receiptTimer);
        this._stopStream(this._camStream);
        this._proximity = null;
        this._goToStep(1);
        document.getElementById('proximityStatus').textContent = '';
        document.getElementById('receiptCountdown').textContent = '';
        document.getElementById('stepVerify').style.display = 'none';
        document.getElementById('stepDone').style.display = 'none';
        if (this.proximityMethod === 'ble') {
            document.getElementById('stepBleRoom').style.display = 'block';
            document.getElementById('bleRoomStatus').textContent = '';
            const warn = document.getElementById('bleUnsupportedWarning');
            const btn  = document.getElementById('bleRoomBtn');
            if (!navigator.bluetooth) {
                // Covers any unsupported browser, not just the iOS UA sniff
                // that may already have shown a warning server-side.
                warn.textContent = this.sessionCodeEnabled
                    ? 'เบราว์เซอร์นี้ไม่รองรับ Web Bluetooth — ใช้รหัสจากจอของอาจารย์ด้านล่างแทน'
                    : 'เบราว์เซอร์นี้ไม่รองรับ Web Bluetooth — กรุณาใช้ Chrome หรือ Edge บนคอมพิวเตอร์ Windows เพื่อเช็คชื่อ';
                warn.style.display = 'block';
                btn.disabled = true;
                btn.textContent = 'ไม่รองรับ Bluetooth';
            } else {
                warn.style.display = 'none';
                btn.disabled = false;
                btn.textContent = 'หาอุปกรณ์ในห้อง';
            }
        } else if (this.proximityMethod === 'broadcast') {
            document.getElementById('stepBroadcastRoom').style.display = 'block';
            document.getElementById('broadcastStatus').textContent = '';
            document.getElementById('broadcastRetryBtn').style.display = 'none';
            const warn = document.getElementById('broadcastUnsupportedWarning');
            if (typeof BroadcastRoomScanner !== 'undefined' && BroadcastRoomScanner.available()) {
                warn.style.display = 'none';
                this.startBroadcastScan();
            } else {
                warn.textContent = this.sessionCodeEnabled
                    ? 'การหาสัญญาณห้องอัตโนมัติใช้ได้ในแอป SmartCheck บน Android — บนเครื่องนี้ให้ใช้รหัสจากจอของอาจารย์ด้านล่าง'
                    : 'การหาสัญญาณห้องอัตโนมัติใช้ได้ในแอป SmartCheck บน Android เท่านั้น — กรุณาเปิดหน้านี้ในแอป หรือแจ้งอาจารย์';
                warn.style.display = 'block';
            }
        } else {
            document.getElementById('stepRoomCode').style.display = 'block';
            document.getElementById('roomCode').value = '';
            document.getElementById('roomCode').focus();
        }
        const codeForm = document.getElementById('stepSessionCode');
        if (codeForm && this.sessionCodeEnabled) {
            codeForm.style.display = 'block';
            document.getElementById('sessionCode').value = '';
        }
    }

    async submitSessionCode(event) {
        event.preventDefault();
        const input = document.getElementById('sessionCode');
        if (!input.reportValidity() || this._submitting) return;
        this._submitting = true;
        try {
            await this._verifyProximity('', null, 'code', {session_code: input.value});
        } finally {
            this._submitting = false;
        }
    }

    async submitRoomCode(event) {
        event.preventDefault();
        const input = document.getElementById('roomCode');
        if (!input.reportValidity() || this._submitting) return;
        this._submitting = true;
        try {
            await this._verifyProximity(input.value);
        } finally {
            this._submitting = false;
        }
    }

    async startBleRoomScan() {
        if (this._submitting) return;
        this._submitting = true;
        const btn    = document.getElementById('bleRoomBtn');
        const status = document.getElementById('bleRoomStatus');
        btn.disabled = true;
        status.textContent = 'กำลังเปิดตัวเลือกอุปกรณ์ Bluetooth...';

        let result;
        try {
            result = await new BLERoomScanner().proveRoom(room => this._fetchBleChallenge(room));
        } catch (error) {
            console.info('step=ble_read result=error details={}');
            result = {ok: false, error: 'ไม่สามารถเชื่อมต่อหรืออ่านค่าจากอุปกรณ์ได้ — กรุณาลองใหม่อีกครั้ง'};
        }
        this._submitting = false;

        if (!result.ok) {
            console.info('step=ble_read result=reject details=' + JSON.stringify({code: result.code || 'read_error'}));
            status.textContent = result.error;
            btn.disabled = false;
            btn.textContent = 'ลองใหม่';
            return;
        }

        status.textContent = 'อ่านค่าห้องแล้ว — กำลังตรวจสอบก่อนเปิดกล้อง...';
        this._submitting = true;
        try {
            await this._verifyProximity(result.room, result.response);
        } finally {
            this._submitting = false;
        }
    }

    // Own guard, not _submitting: _verifyProximity stays pending through the whole
    // camera step, and an expired receipt must be able to start a fresh scan.
    async startBroadcastScan() {
        if (this._broadcastScanning) return;
        this._broadcastScanning = true;
        const status = document.getElementById('broadcastStatus');
        document.getElementById('broadcastRetryBtn').style.display = 'none';
        status.textContent = 'กำลังหาสัญญาณห้องเรียน...';
        document.querySelector('.ci-radar')?.classList.remove('is-idle');
        // Board first: no camera, no face scan until the phone is in the room's range.
        let result;
        try {
            result = await new BroadcastRoomScanner().findRoom(this.expectedRoom);
        } catch {
            result = {ok: false, code: 'scan_error', error: 'หาสัญญาณห้องเรียนไม่สำเร็จ — กรุณาลองใหม่'};
        } finally {
            this._broadcastScanning = false;
        }
        if (!result.ok) {
            console.info('step=broadcast_scan result=reject details=' + JSON.stringify({code: result.code || 'scan_error'}));
            status.textContent = result.code === 'not_found'
                ? 'ไม่พบสัญญาณห้องเรียน — กรุณาอยู่ภายในระยะสัญญาณของห้องเรียน แล้วกด "ลองใหม่"'
                : result.error;
            document.getElementById('broadcastRetryBtn').style.display = 'block';
            // Not searching any more: stop the rings so the screen does not look busy.
            document.querySelector('.ci-radar')?.classList.add('is-idle');
            return;
        }
        console.info('step=broadcast_scan result=pass details={}');
        status.textContent = 'พบสัญญาณห้อง ' + result.room;
        // The camera opens now; the server checks the board code meanwhile.
        await this._verifyProximity(result.room, null, null, {broadcast_payload: result.payload}, true);
    }

    /** Pending proximity (no receipt yet) and the camera starting: {pending, settle, capture}. */
    _openCameraEarly(room = null) {
        const started = performance.now();
        let settle = () => {};
        const receiptReady = new Promise(resolve => { settle = resolve; });
        const pending = {room, receipt: null, receiptReady,
            deadline: started + CheckinFlow.EARLY_RECEIPT_MS, captureStarted: started};
        this._proximity = pending;
        return {pending, settle, capture: this._startVerify()};
    }

    async _fetchBleChallenge(room) {
        try {
            const res = await fetch('/api/checkin/ble/challenge', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-Token': document.querySelector('meta[name="csrf-token"]')?.content || '',
                },
                body: JSON.stringify({session_id: this.sessionId, room_code: room}),
            });
            const body = await res.json();
            if (res.ok && body.ok && typeof body.nonce === 'string') return {ok: true, nonce: body.nonce};
            return {ok: false, error: body.error || 'ขอรหัสยืนยันจากระบบไม่สำเร็จ กรุณาลองใหม่'};
        } catch {
            return {ok: false, error: 'ไม่สามารถเชื่อมต่อระบบได้ กรุณาลองใหม่'};
        }
    }

    // early (broadcast board): the camera starts while the server checks the board
    // code, and the check-in itself waits for the receipt (_submitCheckin). Pass true
    // to open the camera here, or the {pending, settle, capture} of _openCameraEarly.
    static EARLY_RECEIPT_MS = 90000;

    async _verifyProximity(room, bleResponse, via = null, extra = {}, early = false) {
        const status = document.getElementById('proximityStatus');
        status.textContent = 'กำลังตรวจสอบตำแหน่งห้องเรียน...';
        const started = performance.now();
        let issued = false;
        let pending = null, settle = () => {}, capture = null;
        if (early) {
            ({pending, settle, capture} = early === true ? this._openCameraEarly(room) : early);
            pending.room = room;
        }
        try {
            const response = await fetch('/api/checkin/proximity', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-Token': document.querySelector('meta[name="csrf-token"]')?.content || '',
                },
                body: JSON.stringify({session_id: this.sessionId, room_code: room,
                                      ...(bleResponse ? {ble_response: bleResponse} : {}),
                                      ...(via ? {proximity_via: via} : {}), ...extra}),
            });
            const result = await response.json();
            if (early && this._proximity !== pending) return;   // the attempt was replaced meanwhile
            if (!response.ok || !result.ok) {
                console.info('step=proximity_preflight result=reject details=' + JSON.stringify({status: response.status}));
                if (early) this._earlyRefused();
                status.textContent = result.error || 'ตรวจสอบตำแหน่งไม่สำเร็จ กรุณาลองใหม่';
                return;
            }
            if (early) {
                Object.assign(pending, {receipt: result.proximity_receipt,
                    deadline: started + result.expires_in * 1000});
            } else {
                this._proximity = {room, receipt: result.proximity_receipt, ...(via ? {via} : {}),
                    deadline: started + result.expires_in * 1000, captureStarted: performance.now()};
            }
            issued = true;
            settle(true);
            if (this._debug) {
                this._debug.receiptDue = this._proximity.deadline;
                this._debug.event('new receipt');
            }
            status.textContent = 'ยืนยันห้องแล้ว';
            // The room receipt lasts 90 s. Say so only near the end (last 15 s),
            // and write the timer only when the number changes.
            const countdown = document.getElementById('receiptCountdown');
            this._receiptTimer = setInterval(() => {
                if (!this._proximity) return;
                const seconds = Math.max(0, Math.ceil((this._proximity.deadline - performance.now()) / 1000));
                const text = seconds <= 15 ? `อีก ${seconds} วินาทีต้องหาสัญญาณห้องใหม่` : '';
                if (countdown.textContent !== text) countdown.textContent = text;
                if (!seconds) this._expireProximity();
            }, 250);
            await (capture || this._startVerify());
        } catch (error) {
            console.info('step=proximity_preflight result=error details={}');
            if (early && this._proximity === pending) this._earlyRefused();
            status.textContent = 'ไม่สามารถเชื่อมต่อระบบตรวจสอบตำแหน่งได้ กรุณาลองใหม่';
        } finally {
            settle(false);   // no-op once the receipt arrived
            if (this.proximityMethod === 'ble' && navigator.bluetooth) {
                document.getElementById('bleRoomBtn').disabled = false;
            }
            // Refused (no receipt): a new board code needs a new scan. An early attempt
            // that was replaced (e.g. expiry started a new scan) leaves the button alone.
            if (this.proximityMethod === 'broadcast' && !issued && via === null && !(early && this._proximity)) {
                document.getElementById('broadcastRetryBtn').style.display = 'block';
            }
        }
    }

    /** The board code was refused after the camera had opened: back to the room step. */
    _earlyRefused() {
        this._stopStream(this._camStream);
        this._proximity = null;
        this._goToStep(1);
        document.getElementById('broadcastStatus').textContent = '';
    }

    _expireProximity() {
        this._debug?.event('receipt expired');
        console.info('step=proximity_receipt_expired result=reject details={}');
        this._requestRoomCode();
        document.getElementById('proximityStatus').textContent = 'ผลยืนยันตำแหน่งห้องเรียนหมดอายุ กรุณายืนยันใหม่ก่อนเช็คชื่อ';
    }

    retry() {
        if (this._debug) {
            this._debug.retries++;
            this._debug.event('user pressed Retry');
        }
        if (this._retryRoomCode || !this._proximity || performance.now() >= this._proximity.deadline) {
            this._requestRoomCode();
        } else {
            this._startVerify();
        }
    }

    async _submitCheckin(faceImage, livenessAction) {
        // Early camera: the face can be ready before the server has accepted the board code.
        const pending = this._proximity;
        if (pending && pending.receipt === null) {
            document.getElementById('verifyStatus').textContent = '✓ ถ่ายแล้ว — กำลังยืนยันห้องเรียน...';
            const accepted = await pending.receiptReady;
            if (!accepted || this._proximity !== pending) return;   // _earlyRefused already reset the page
        }
        if (!this._proximity || performance.now() >= this._proximity.deadline) {
            this._expireProximity();
            return;
        }
        clearInterval(this._receiptTimer);
        console.info('step=capture result=complete details=' + JSON.stringify({elapsed_ms: Math.round(performance.now() - this._proximity.captureStarted)}));
        this._debug?.event('submit check-in');
        document.getElementById("stepRoomCode").style.display = "none";
        document.getElementById("stepBleRoom").style.display = "none";
        this._retryRoomCode = false;
        this._goToStep(3);
        document.getElementById('doneLoadingView').style.display  = 'block';
        const loadingMessage = document.querySelector('#doneLoadingView p');
        if (loadingMessage) loadingMessage.textContent = 'กำลังเปรียบเทียบใบหน้ากับข้อมูลที่ลงทะเบียน กรุณารอและอย่าปิดหน้านี้';
        document.getElementById('doneResultView').style.display   = 'none';
        this._barStart();

        try {
            const deviceToken = localStorage.getItem('sc_device_token');
            const res = await fetch(this.apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type':  'application/json',
                    'X-CSRF-Token':  document.querySelector('meta[name="csrf-token"]')?.content || '',
                    ...(deviceToken ? { 'Authorization': `DeviceToken ${deviceToken}` } : {}),
                },
                body: JSON.stringify({
                    session_id:      this.sessionId,
                    room_code:       this._proximity.room,
                    proximity_receipt: this._proximity.receipt,
                    ...(this._proximity.via ? {proximity_via: this._proximity.via} : {}),
                    liveness_action: livenessAction,
                    liveness_pass:   true,
                    face_image:      faceImage,
                    face_images:     this._capturedFrames || [faceImage],
                    // head_turn only: the server re-checks these against face_image.
                    liveness_nonce:       this._liveness?.nonce,
                    liveness_turn_frames: this._liveness?.turnFrames,
                    liveness_after:       this._liveness?.after,
                }),
            });

            const json = await this._readCheckinResponse(res);
            this._barStop();
            document.getElementById('doneLoadingView').style.display = 'none';
            document.getElementById('doneResultView').style.display  = 'block';

            this._retryRoomCode = json.retry_room_code === true;
            if (json.ok || json.already_checked) this._clearNotification();
            if (json.ok) {
                this._showDone('success', json.message, false, json);
            } else if (json.already_checked) {
                this._showDone('info', json.error, false, json);
            } else if (json.spoof) {
                this._showDone('spoof', json.error, true);
            } else {
                this._showDone('error', json.error || json.message || 'เช็คชื่อไม่สำเร็จ', json.retry_face === true || this._retryRoomCode);
            }
        } catch (e) {
            this._barStop();
            document.getElementById('doneLoadingView').style.display = 'none';
            document.getElementById('doneResultView').style.display  = 'block';
            console.error('Check-in response failed', {name: e.name, status: e.httpStatus});
            this._showDone('error', e.userMessage || 'การเชื่อมต่อขาดหาย — ดูที่หน้าวันนี้ ถ้าคาบนี้ขึ้น "มา" พร้อมเวลาแสดงว่าบันทึกแล้ว ถ้ายังไม่ขึ้นให้กลับมาเช็คชื่อใหม่', false);
        }
    }

    /** Remove this class's "open" notification from the bar (push_notifications.js; app only). */
    _clearNotification() {
        if (typeof SmartCheckPush !== 'undefined') SmartCheckPush.clear(this.sessionId).catch(() => {});
    }

    async _readCheckinResponse(res) {
        const fail = (message) => {
            const error = new Error('Invalid check-in response');
            error.userMessage = message;
            error.httpStatus = res.status;
            throw error;
        };
        if (res.status === 401 || (res.redirected && new URL(res.url, location.href).pathname === '/login')) {
            fail('เซสชันหมดอายุ — เข้าสู่ระบบใหม่ แล้วดูที่หน้าวันนี้ว่าคาบนี้บันทึกแล้วหรือยัง');
        }
        let json;
        try { json = await res.json(); } catch (_) {
            if (res.status === 429) fail('คำขอมากเกินไป กรุณารอสักครู่แล้วโหลดหน้าใหม่');
            if ([502, 503, 504, 520, 521, 522, 523, 524, 530].includes(res.status)) {
                fail(`เชื่อมต่อเซิร์ฟเวอร์ไม่สำเร็จ (HTTP ${res.status}) — ดูที่หน้าวันนี้ว่าคาบนี้บันทึกแล้วหรือยัง ก่อนลองใหม่`);
            }
            fail(`เซิร์ฟเวอร์ตอบกลับผิดรูปแบบ (HTTP ${res.status}) — ดูที่หน้าวันนี้ว่าคาบนี้บันทึกแล้วหรือยัง`);
        }
        if (!json || typeof json !== 'object' || Array.isArray(json)) {
            fail(`เซิร์ฟเวอร์ตอบกลับผิดรูปแบบ (HTTP ${res.status}) กรุณาโหลดหน้าใหม่`);
        }
        if (!res.ok) json.ok = false;
        return json;
    }

    // ─── Result ──────────────────────────────────────────

    // Result: the pass band becomes the answer, like a gate display. Green "เช็คชื่อแล้ว"
    // (orange when late) with the recorded time; red when nothing was recorded; neutral
    // "ถ่ายใหม่" when the photo could not be confirmed as a live face (never "cheating").
    static DONE_ICONS = {
        check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path class="draw" d="M7.5 12.5l3 3 6-6.5"/></svg>',
        cross: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M15 9l-6 6M9 9l6 6"/></svg>',
        retake: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 0 1 15.4-6.4L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-15.4 6.4L3 16"/><path d="M3 21v-5h5"/></svg>',
    };

    /** "08:03" in Thai time from an ISO timestamp ('' when missing or bad). */
    static thaiTime(iso) {
        const d = new Date(iso);
        if (!iso || Number.isNaN(d.getTime())) return '';
        return new Intl.DateTimeFormat('en-GB', {timeZone: 'Asia/Bangkok', hour: '2-digit', minute: '2-digit', hourCycle: 'h23'}).format(d);
    }

    /** Camera failures in Thai, each with the step that fixes it. */
    static cameraErrorText(e) {
        const name = e && e.name;
        if (name === 'VirtualCameraError') return 'ตรวจพบกล้องเสมือน — กรุณาใช้กล้องหน้าของเครื่องจริง';
        if (name === 'NotAllowedError' || name === 'SecurityError') {
            return 'ยังไม่ได้อนุญาตให้ใช้กล้อง — เปิดการตั้งค่าเครื่อง > แอป > SmartCheck > สิทธิ์ > กล้อง เลือก "อนุญาต" แล้วกดลองใหม่';
        }
        if (name === 'NotFoundError' || name === 'OverconstrainedError') return 'ไม่พบกล้องหน้าในเครื่องนี้ — กรุณาใช้มือถือที่มีกล้องหน้า';
        if (name === 'NotReadableError' || name === 'AbortError') return 'กล้องถูกแอปอื่นใช้อยู่ — ปิดแอปที่ใช้กล้องแล้วกดลองใหม่';
        return 'เปิดกล้องไม่ได้ — ปิดแล้วเปิดแอปใหม่ แล้วลองอีกครั้ง';
    }

    _showDone(state, message, canRetry, result = {}) {
        const late = result.status === 'late';
        const time = CheckinFlow.thaiTime(result.check_in_at);
        const views = {
            success: {band: late ? 'late' : 'boarded', label: late ? 'เช็คชื่อแล้ว · มาสาย' : 'เช็คชื่อแล้ว · มาเรียน',
                      icon: 'check', tone: 'go', title: time || (late ? 'มาสาย' : 'มาเรียน'),
                      msg: `บันทึกแล้ว${late ? ' (มาสาย)' : ''} ไม่ต้องทำอะไรเพิ่ม`},
            info:    {band: late ? 'late' : 'boarded', label: 'เช็คชื่อแล้ว', icon: 'check', tone: 'go',
                      title: 'เช็คชื่อไว้แล้ว', msg: message || 'คาบนี้บันทึกไว้แล้ว ไม่ต้องทำอะไรเพิ่ม'},
            spoof:   {band: 'closed', label: 'ถ่ายใหม่อีกครั้ง', icon: 'retake', tone: 'muted',
                      title: 'ยังยืนยันใบหน้าไม่ได้',
                      msg: 'หันหน้าเข้าหาแสง ถือมือถือให้นิ่ง และใช้ใบหน้าจริง (ไม่ใช่รูปหรือหน้าจอ) แล้วกด "ลองใหม่"'},
            error:   {band: 'failed', label: 'ยังไม่ได้บันทึก', icon: 'cross', tone: 'stop',
                      title: 'เช็คชื่อไม่สำเร็จ', msg: message || 'กรุณาลองใหม่'},
        };
        const v = views[state] || views.error;
        document.body.classList.remove('ci-focus');
        const recorded = state === 'success' || state === 'info';

        const band = document.getElementById('checkinBand');
        const label = document.getElementById('checkinStageTitle');
        if (band) band.dataset.state = v.band;
        if (label) {
            label.textContent = v.label;
            label.classList.remove('turn'); void label.offsetWidth; label.classList.add('turn');
        }
        for (let i = 1; i <= 3; i++) {
            const dot = document.getElementById('dot' + i);
            if (dot) dot.classList.toggle('done', i < 3 || recorded);
        }
        // The room-receipt countdown means nothing once there is an answer.
        clearInterval(this._receiptTimer);
        document.getElementById('receiptCountdown').textContent = '';
        document.getElementById('proximityStatus').textContent = '';

        const iconEl  = document.getElementById('doneIcon');
        const titleEl = document.getElementById('doneTitle');
        iconEl.innerHTML = CheckinFlow.DONE_ICONS[v.icon];
        iconEl.dataset.tone = v.tone;
        titleEl.textContent = v.title;
        titleEl.classList.toggle('is-time', state === 'success' && !!time);
        document.getElementById('doneMsg').textContent = v.msg;
        document.getElementById('doneRetry').style.display = canRetry ? 'block' : 'none';
        // With no retry, going back to today is the one thing left to do.
        const back = document.getElementById('doneBack');
        if (back) back.className = canRetry ? 'btn-quiet' : 'btn-board';
        // Screen readers: land on the answer.
        titleEl.setAttribute('tabindex', '-1');
        titleEl.focus({preventScroll: true});
    }

    // ─── Helpers ─────────────────────────────────────────

    // Checking bar: an estimate easing toward 90% (the server took 1.65 s in the first
    // phone test, 2026-11-02), not real progress; the result screen replaces it.
    static BAR_TAU_MS = 1500;

    _barStart() {
        this._barStop();
        const bar = document.getElementById('doneBar');
        if (!bar) return;
        const t0 = performance.now();
        bar.style.width = '0%';
        this._barTimer = setInterval(() => {
            bar.style.width = `${(90 * (1 - Math.exp(-(performance.now() - t0) / CheckinFlow.BAR_TAU_MS))).toFixed(1)}%`;
        }, 100);
    }

    _barStop() {
        if (this._barTimer) { clearInterval(this._barTimer); this._barTimer = null; }
    }



    _stopStream(stream) {
        this._captureGeneration = (this._captureGeneration || 0) + 1;
        this._debug?.stopped();
        clearTimeout(this._streamTimeoutId);
        clearTimeout(this._frameTimer);
        if (this._activeMesh) {
            const mesh = this._activeMesh;
            this._activeMesh = null;
            this._meshCleanup = Promise.resolve(this._meshWork).catch(() => {}).then(() => mesh.close())
                .catch(() => this._debug?.event('camera cleanup failed'));
        }
        if (stream) stream.getTracks().forEach(t => t.stop());
        this._camStream = null;
        // H3: also stop MediaPipe Camera instance if stored
        if (this._faceMeshCam) {
            const cam = this._faceMeshCam;
            this._faceMeshCam = null;
            Promise.resolve().then(() => cam.stop()).catch(() => this._debug?.event('camera stop failed'));
        }
    }

    _sleep(ms) {
        return new Promise(r => setTimeout(r, ms));
    }
}
