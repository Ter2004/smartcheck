/**
 * checkin_flow.js — SmartCheck Check-in Flow
 *
 * Flow (Final 1): Verify (face detect + capture) → Result
 * Flow (Phase 3+): BLE → Verify (face detect + liveness) → Result
 */

class CheckinFlow {
    constructor(opts) {
        this.sessionId     = opts.sessionId;
        this.beaconUUID    = opts.beaconUUID;
        this.rssiThreshold = opts.rssiThreshold;
        this.baselineEAR   = opts.baselineEAR;
        this.apiUrl        = opts.apiUrl || '/api/checkin';

        this._bleRSSI          = null;
        this._bleSkip          = false;
        this._camStream        = null;
        this._lastSpoofResult  = null;   // cache last passive spoof result for overlay
    }

    start() {
        // Final 1: ข้าม BLE — เริ่ม verify ใบหน้าทันที
        // TODO Phase 3: เปลี่ยนกลับเป็น this._goToStep(1) เมื่อมี Beacon จริง
        this._bleRSSI = null;
        this._bleSkip = true;
        this._startVerify();
    }

    // ─── Step dots ───────────────────────────────────────

    _goToStep(n) {
        // Final 1: BLE ถูกซ่อน — stepVerify=1, stepDone=2
        const steps = ['stepVerify', 'stepDone'];
        steps.forEach((id, i) => {
            const el = document.getElementById(id);
            if (el) el.style.display = (i + 1 === n) ? 'block' : 'none';
        });
        for (let i = 1; i <= 2; i++) {
            const dot = document.getElementById('dot' + i);
            if (dot) {
                dot.classList.toggle('active', i <= n);
                dot.classList.toggle('done',   i < n);
            }
        }
    }

    // ─── Step 1: BLE ─────────────────────────────────────

    skipBLE() {
        this._bleRSSI = -60;
        this._bleSkip = true;
        document.getElementById('bleStatus').textContent = '⚙️ ข้าม BLE (โหมดทดสอบ)';
        setTimeout(() => this._startVerify(), 400);
    }

    async startBLEScan() {
        const btn    = document.getElementById('bleBtn');
        const status = document.getElementById('bleStatus');
        btn.disabled = true;
        status.textContent = 'กำลังสแกน Bluetooth...';

        const scanner = new BLEScanner(this.beaconUUID, this.rssiThreshold);
        const result  = await scanner.scan();

        if (result.error) {
            status.textContent = result.error;
            btn.disabled = false;
            btn.textContent = 'ลองใหม่';
            return;
        }

        this._bleRSSI = result.rssi;

        if (!result.pass) {
            status.textContent = `อยู่นอกห้องเรียน — RSSI: ${result.rssi} dBm (ต้องการ ≥ ${this.rssiThreshold})`;
            btn.disabled = false;
            btn.textContent = 'สแกนใหม่';
            return;
        }

        status.textContent = `✓ พบ Beacon — RSSI: ${result.rssi} dBm`;
        setTimeout(() => this._startVerify(), 600);
    }

    // ─── Step 2: Verify (face detect → countdown → liveness) ─

    async _startVerify() {
        this._goToStep(2);

        const video    = document.getElementById('videoVerify');
        const canvas   = document.getElementById('canvasVerify');
        const guide    = document.getElementById('faceGuideVerify');
        const status   = document.getElementById('verifyStatus');
        const countdown = document.getElementById('countdownBadge');

        // เปิดกล้อง
        try {
            this._camStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'user', width: 640, height: 480 }
            });
            const vcCheck = await detectVirtualCamera(this._camStream);
            if (vcCheck.blocked) {
                this._camStream.getTracks().forEach(t => t.stop());
                status.textContent = 'ตรวจพบกล้องเสมือน — กรุณาใช้กล้องจริงเท่านั้น';
                alert(`ไม่อนุญาตให้ใช้กล้องเสมือน (${vcCheck.label}) — กรุณาใช้กล้องจริงเท่านั้น`);
                throw new Error('Virtual Camera Detected');
            }
            video.srcObject = this._camStream;
        } catch (e) {
            status.textContent = 'ไม่สามารถเปิดกล้องได้: ' + e.message;
            return;
        }

        status.textContent = 'เตรียมกล้อง — จัดใบหน้าให้อยู่ในกรอบวงรี';

        // Timeout: ถ้า 40 วินาทีแล้วยังไม่พบใบหน้า ให้ปิดกล้องและแสดงข้อผิดพลาด
        const _streamTimeoutId = setTimeout(() => {
            if (!verified && !countingDown) {
                faceMesh.close();
                this._stopStream(this._camStream);
                status.textContent = 'หมดเวลา — ไม่พบใบหน้า กรุณาลองใหม่อีกครั้ง';
                guide.classList.remove('ok');
                guide.classList.add('fail');
            }
        }, 40000);

        // รอให้กล้องเริ่มก่อน 1.5 วินาที
        await this._sleep(1500);
        status.textContent = 'จัดใบหน้าให้อยู่ในกรอบวงรี';

        // FaceMesh ตรวจตำแหน่งหน้า
        const faceMesh = new FaceMesh({ locateFile: f =>
            `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}` });
        faceMesh.setOptions({
            maxNumFaces: 1, refineLandmarks: false,
            minDetectionConfidence: 0.7, minTrackingConfidence: 0.7,
        });

        let faceReadyFrames = 0;
        let countingDown    = false;
        let verified        = false;
        let passiveResult   = null;
        let passiveSent     = false;

        faceMesh.onResults(async results => {
            if (verified || countingDown) return;

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
                guide.classList.remove('ok');
                faceReadyFrames = 0;
                status.textContent = 'แสงน้อยเกินไป — กรุณาอยู่ในพื้นที่ที่มีแสงสว่างเพียงพอ';
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                return;
            }

            const hasFace = results.multiFaceLandmarks?.length > 0;
            if (!hasFace) {
                guide.classList.remove('ok', 'fail');
                faceReadyFrames = 0;
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

            // ตาต้องเปิดปกติ (EAR ≥ 75% baseline)
            const earL   = this._calcEAR(lm, [33,160,158,133,153,144]);
            const earR   = this._calcEAR(lm, [362,385,387,263,373,380]);
            const earNow = (earL + earR) / 2;
            const earMin = (this.baselineEAR || 0.25) * 0.75;
            const eyesOk = earNow >= earMin;

            // ปากต้องหุบ
            const mouthOpen = dist2D(lm[13], lm[14]);
            const mouthOk   = faceH > 0 && (mouthOpen / faceH) < 0.10;

            const faceOk = inCenter && closeEnough && pitchOk && rollOk && yawOk && eyesOk && mouthOk;

            // วาด overlay ตา+ปาก — สีเขียวถ้าพร้อม, ขาวถ้ายังไม่พร้อม
            this._drawFaceFeatures(ctx, lm, canvas.width, canvas.height,
                faceOk ? 'rgba(74,222,128,0.95)' : 'rgba(255,255,255,0.6)');

            // Anti-spoof bounding box overlay (อัปเดตทุก frame ด้วยผลล่าสุด)
            this._drawAntiSpoofOverlay(ctx, lm, canvas.width, canvas.height, this._lastSpoofResult);

            if (faceOk) {
                guide.classList.remove('fail');
                guide.classList.add('ok');
                faceReadyFrames++;
                status.textContent = '✓ พบใบหน้า — กรุณานิ่งสักครู่...';

                // ส่ง passive anti-spoof พร้อมกันตั้งแต่เฟรมที่ 5
                if (faceReadyFrames === 5 && !passiveSent) {
                    passiveSent = true;
                    this._captureFrame(video).then(b64 => {
                        fetch('/api/antispoof-passive', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'X-CSRF-Token': document.querySelector('meta[name="csrf-token"]')?.content || '',
                            },
                            body: JSON.stringify({ face_image: b64 }),
                        })
                        .then(r => r.json())
                        .then(r => { passiveResult = r; this._lastSpoofResult = r; })
                        // Fail-closed: network error → treat as spoof, not real
                        .catch(() => {
                            passiveResult = { real: false, score: 0.0, _networkError: true };
                            this._lastSpoofResult = passiveResult;
                        });
                    });
                }

                if (faceReadyFrames >= 25) {
                    countingDown = true;
                    clearTimeout(_streamTimeoutId);
                    faceMesh.close();
                    // Final 1: ถ่ายรูปทันที ข้าม liveness
                    // TODO Phase 3: เปลี่ยนกลับเป็น _countdownThenLiveness(...)
                    status.textContent = '✓ พบใบหน้า — กำลังเช็คชื่อ...';
                    const snap = document.createElement('canvas');
                    snap.width  = video.videoWidth  || 640;
                    snap.height = video.videoHeight || 480;
                    snap.getContext('2d').drawImage(video, 0, 0);
                    const capturedFrame = snap.toDataURL('image/jpeg', 0.85);
                    this._stopStream(this._camStream);
                    await this._submitCheckin(capturedFrame, '');
                }
            } else {
                guide.classList.remove('ok');
                guide.classList.add('fail');
                faceReadyFrames = 0;
                if (!closeEnough)  status.textContent = 'เข้าใกล้กล้องอีกหน่อย';
                else if (!eyesOk)  status.textContent = 'กรุณาเปิดตาให้ปกติ';
                else if (!mouthOk) status.textContent = 'กรุณาหุบปาก';
                else if (!pitchOk) status.textContent = 'กรุณาอย่าก้มหรือเงยหน้า';
                else if (!rollOk)  status.textContent = 'กรุณาอย่าเอียงศีรษะ';
                else if (!yawOk)   status.textContent = 'กรุณามองตรงเข้ากล้อง';
                else               status.textContent = 'ขยับหน้าให้อยู่กลางกรอบ';
            }
        });

        const cam = new Camera(video, {
            onFrame: async () => { await faceMesh.send({ image: video }); },
            width: 640, height: 480,
        });
        this._faceMeshCam = cam;  // H3: store for cleanup on error paths
        cam.start();
    }

    async _countdownThenLiveness(video, canvas, guide, status, countdownEl, passiveResult) {
        // Countdown 3→2→1
        for (let i = 3; i >= 1; i--) {
            countdownEl.textContent = i;
            countdownEl.style.display = 'block';
            status.textContent = 'เตรียมพร้อม...';
            await this._sleep(700);
        }
        countdownEl.style.display = 'none';

        // รอ passive result ถ้ายังไม่กลับมา (max 1.5s)
        if (!passiveResult) {
            status.textContent = 'กำลังตรวจสอบ...';
            for (let i = 0; i < 15 && !passiveResult; i++) await this._sleep(100);
        }

        const spoofScore = passiveResult?.score ?? 0.0;
        // Fail-closed: if passive result not available within timeout, treat as spoof
        const isReal     = passiveResult?.real ?? false;

        // ถ้า passive บอกว่า spoof → ปฏิเสธทันที
        if (!isReal && spoofScore < 0.50) {
            this._stopStream(this._camStream);
            this._showDone('spoof', 'ตรวจพบการโกง!', false);
            return;
        }

        const capturedFrame = await this._captureFrame(video);
        let livenessAction  = 'passive';

        if (spoofScore >= 0.98) {
            // Passive ผ่าน — ข้าม manual liveness
            status.textContent = '✓ ตรวจสอบอัตโนมัติผ่าน';
            await this._sleep(400);
        } else {
            // Borderline — ขอ 1 manual challenge (blink หรือ turn_left เท่านั้น)
            const action  = Math.random() < 0.5 ? 'blink' : 'turn_left';
            const labels  = { blink: 'กะพริบตา 1 ครั้ง', turn_left: 'หันหน้าไปทางซ้าย' };
            status.textContent = `กรุณา${labels[action]}`;

            const detector = new LivenessDetector(video, canvas, this.baselineEAR);
            const result   = await detector.run(action, text => { status.textContent = text; });

            if (!result.pass) {
                this._stopStream(this._camStream);
                this._showDone('error', result.error || 'Liveness ไม่ผ่าน — กรุณาลองใหม่', true);
                return;
            }
            livenessAction = action;
        }

        this._stopStream(this._camStream);
        status.textContent = '✓ ยืนยันตัวตนสำเร็จ — กำลังเช็คชื่อ...';
        await this._submitCheckin(capturedFrame, livenessAction);
    }

    async _captureFrame(videoEl) {
        const c = document.createElement('canvas');
        c.width  = videoEl.videoWidth;
        c.height = videoEl.videoHeight;
        c.getContext('2d').drawImage(videoEl, 0, 0);
        return c.toDataURL('image/jpeg', 0.85);
    }

    // ─── Submit ──────────────────────────────────────────

    async _submitCheckin(faceImage, livenessAction) {
        this._goToStep(2);
        document.getElementById('doneLoadingView').style.display  = 'block';
        document.getElementById('doneResultView').style.display   = 'none';

        try {
            const res = await fetch(this.apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type':  'application/json',
                    'X-CSRF-Token':  document.querySelector('meta[name="csrf-token"]')?.content || '',
                    'Authorization': `DeviceToken ${localStorage.getItem('sc_device_token') || ''}`,
                },
                body: JSON.stringify({
                    session_id:      this.sessionId,
                    ble_rssi:        this._bleRSSI,
                    ble_skip:        this._bleSkip || false,
                    liveness_action: livenessAction,
                    liveness_pass:   true,
                    face_image:      faceImage,
                }),
            });

            const json = await res.json();
            document.getElementById('doneLoadingView').style.display = 'none';
            document.getElementById('doneResultView').style.display  = 'block';

            if (json.ok) {
                this._showDone('success', json.message || 'เช็คชื่อสำเร็จ!', false);
            } else if (json.already_checked) {
                this._showDone('info', json.error, false);
            } else if (json.spoof) {
                this._showDone('spoof', json.error, true);
            } else {
                this._showDone('error', json.error || 'เช็คชื่อไม่สำเร็จ', json.retry_face === true);
            }
        } catch (e) {
            document.getElementById('doneLoadingView').style.display = 'none';
            document.getElementById('doneResultView').style.display  = 'block';
            this._showDone('error', 'ไม่สามารถเชื่อมต่อ server ได้', true);
        }
    }

    // ─── Result ──────────────────────────────────────────

    _showDone(state, message, canRetry) {
        const icons  = { success: '✅', info: 'ℹ️', error: '❌', spoof: '⚠️' };
        const titles = { success: 'เช็คชื่อสำเร็จ!', info: 'เช็คชื่อแล้ว', error: 'เช็คชื่อไม่สำเร็จ', spoof: 'ตรวจพบการโกง!' };
        const iconEl  = document.getElementById('doneIcon');
        const titleEl = document.getElementById('doneTitle');
        iconEl.textContent  = icons[state]  || '❌';
        titleEl.textContent = titles[state] || 'เกิดข้อผิดพลาด';
        if (state === 'spoof') {
            iconEl.style.color  = '#f59e0b';
            titleEl.style.color = '#b45309';
        } else {
            iconEl.style.color  = '';
            titleEl.style.color = '';
        }
        document.getElementById('doneMsg').textContent   = message;
        document.getElementById('doneRetry').style.display = canRetry ? 'block' : 'none';
    }

    // ─── Helpers ─────────────────────────────────────────

    _drawFaceFeatures(ctx, lm, w, h, color) {
        const EYE_L = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246];
        const EYE_R = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398];
        const MOUTH = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146];

        ctx.strokeStyle = color;
        ctx.lineWidth   = 1.8;
        ctx.setLineDash([4, 3]);

        [EYE_L, EYE_R, MOUTH].forEach(indices => {
            ctx.beginPath();
            indices.forEach((idx, i) => {
                const x = lm[idx].x * w, y = lm[idx].y * h;
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            });
            ctx.closePath();
            ctx.stroke();
        });
        ctx.setLineDash([]);
    }

    /**
     * Draw a coloured bounding box + REAL/SPOOF label on the canvas.
     * Called every onResults frame with the cached _lastSpoofResult.
     * @param {CanvasRenderingContext2D} ctx
     * @param {Array} lm  – MediaPipe 468-landmark array
     * @param {number} w  – canvas pixel width
     * @param {number} h  – canvas pixel height
     * @param {object|null} result – { real, score } or { is_real, confidence } or null
     */
    _drawAntiSpoofOverlay(ctx, lm, w, h, result) {
        if (!lm || lm.length < 468) return;

        // Bounding box from outer face landmarks (left-ear, right-ear, forehead, chin)
        // Add 10 px padding so the box doesn't clip the face mesh lines
        const x1 = lm[234].x * w - 10;
        const y1 = lm[10].y  * h - 10;
        const bw = (lm[454].x - lm[234].x) * w + 20;
        const bh = (lm[152].y - lm[10].y)  * h + 20;

        // Normalise field names (passive API uses .real/.score;
        // _callSpoofCheckSafe uses .is_real/.confidence)
        const isReal = result ? (result.real ?? result.is_real ?? false) : null;
        const score  = result ? (result.score ?? result.confidence ?? 0) : 0;

        // Colour scheme: green = real, red = spoof, grey = waiting
        const color = isReal === null ? 'rgba(148,163,184,0.7)'
                    : isReal          ? '#4ade80'
                    :                   '#f87171';
        const label = isReal === null ? 'DETECTING...'
                    : isReal          ? `REAL ${score.toFixed(3)}`
                    :                   `SPOOF ${score.toFixed(3)}`;

        // Box
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth   = 2.5;
        ctx.strokeRect(x1, y1, bw, bh);

        // Label background + text
        ctx.font = 'bold 13px monospace';
        const textW = ctx.measureText(label).width + 10;
        const bgColor = isReal === null ? 'rgba(100,116,139,0.85)'
                      : isReal          ? 'rgba(74,222,128,0.85)'
                      :                   'rgba(248,113,113,0.85)';
        ctx.fillStyle = bgColor;
        ctx.fillRect(x1, y1 - 22, textW, 20);
        ctx.fillStyle = '#000';
        ctx.fillText(label, x1 + 5, y1 - 7);
        ctx.restore();
    }

    _calcEAR(lm, idx) {
        const d = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
        const [p1,p2,p3,p4,p5,p6] = idx.map(i => lm[i]);
        return (d(p2,p6) + d(p3,p5)) / (2.0 * d(p1,p4));
    }

    _stopStream(stream) {
        if (stream) stream.getTracks().forEach(t => t.stop());
        // H3: also stop MediaPipe Camera instance if stored
        if (this._faceMeshCam) {
            try { this._faceMeshCam.stop(); } catch (_) {}
            this._faceMeshCam = null;
        }
    }

    _sleep(ms) {
        return new Promise(r => setTimeout(r, ms));
    }
}
