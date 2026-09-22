/**
 * checkin_flow.js — SmartCheck Check-in Flow
 *
 * Flow: BLE read / TOTP → server preflight → camera → receipt-backed submit.
 */

class CheckinFlow {
    constructor(opts) {
        this.sessionId     = opts.sessionId;
        this.beaconUUID    = opts.beaconUUID;
        this.rssiThreshold = opts.rssiThreshold;
        this.baselineEAR   = opts.baselineEAR;
        this.apiUrl        = opts.apiUrl || '/api/checkin';
        this.proximityMethod = opts.proximityMethod || 'totp';

        this._bleRSSI    = null;
        this._bleSkip    = false;
        this._camStream  = null;
        this._earSamples = [];
        this._proximity = null;
        this._debug = opts.debug === true ? new CheckinDebug(this.baselineEAR) : null;
    }

    async start() {
        if (!document.getElementById('stepRoomCode')) return;
        // Proximity always gates the camera; legacy RSSI is a separate option.
        this._bleRSSI = null;
        this._bleSkip = true;
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
        if (title) title.textContent = ['', this.proximityMethod === 'ble' ? 'หาสัญญาณห้องเรียน' : 'ยืนยันรหัสห้องเรียน', 'ยืนยันตัวตนด้วยใบหน้า', 'ผลการเช็คชื่อ'][n];
        if (help) help.textContent = ['', 'ตรวจสอบว่าคุณอยู่ในห้องเรียนของคาบที่เลือก', 'จัดใบหน้าในกรอบ ระบบจะถ่ายภาพและเช็คชื่อให้อัตโนมัติ', 'ระบบตรวจสอบใบหน้าและบันทึกการเข้าเรียน'][n];
        // 1=proximity, 2=camera, 3=result.
        const steps = [this.proximityMethod === 'ble' ? 'stepBleRoom' : 'stepRoomCode', 'stepVerify', 'stepDone'];
        steps.forEach((id, i) => {
            const el = document.getElementById(id);
            if (el) el.style.display = (i + 1 === n) ? 'block' : 'none';
        });
        for (let i = 1; i <= 3; i++) {
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

    // Step 2: Detect face and submit automatically; no interactive challenge.

    async _startVerify() {
        if (!this._proximity || performance.now() >= this._proximity.deadline) {
            this._requestRoomCode();
            return;
        }
        this._stopStream(this._camStream);
        const generation = this._captureGeneration;
        const active = () => generation === this._captureGeneration && !!this._proximity;
        this.baselineEAR = null;
        this._earSamples = [];
        let faceReadyFrames = 0, countingDown = false, verified = false;
        this._debug?.begin();
        this._goToStep(2);
        document.getElementById("stepRoomCode").style.display = "none";
        this._capturedFrames = [];

        const video    = document.getElementById('videoVerify');
        const canvas   = document.getElementById('canvasVerify');
        const guide    = document.getElementById('faceGuideVerify');
        const status   = document.getElementById('verifyStatus');
        const countdown = document.getElementById('countdownBadge');

        await StepGuide.show('📷', 'เตรียมยืนยันใบหน้า',
            'จัดหน้าในกรอบแล้วอยู่นิ่ง ระบบจะถ่ายและเช็คชื่อให้อัตโนมัติ เวลายืนยันห้องยังคงนับถอยหลัง');
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
                status.textContent = 'ตรวจพบกล้องเสมือน — กรุณาใช้กล้องจริงเท่านั้น';
                alert(`ไม่อนุญาตให้ใช้กล้องเสมือน (${vcCheck.label}) — กรุณาใช้กล้องจริงเท่านั้น`);
                throw new Error('Virtual Camera Detected');
            }
            video.srcObject = this._camStream;
        } catch (e) {
            if (!active()) return;
            this._stopStream(this._camStream);
            console.info('step=camera_open result=error details={}');
            status.textContent = 'ไม่สามารถเปิดกล้องได้: ' + e.message;
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

        // รอให้กล้องเริ่มก่อน 1.5 วินาที
        await this._sleep(1500);
        if (!active()) return;
        status.textContent = 'จัดใบหน้าให้อยู่ในกรอบวงรี';

        // FaceMesh ตรวจตำแหน่งหน้า
        const faceMesh = new FaceMesh({ locateFile: f =>
            `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}` });
        this._activeMesh = faceMesh;
        faceMesh.setOptions({
            maxNumFaces: 1, refineLandmarks: false,
            minDetectionConfidence: 0.7, minTrackingConfidence: 0.7,
        });

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

            // วาด overlay ตา+ปาก — สีเขียวถ้าพร้อม, ขาวถ้ายังไม่พร้อม
            this._drawFaceFeatures(ctx, lm, canvas.width, canvas.height,
                faceOk ? 'rgba(74,222,128,0.95)' : 'rgba(255,255,255,0.6)');


            if (faceOk) {
                guide.classList.remove('fail');
                guide.classList.add('ok');
                faceReadyFrames++;
                status.textContent = '✓ พบใบหน้า — กรุณานิ่งสักครู่...';

                if (faceReadyFrames >= 25) {
                    this._debug?.event('25 ready frames; capture complete');
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
            this._meshWork = Promise.resolve(faceMesh.initialize());
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
                warn.textContent = 'เบราว์เซอร์นี้ไม่รองรับ Web Bluetooth — กรุณาใช้ Chrome หรือ Edge บนคอมพิวเตอร์ Windows เพื่อเช็คชื่อ';
                warn.style.display = 'block';
                btn.disabled = true;
                btn.textContent = 'ไม่รองรับ Bluetooth';
            } else {
                warn.style.display = 'none';
                btn.disabled = false;
                btn.textContent = 'หาอุปกรณ์ในห้อง';
            }
        } else {
            document.getElementById('stepRoomCode').style.display = 'block';
            document.getElementById('roomCode').value = '';
            document.getElementById('roomCode').focus();
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
            result = await new BLERoomScanner().findRoom();
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
            await this._verifyProximity(result.room);
        } finally {
            this._submitting = false;
        }
    }

    async _verifyProximity(room) {
        const status = document.getElementById('proximityStatus');
        status.textContent = 'กำลังตรวจสอบตำแหน่งห้องเรียน...';
        const started = performance.now();
        try {
            const response = await fetch('/api/checkin/proximity', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-Token': document.querySelector('meta[name="csrf-token"]')?.content || '',
                },
                body: JSON.stringify({session_id: this.sessionId, room_code: room}),
            });
            const result = await response.json();
            if (!response.ok || !result.ok) {
                console.info('step=proximity_preflight result=reject details=' + JSON.stringify({status: response.status}));
                status.textContent = result.error || 'ตรวจสอบตำแหน่งไม่สำเร็จ กรุณาลองใหม่';
                return;
            }
            this._proximity = {room, receipt: result.proximity_receipt,
                deadline: started + result.expires_in * 1000, captureStarted: performance.now()};
            if (this._debug) {
                this._debug.receiptDue = this._proximity.deadline;
                this._debug.event('new receipt');
            }
            status.textContent = 'ยืนยันห้องแล้ว 90 วินาที — ถ่ายภาพได้รอบละ 40 วินาที';
            this._receiptTimer = setInterval(() => {
                if (!this._proximity) return;
                const seconds = Math.max(0, Math.ceil((this._proximity.deadline - performance.now()) / 1000));
                document.getElementById('receiptCountdown').textContent = `เวลายืนยันตำแหน่งคงเหลือ ${seconds} วินาที`;
                if (!seconds) this._expireProximity();
            }, 250);
            await this._startVerify();
        } catch (error) {
            console.info('step=proximity_preflight result=error details={}');
            status.textContent = 'ไม่สามารถเชื่อมต่อระบบตรวจสอบตำแหน่งได้ กรุณาลองใหม่';
        } finally {
            if (this.proximityMethod === 'ble' && navigator.bluetooth) {
                document.getElementById('bleRoomBtn').disabled = false;
            }
        }
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
                    ble_rssi:        this._bleRSSI,
                    ble_skip:        this._bleSkip || false,
                    liveness_action: livenessAction,
                    liveness_pass:   true,
                    face_image:      faceImage,
                    face_images:     this._capturedFrames || [faceImage],
                }),
            });

            const json = await this._readCheckinResponse(res);
            document.getElementById('doneLoadingView').style.display = 'none';
            document.getElementById('doneResultView').style.display  = 'block';

            this._retryRoomCode = json.retry_room_code === true;
            if (json.ok) {
                this._showDone('success', json.message || 'เช็คชื่อสำเร็จ!', false);
            } else if (json.already_checked) {
                this._showDone('info', json.error, false);
            } else if (json.spoof) {
                this._showDone('spoof', json.error, true);
            } else {
                this._showDone('error', json.error || json.message || 'เช็คชื่อไม่สำเร็จ', json.retry_face === true || this._retryRoomCode);
            }
        } catch (e) {
            document.getElementById('doneLoadingView').style.display = 'none';
            document.getElementById('doneResultView').style.display  = 'block';
            console.error('Check-in response failed', {name: e.name, status: e.httpStatus});
            this._showDone('error', e.userMessage || 'การเชื่อมต่อขาดหาย กรุณาตรวจประวัติเช็คชื่อก่อนลองใหม่ หากยังไม่ได้บันทึกให้โหลดหน้าใหม่', false);
        }
    }

    async _readCheckinResponse(res) {
        const fail = (message) => {
            const error = new Error('Invalid check-in response');
            error.userMessage = message;
            error.httpStatus = res.status;
            throw error;
        };
        if (res.status === 401 || (res.redirected && new URL(res.url, location.href).pathname === '/login')) {
            fail('เซสชันหมดอายุ กรุณาเข้าสู่ระบบใหม่แล้วตรวจประวัติเช็คชื่อ');
        }
        let json;
        try { json = await res.json(); } catch (_) {
            if (res.status === 429) fail('คำขอมากเกินไป กรุณารอสักครู่แล้วโหลดหน้าใหม่');
            if ([502, 503, 504, 520, 521, 522, 523, 524, 530].includes(res.status)) {
                fail(`เว็บเชื่อมต่อเซิร์ฟเวอร์ไม่สำเร็จ (HTTP ${res.status}) กรุณาตรวจประวัติเช็คชื่อก่อนลองใหม่`);
            }
            fail(`เซิร์ฟเวอร์ตอบกลับผิดรูปแบบ (HTTP ${res.status}) กรุณาโหลดหน้าใหม่และตรวจประวัติเช็คชื่อ`);
        }
        if (!json || typeof json !== 'object' || Array.isArray(json)) {
            fail(`เซิร์ฟเวอร์ตอบกลับผิดรูปแบบ (HTTP ${res.status}) กรุณาโหลดหน้าใหม่`);
        }
        if (!res.ok) json.ok = false;
        return json;
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
