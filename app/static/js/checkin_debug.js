// Temporary, opt-in diagnostics. No network, persistence, images or credentials.
class CheckinDebug {
    constructor(baseline) {
        this.baseline = baseline;
        this.started = performance.now();
        this.wallStarted = Date.now();
        this.attempts = 0;
        this.retries = 0;
        this.events = [];
        this.frame = {};
        this.timerState = 'not armed';
        this.panel = document.createElement('pre');
        this.panel.id = 'checkinDebug';
        this.panel.style.cssText = 'position:absolute;left:4px;right:4px;bottom:4px;z-index:30;margin:0;padding:6px;background:rgba(0,0,0,.82);color:#fff;font:11px/1.3 monospace;white-space:pre-wrap;overflow-wrap:anywhere;pointer-events:none;text-align:left;';
        this.history = document.createElement('pre');
        this.history.id = 'checkinDebugHistory';
        this.history.style.cssText = 'padding:6px;margin:4px 0;background:#111;color:#fff;font:11px/1.3 monospace;white-space:pre-wrap;overflow-wrap:anywhere;';
        this.mount(1);
        this.event('diagnostics loaded');
        this.lastTick = performance.now();
        this.maxTickGap = 0;
        this.interval = setInterval(() => {
            const now = performance.now();
            this.maxTickGap = Math.max(this.maxTickGap, now - this.lastTick);
            this.lastTick = now;
            this.render();
        }, 250);
        // Record error kind/location only: arbitrary error messages may contain data.
        window.addEventListener('error', e => this.event(`JS error line ${e.lineno || '?'} (${e.error?.name || 'error'})`));
        window.addEventListener('unhandledrejection', () => this.event('unhandled promise rejection'));
        document.addEventListener('visibilitychange', () => this.event(`visibility ${document.visibilityState}`));
        window.addEventListener('pagehide', () => this.event('pagehide'));
        window.addEventListener('pageshow', () => this.event('pageshow'));
    }

    mount(step) {
        const video = document.getElementById('videoVerify');
        const box = video?.parentElement;
        if (!box) return;
        document.getElementById('stepVerify').after(this.history);
        // Keep the last readings photographable after timeout/expiry hides video.
        if (step === 2) {
            this.panel.style.position = 'absolute';
            box.appendChild(this.panel);
        } else {
            this.panel.style.position = 'relative';
            document.getElementById('stepVerify').after(this.panel);
        }
    }

    event(label) {
        this.events.push(`${((performance.now() - this.started) / 1000).toFixed(1)}s A${this.attempts} ${label}`);
        this.events = this.events.slice(-4);
        this.render();
    }

    begin() {
        this.attempts++;
        this.attemptStart = performance.now();
        this.timerState = 'not armed';
        this.timerDue = null;
        this.frame = {};
        this.frameAt = null;
        this.event('capture start');
    }

    arm() {
        this.timerDue = performance.now() + 40000;
        this.timerState = 'armed';
        this.event('40s timer armed');
    }

    fired() {
        this.timerState = 'fired';
        this.event(`40s callback fired; late ${Math.max(0, performance.now() - this.timerDue).toFixed(0)}ms`);
    }

    stopped() {
        if (this.timerState === 'armed') {
            this.timerState = 'cleared';
            this.event('40s timer cleared');
        }
    }

    sample(values) {
        this.frame = values;
        this.frameAt = performance.now();
        this.render();
    }

    render() {
        const now = performance.now();
        const f = this.frame;
        const video = document.getElementById('videoVerify');
        const num = v => v == null ? 'n/a' : Number.isFinite(v) ? v.toFixed(6) : String(v);
        const age = t => t == null ? 'n/a' : ((now - t) / 1000).toFixed(1) + 's';
        const remaining = this.receiptDue == null ? 'n/a' : ((this.receiptDue - now) / 1000).toFixed(1) + 's';
        const timer = this.timerState === 'armed' ? `${((this.timerDue - now) / 1000).toFixed(1)}s left` : this.timerState;
        this.panel.textContent = [
            'CHECKIN DEBUG v2 — pixel-v1 EAR',
            `legacy baseline (unused)=${String(this.baseline)}`,
            `calibration=${f.calibration || 'awaiting face'} baseline=${num(f.baseline)}`,
            `earL=${num(f.earL)} earR=${num(f.earR)}`,
            `earNow=${num(f.earNow)} earMin=${num(f.earMin)}`,
            `video=${video?.videoWidth ?? '?'}×${video?.videoHeight ?? '?'} ready=${f.ready ?? 0}/25`,
            `faceW=${num(f.faceW)} (>0.28) faceH=${num(f.faceH)} (>0.36)`,
            `failed=${f.failed?.join(', ') || 'none / awaiting frame'}`,
            `faces returned=${f.faces ?? '?'} (configured max=1) frame age=${age(this.frameAt)}`,
            `attempt=${this.attempts} retries=${this.retries} age=${age(this.attemptStart)}`,
            `40s timer=${timer}; receipt=${remaining}`,
            `page=${age(this.started)} wall=${((Date.now() - this.wallStarted) / 1000).toFixed(1)}s ${document.visibilityState}`,
            `tick gap max=${((this.maxTickGap || 0) / 1000).toFixed(1)}s`,
        ].join('\n');
        this.history.textContent = this.events.join('\n');
    }
}
