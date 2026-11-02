/**
 * push_notifications.js — class-open notifications inside the SmartCheck Android app
 *
 * Loaded by base.html only for students in an app build that has Firebase
 * (User-Agent SmartCheckPush) while the server can send (app/routes/push.py).
 *
 *  1. Creates the "checkin" channel (heads-up), asks for the notification
 *     permission once, and registers the phone's token with the server
 *     (POST /api/push/register) when it changes, or weekly.
 *  2. Tapping a notification opens its check-in page.
 *  3. SmartCheckPush.clear(sessionId) removes that class's notification from the
 *     bar after a check-in (checkin_flow.js).
 *
 * The site is remote, so plugin calls go to the native bridge directly
 * (Capacitor.Plugins.PushNotifications), as in broadcast_room_scanner.js.
 */
const SmartCheckPush = (() => {
    const CHANNEL = 'checkin';
    const IMPORTANCE_HIGH = 4;            // heads-up in the bar
    const STORE_KEY = 'sc_push_token';
    const REFRESH_MS = 7 * 24 * 3600 * 1000;

    function plugin() {
        const cap = globalThis.Capacitor;
        if (!cap || typeof cap.isNativePlatform !== 'function' || !cap.isNativePlatform()) return null;
        return (cap.Plugins && cap.Plugins.PushNotifications) || null;
    }

    function stored() {
        try { return JSON.parse(localStorage.getItem(STORE_KEY) || 'null'); } catch { return null; }
    }

    async function sendToken(token) {
        const last = stored();
        if (last && last.token === token && Date.now() - last.at < REFRESH_MS) return;
        const res = await fetch('/api/push/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': document.querySelector('meta[name="csrf-token"]')?.content || '',
            },
            body: JSON.stringify({token}),
        });
        if (res.ok) {
            try { localStorage.setItem(STORE_KEY, JSON.stringify({token, at: Date.now()})); } catch { /* private mode */ }
        }
    }

    /** Only same-site check-in pages: the URL comes from the notification's data. */
    function checkinPath(data) {
        const url = data && typeof data.url === 'string' ? data.url : '';
        return /^\/student\/checkin\?session_id=[0-9a-fA-F-]{36}$/.test(url) ? url : null;
    }

    async function start() {
        const push = plugin();
        if (!push) return;
        await push.addListener('pushNotificationActionPerformed', event => {
            const path = checkinPath(event && event.notification && event.notification.data);
            if (path && location.pathname + location.search !== path) location.href = path;
        });
        await push.addListener('registration', t => { sendToken(t.value).catch(() => {}); });
        await push.addListener('registrationError', e => console.info('step=push_register result=error ' + (e && e.error)));
        try {
            await push.createChannel({id: CHANNEL, name: 'เช็คชื่อ', description: 'แจ้งเตือนเมื่อเปิดคาบเรียน',
                importance: IMPORTANCE_HIGH, visibility: 1, vibration: true});
        } catch { /* Android < 8 has no channels */ }
        let state = (await push.checkPermissions()).receive;
        if (state === 'prompt' || state === 'prompt-with-rationale') state = (await push.requestPermissions()).receive;
        if (state !== 'granted') return;
        await push.register();
    }

    async function clear(sessionId) {
        const push = plugin();
        if (!push || !sessionId) return;
        try {
            const {notifications} = await push.getDeliveredNotifications();
            const mine = (notifications || []).filter(n => n.tag === sessionId);
            if (mine.length) await push.removeDeliveredNotifications({notifications: mine});
        } catch { /* nothing to clear */ }
    }

    return {start, clear, checkinPath};
})();

if (typeof module !== 'undefined') {
    module.exports = SmartCheckPush;
} else {
    SmartCheckPush.start().catch(e => console.info('step=push_start result=error ' + (e && e.message)));
}
