// push_notifications.js: permission, token registration, tap → check-in page, clear after check-in.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../app/static/js/push_notifications.js'), 'utf8');

const SESSION = '0b1c2d3e-4f50-6172-8394-a5b6c7d8e9f0';

function load({permission = 'prompt', grant = 'granted', native = true, stored = null} = {}) {
    const calls = [], listeners = {}, posted = [], store = {};
    if (stored) store.sc_push_token = JSON.stringify(stored);
    const plugin = {
        async addListener(name, fn) { listeners[name] = fn; return {remove() {}}; },
        async createChannel(channel) { calls.push(['createChannel', channel]); },
        async checkPermissions() { calls.push(['check']); return {receive: permission}; },
        async requestPermissions() { calls.push(['request']); return {receive: grant}; },
        async register() { calls.push(['register']); },
        async getDeliveredNotifications() {
            return {notifications: [{id: 0, tag: SESSION}, {id: 0, tag: 'other-session'}]};
        },
        async removeDeliveredNotifications(arg) { calls.push(['remove', arg.notifications]); },
    };
    const location = {pathname: '/student/dashboard', search: '', href: ''};
    const context = {
        globalThis: null, console, location, Date, JSON,
        Capacitor: {isNativePlatform: () => native, Plugins: {PushNotifications: plugin}},
        document: {querySelector: () => ({content: 'csrf'})},
        localStorage: {getItem: k => store[k] ?? null, setItem: (k, v) => { store[k] = v; }},
        fetch: async (url, opts) => { posted.push([url, JSON.parse(opts.body), opts.headers]); return {ok: true}; },
        module: {exports: null},
    };
    context.globalThis = context;
    vm.runInNewContext(source, context);
    return {push: context.module.exports, calls, listeners, posted, location, store};
}

const tick = () => new Promise(resolve => setImmediate(resolve));

(async () => {
    {   // First start: heads-up channel, asks once, registers, sends the token.
        const t = load();
        await t.push.start();
        assert.deepEqual(t.calls.map(c => c[0]), ['createChannel', 'check', 'request', 'register']);
        assert.equal(t.calls[0][1].id, 'checkin');
        assert.equal(t.calls[0][1].importance, 4);
        t.listeners.registration({value: 'tok-1'});
        await tick();
        assert.deepEqual(t.posted.map(p => [p[0], p[1].token, p[2]['X-CSRF-Token']]),
            [['/api/push/register', 'tok-1', 'csrf']]);
        assert.equal(JSON.parse(t.store.sc_push_token).token, 'tok-1');
    }
    {   // Same token within a week is not sent again; a new token is.
        const t = load({permission: 'granted', stored: {token: 'tok-1', at: Date.now()}});
        await t.push.start();
        assert.ok(!t.calls.some(c => c[0] === 'request'));
        t.listeners.registration({value: 'tok-1'});
        await tick();
        assert.equal(t.posted.length, 0);
        t.listeners.registration({value: 'tok-2'});
        await tick();
        assert.equal(t.posted.length, 1);
    }
    {   // Refused permission: no registration.
        const t = load({grant: 'denied'});
        await t.push.start();
        assert.ok(!t.calls.some(c => c[0] === 'register'));
    }
    {   // Not in the app: nothing happens.
        const t = load({native: false});
        await t.push.start();
        assert.equal(t.calls.length, 0);
    }
    {   // Tap opens that class's check-in page; anything else is ignored.
        const t = load({permission: 'granted'});
        await t.push.start();
        const url = '/student/checkin?session_id=' + SESSION;
        t.listeners.pushNotificationActionPerformed({notification: {data: {url: 'https://evil.example/x'}}});
        assert.equal(t.location.href, '');
        t.listeners.pushNotificationActionPerformed({notification: {data: {url}}});
        assert.equal(t.location.href, url);
    }
    {   // After a check-in only that class's notification is removed.
        const t = load({permission: 'granted'});
        await t.push.clear(SESSION);
        const removed = t.calls.find(c => c[0] === 'remove');
        assert.deepEqual(removed[1].map(n => n.tag), [SESSION]);
    }
    console.log('push_notifications.js: all checks passed');
})().catch(e => { console.error(e); process.exit(1); });
