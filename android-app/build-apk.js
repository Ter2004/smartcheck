// Build the SmartCheck Android app (debug APK) for one server URL.
//
//   node build-apk.js https://your-smartcheck-host
//
// Capacitor injects its bridge (and so the Bluetooth plugin) only into pages from
// server.url, so the URL is fixed into each build. When the URL changes (e.g. a new
// cloudflared quick tunnel), run this again. Output: dist/SmartCheck-debug.apk
const fs = require('node:fs');
const path = require('node:path');
const {execFileSync} = require('node:child_process');

const APP_ID = 'com.smartcheck.app';
const APP_NAME = 'SmartCheck';
const root = __dirname;

function fail(message) {
    console.error('ERROR: ' + message);
    process.exit(1);
}

function serverUrl() {
    const raw = process.argv[2] || process.env.SMARTCHECK_URL || '';
    if (!raw) fail('give the SmartCheck server URL, e.g. node build-apk.js https://xxxx.trycloudflare.com');
    let url;
    try {
        url = new URL(raw);
    } catch {
        fail('not a URL: ' + raw);
    }
    // Camera and the app bridge need https; a path would break same-origin page links.
    if (url.protocol !== 'https:') fail('the URL must start with https://');
    if (url.pathname !== '/' || url.search || url.hash) fail('use the site root only, e.g. https://host (no path)');
    return url.origin;
}

function run(command, args, options = {}) {
    console.log('> ' + command + ' ' + args.join(' '));
    execFileSync(command, args, {stdio: 'inherit', shell: process.platform === 'win32', ...options});
}

const url = serverUrl();
const config = {
    appId: APP_ID,
    appName: APP_NAME,
    webDir: 'www',
    server: {url, errorPath: 'error.html'},
    android: {allowMixedContent: false},
};
fs.writeFileSync(path.join(root, 'capacitor.config.json'), JSON.stringify(config, null, 2) + '\n');
fs.writeFileSync(path.join(root, 'www', 'server-url.js'), 'window.SMARTCHECK_URL = ' + JSON.stringify(url + '/') + ';\n');
console.log('server URL: ' + url);

const env = {...process.env};
if (!env.JAVA_HOME) {
    const studioJbr = 'C:\\Program Files\\Android\\Android Studio\\jbr';
    if (fs.existsSync(studioJbr)) env.JAVA_HOME = studioJbr;
}
if (!env.ANDROID_HOME && env.LOCALAPPDATA) env.ANDROID_HOME = path.join(env.LOCALAPPDATA, 'Android', 'Sdk');
if (!env.JAVA_HOME) fail('JAVA_HOME is not set and Android Studio\'s Java was not found');
if (!env.ANDROID_HOME || !fs.existsSync(env.ANDROID_HOME)) fail('Android SDK not found; set ANDROID_HOME');

const android = path.join(root, 'android');
if (!fs.existsSync(android)) run('npx', ['cap', 'add', 'android'], {cwd: root, env});
run('npx', ['cap', 'sync', 'android'], {cwd: root, env});
fs.writeFileSync(path.join(android, 'local.properties'),
    'sdk.dir=' + env.ANDROID_HOME.replace(/\\/g, '\\\\').replace(/:/g, '\\:') + '\n');
// Full path: cmd may not search the current folder (NoDefaultCurrentDirectoryInExePath).
const gradlew = path.join(android, process.platform === 'win32' ? 'gradlew.bat' : 'gradlew');
run(JSON.stringify(gradlew), ['assembleDebug'], {cwd: android, env});

const apk = path.join(android, 'app', 'build', 'outputs', 'apk', 'debug', 'app-debug.apk');
if (!fs.existsSync(apk)) fail('build finished but no APK at ' + apk);
fs.mkdirSync(path.join(root, 'dist'), {recursive: true});
const out = path.join(root, 'dist', 'SmartCheck-debug.apk');
fs.copyFileSync(apk, out);
console.log('\nAPK: ' + out + '\nserver URL inside: ' + url);
