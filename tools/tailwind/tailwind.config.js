// Builds app/static/css/tailwind.css (npm run build here). Same settings the pages
// gave the Tailwind Play CDN runtime before 2026-11-02; run it again after adding or
// changing Tailwind classes in app/templates or app/static/js.
module.exports = {
    content: ['../../app/templates/**/*.html', '../../app/static/js/**/*.js'],
    corePlugins: {preflight: false},
    theme: {extend: {
        fontFamily: {
            sans: ["'Noto Sans Thai'", 'system-ui', 'sans-serif'],
            mono: ["'JetBrains Mono'", 'ui-monospace', 'monospace'],
        },
        colors: {
            // The app's action scale. Since Phase 7c (boarding-pass world) it is tarmac navy,
            // so every existing bg-indigo-600 button and text-indigo-600 link follows.
            indigo: {50: '#EEF2F8', 100: '#DDE5F1', 200: '#BCC9DE', 300: '#8FA2C3', 400: '#5B729B', 500: '#2E4A7A', 600: '#12284D', 700: '#0B1D3A', 800: '#071328', 900: '#040B18'},
            brand: {50: '#eef2ff', 100: '#e0e7ff', 200: '#c7d2fe', 500: '#6366f1', 600: '#4f46e5', 700: '#4338ca', 800: '#3730a3'},
        },
    }},
};
