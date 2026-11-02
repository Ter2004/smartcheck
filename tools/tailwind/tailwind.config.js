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
            indigo: {50: '#eef5ff', 100: '#deecff', 200: '#bedaff', 300: '#91bfff', 400: '#5299f0', 500: '#187de0', 600: '#0071e3', 700: '#0062c4', 800: '#004f9e', 900: '#003c78'},
            brand: {50: '#eef2ff', 100: '#e0e7ff', 200: '#c7d2fe', 500: '#6366f1', 600: '#4f46e5', 700: '#4338ca', 800: '#3730a3'},
        },
    }},
};
