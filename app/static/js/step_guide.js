/* Shared, tap-anywhere instructions. Camera work starts after the promise resolves. */
window.StepGuide = (() => {
    let pending = null;
    const style = document.createElement('style');
    style.textContent = `
        .step-guide { position:fixed; inset:0; width:100%; height:100%; max-width:none;
            max-height:none; margin:0; padding:24px; border:0; background:transparent;
            color:#1e293b; cursor:pointer; box-sizing:border-box; }
        .step-guide[open] { display:flex; align-items:center; justify-content:center; }
        .step-guide::backdrop { background:rgba(15,23,42,.55); backdrop-filter:blur(4px); }
        .step-guide-card { width:100%; max-width:390px; max-height:85dvh; overflow:auto;
            padding:32px 26px; border-radius:24px; background:white; text-align:center;
            box-shadow:0 24px 80px rgba(15,23,42,.2); animation:step-guide-pop .45s ease both; }
        .step-guide-icon { font-size:40px; margin-bottom:12px; }
        .step-guide h2 { margin:0 0 12px; font-size:21px; line-height:1.5; }
        .step-guide p { margin:0; font-size:15px; line-height:1.8; white-space:pre-line; color:#64748b; }
        .step-guide button { margin-top:24px; padding:12px 16px; width:100%; border:0;
            border-radius:12px; background:#eef5ff; color:#0071e3; font:inherit; cursor:pointer; }
        .step-guide button:focus-visible { outline:3px solid #0071e3; outline-offset:3px; }
        @keyframes step-guide-pop { 0% { opacity:0; transform:translateY(36px) scale(.88); }
            65% { opacity:1; transform:translateY(-6px) scale(1.025); }
            100% { opacity:1; transform:translateY(0) scale(1); } }
        @media (prefers-reduced-motion:reduce) { .step-guide-card { animation:none; } }
    `;
    document.head.appendChild(style);
    function dismiss() { if (pending) pending(); }
    function show(icon, title, description) {
        dismiss();
        return new Promise(resolve => {
            const previousFocus = document.activeElement;
            const overflow = document.body.style.overflow;
            const dialog = document.createElement('dialog');
            dialog.className = 'step-guide';
            dialog.setAttribute('aria-labelledby', 'step-guide-title');
            dialog.setAttribute('aria-describedby', 'step-guide-description');
            dialog.innerHTML = '<div class="step-guide-card"><div class="step-guide-icon" aria-hidden="true"></div><h2 id="step-guide-title"></h2><p id="step-guide-description"></p><button type="button">แตะที่ไหนก็ได้เพื่อเริ่มขั้นตอนนี้</button></div>';
            dialog.querySelector('.step-guide-icon').textContent = icon;
            dialog.querySelector('h2').textContent = title;
            dialog.querySelector('p').textContent = description;
            let closed = false;
            pending = () => {
                if (closed) return;
                closed = true;
                pending = null;
                dialog.close();
                dialog.remove();
                document.body.style.overflow = overflow;
                if (previousFocus?.isConnected) previousFocus.focus();
                resolve();
            };
            dialog.addEventListener('click', event => { event.stopPropagation(); dismiss(); });
            dialog.addEventListener('cancel', event => { event.preventDefault(); dismiss(); });
            document.body.appendChild(dialog);
            document.body.style.overflow = 'hidden';
            dialog.showModal();
        });
    }
    return {show, dismiss};
})();
