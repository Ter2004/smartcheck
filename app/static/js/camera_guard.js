/**
 * camera_guard.js — Virtual Camera Detection Utility
 *
 * Blocks known virtual/software cameras (OBS, ManyCam, SplitCam, etc.)
 * that can be used to inject deepfake or illustration images into the
 * enrollment and check-in pipeline.
 *
 * MUST be called AFTER navigator.mediaDevices.getUserMedia() so that
 * the browser reveals device labels (browser privacy policy).
 */

const _VC_BLOCKED_KEYWORDS = [
    'obs', 'virtual', 'splitcam', 'manycam', 'snap camera',
    'epoccam', 'droidcam', 'iriun', 'camo', 'mmhmm', 'xsplit',
];

/**
 * Check if the active video track comes from a known virtual/software camera.
 *
 * @param {MediaStream} stream – the stream returned by getUserMedia
 * @returns {Promise<{ blocked: boolean, label: string }>}
 */
async function detectVirtualCamera(stream) {
    try {
        const videoTracks = stream.getVideoTracks();
        if (!videoTracks.length) return { blocked: false, label: '' };

        // Primary check: active track label (most reliable)
        const activeLabel = videoTracks[0].label.toLowerCase();
        for (const kw of _VC_BLOCKED_KEYWORDS) {
            if (activeLabel.includes(kw)) {
                return { blocked: true, label: videoTracks[0].label };
            }
        }

        // Fallback: enumerate all video devices and cross-check
        const devices = await navigator.mediaDevices.enumerateDevices();
        for (const device of devices) {
            if (device.kind !== 'videoinput') continue;
            const devLabel = device.label.toLowerCase();
            if (devLabel !== activeLabel) continue;
            for (const kw of _VC_BLOCKED_KEYWORDS) {
                if (devLabel.includes(kw)) {
                    return { blocked: true, label: device.label };
                }
            }
            break;
        }

        return { blocked: false, label: videoTracks[0].label };
    } catch (_) {
        // Fail-open: if enumerateDevices is unavailable, don't block real cameras
        return { blocked: false, label: '' };
    }
}
