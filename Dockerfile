FROM python:3.11-slim

WORKDIR /app

# Force legacy Keras (belt-and-suspenders even though tensorflow 2.15 defaults to Keras 2)
ENV TF_USE_LEGACY_KERAS=1
ENV TF_CPP_MIN_LOG_LEVEL=2

COPY requirements.txt constraints.txt ./

# Change CACHE_BUST to force re-run of all layers below (pip install + model download)
ARG CACHE_BUST=20260419e
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    \
    # Step 1: Install torch FIRST from PyTorch CPU-only CDN.
    # --index-url (not --extra-index-url) forces this URL for torch,
    # bypassing PyPI entirely. Without this, pip picks the CUDA wheel.
    pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.2.2 && \
    \
    # Step 2: Install everything else from PyPI, constrained by constraints.txt.
    # constraints.txt pins numpy==1.26.4, tensorflow==2.15.0, tf-keras==2.15.1
    # even for transitive deps, preventing silent upgrades to incompatible versions.
    pip install --no-cache-dir \
        --constraint constraints.txt \
        -r requirements.txt && \
    \
    # Step 3: Replace opencv-python (pulled by deepface) with headless build
    (pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true) && \
    pip install --no-cache-dir --constraint constraints.txt "opencv-python-headless>=4.8.0,<4.12" && \
    \
    # Purge build deps and clean caches in SAME layer (critical for image size)
    apt-get purge -y --auto-remove gcc libpq-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /root/.cache/pip /tmp/* /var/tmp/*

# F-16 ROOT-CAUSE FIX (docs/review/10-moire-frr-investigation.md §16-17): this
# blanket `-name test` match used to delete tensorflow/_api/v2/__internal__/test/
# — a real, load-bearing part of TensorFlow's public API (tf.__internal__.test),
# not a bundled test suite — which is what caused the ImportError verified in
# steps 1-2 above. Excluding tensorflow/ from the prune. Cleanup now also runs
# BEFORE the import checks below (STEP 1's reorder), so a regression here would
# fail the build instead of shipping silently. Residual risk: other packages
# under site-packages could have the same false-positive-match problem and
# haven't been individually audited — only tensorflow's case is confirmed.
RUN find /usr/local/lib/python3.11 -depth -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11 -depth -type d \( -name tests -o -name test \) -not -path "*/tensorflow/*" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11 -name "*.pyc" -delete 2>/dev/null || true

# Step 4: Verify the critical imports work BEFORE proceeding.
# Build fails loudly here rather than producing a silently broken image.
# Moved into its own layer, now running AFTER cleanup above (previously ran
# before it, in the same RUN as the installs — which is why it never caught this).
RUN python -c "import numpy; print('numpy', numpy.__version__); assert numpy.__version__.startswith('1.'), 'FAIL: numpy must be 1.x'" && \
    python -c "import torch; print('torch', torch.__version__); import numpy; t = torch.zeros(1); t.numpy(); print('torch.numpy OK')" && \
    python -c "import tensorflow; print('tensorflow', tensorflow.__version__); assert tensorflow.__version__.startswith('2.15'), 'FAIL: tensorflow must be 2.15.x'" && \
    python -c "from deepface import DeepFace; print('DeepFace import OK')" && \
    python -c "import onnxruntime; print('onnxruntime', onnxruntime.__version__)"

# Pre-download model weights so they're baked into the image (no cold-start download).
# Separate layer so it doesn't re-run on code changes, only on dep changes.
RUN python - <<'EOF'
import os
os.environ.setdefault("HOME", "/root")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from deepface import DeepFace

try:
    DeepFace.build_model("Facenet512")
    print("Facenet512 OK")
except Exception as e:
    print(f"Facenet512 warn: {e}")

try:
    from deepface.models.spoofing import FasNet
    FasNet.Fasnet()
    print("Fasnet OK")
except Exception as e:
    print(f"Fasnet warn: {e}")

# Smoke test: actually run Fasnet inference to catch numpy/torch ABI issues
# that wouldn't appear during model load alone.
try:
    import numpy as np
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    faces = DeepFace.extract_faces(
        img_path=dummy,
        detector_backend="opencv",
        anti_spoofing=True,
        enforce_detection=False,
    )
    print(f"Fasnet smoke test OK — {len(faces)} faces detected on random image")
except Exception as e:
    # F-16 VERIFICATION STEP 2 (docs/review/10-moire-frr-investigation.md §17):
    # this used to allowlist only 2 substrings ("Numpy is not available", "cuInit")
    # and silently swallow everything else — including the exact ImportError this
    # bug produces. enforce_detection=False already means "no face found" won't
    # raise here, so there's no expected-failure case left to allowlist. Fail the
    # build on any exception.
    raise
EOF

# F-16: the old cleanup RUN used to live here, AFTER the model predownload/smoke
# test above — that duplicate second pass (unpatched blanket `-name test`) would
# silently re-delete tensorflow/_api/v2/__internal__/test/ even with the fixed
# cleanup block moved earlier, making the build pass for the wrong reason.
# Removed — the single cleanup pass earlier in this file (before Step 4) is the
# only one now. Re-verified after removing this: the built image actually keeps
# tensorflow/_api/v2/__internal__/test/ intact (docs/review/10-moire-frr-investigation.md §17).

COPY . .

RUN printf '#!/bin/sh\nexport TF_USE_LEGACY_KERAS=1\nexport TF_CPP_MIN_LOG_LEVEL=2\nexec gunicorn "app:create_app()" --bind "0.0.0.0:${PORT:-8080}" --workers 1 --timeout 300\n' > /start.sh && chmod +x /start.sh
ENTRYPOINT ["/start.sh"]
