FROM python:3.11-slim

WORKDIR /app

# Force legacy Keras (belt-and-suspenders even though tensorflow 2.15 defaults to Keras 2)
ENV TF_USE_LEGACY_KERAS=1
ENV TF_CPP_MIN_LOG_LEVEL=2

COPY requirements.txt constraints.txt ./

# Change CACHE_BUST to force re-run of all layers below (pip install + model download)
ARG CACHE_BUST=20260419f
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
    pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true && \
    pip install --no-cache-dir --constraint constraints.txt "opencv-python-headless>=4.8.0,<4.12" && \
    \
    # Step 4: Verify the critical imports work BEFORE proceeding.
    # Build fails loudly here rather than producing a silently broken image.
    python -c "import numpy; print('numpy', numpy.__version__); assert numpy.__version__.startswith('1.'), 'FAIL: numpy must be 1.x'" && \
    python -c "import torch; print('torch', torch.__version__); import numpy; t = torch.zeros(1); t.numpy(); print('torch.numpy OK')" && \
    python -c "import tensorflow; print('tensorflow', tensorflow.__version__); assert tensorflow.__version__.startswith('2.15'), 'FAIL: tensorflow must be 2.15.x'" && \
    python -c "from deepface import DeepFace; print('DeepFace import OK')" && \
    \
    # Step 5: Purge build deps and clean caches in SAME layer (critical for image size)
    apt-get purge -y --auto-remove gcc libpq-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /root/.cache/pip /tmp/* /var/tmp/*

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

try:
    from deepface.models.face_detection import YuNet
    YuNet.YuNetClient()
    print("YuNet OK")
except Exception as e:
    print(f"YuNet warn: {e}")

# Smoke test: actually run Fasnet inference to catch numpy/torch ABI issues
# that wouldn't appear during model load alone.
try:
    import numpy as np
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    faces = DeepFace.extract_faces(
        img_path=dummy,
        detector_backend="yunet",
        anti_spoofing=True,
        enforce_detection=False,
    )
    print(f"Fasnet smoke test OK — {len(faces)} faces detected on random image")
except Exception as e:
    err_str = str(e)
    if "Numpy is not available" in err_str or "cuInit" in err_str:
        raise  # these are the broken-install symptoms — fail the build
    print(f"Fasnet smoke test ran (no face expected on random image): {err_str[:100]}")
EOF

# Remove pycache and test files to reduce image size
RUN find /usr/local/lib/python3.11 -depth -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11 -depth -type d -name tests -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11 -depth -type d -name test -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11 -name "*.pyc" -delete 2>/dev/null || true

COPY . .

RUN printf '#!/bin/sh\nexport TF_USE_LEGACY_KERAS=1\nexport TF_CPP_MIN_LOG_LEVEL=2\nexec gunicorn "app:create_app()" --bind "0.0.0.0:${PORT:-8080}" --workers 1 --timeout 300\n' > /start.sh && chmod +x /start.sh
ENTRYPOINT ["/start.sh"]
