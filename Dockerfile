FROM python:3.11-slim

WORKDIR /app

# Install build deps, install packages, then PURGE build deps in the same RUN layer.
# Purging in a separate layer would NOT reduce image size — the earlier layer still contains them.
COPY requirements.txt .

ARG CACHE_BUST=20260419b
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true && \
    pip install --no-cache-dir "opencv-python-headless>=4.8.0" && \
    apt-get purge -y --auto-remove gcc libpq-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /root/.cache/pip /tmp/* /var/tmp/*

# Pre-download model weights so they're baked into the image (no cold-start download).
# - Facenet512: always needed for identity embeddings
# - Fasnet (MiniFASNet v1 + v2): needed for DeepFace anti_spoofing=True
RUN python - <<'EOF'
import os
os.environ.setdefault("HOME", "/root")
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
EOF

# Remove Python bytecode caches and test files from site-packages to save space.
RUN find /usr/local/lib/python3.11 -depth -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11 -depth -type d -name tests -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11 -depth -type d -name test -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11 -name "*.pyc" -delete 2>/dev/null || true

COPY . .

RUN printf '#!/bin/sh\nexec gunicorn "app:create_app()" --bind "0.0.0.0:${PORT:-8080}" --workers 1 --timeout 300\n' > /start.sh && chmod +x /start.sh
ENTRYPOINT ["/start.sh"]
