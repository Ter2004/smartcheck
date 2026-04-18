FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Change CACHE_BUST value to force re-run of all layers below (pip install + model download)
ARG CACHE_BUST=20260419a
RUN pip install --no-cache-dir -r requirements.txt

# deepface pulls in opencv-python (non-headless) as a transitive dep.
# Replace with headless BEFORE model download so libGL.so.1 is not required.
RUN pip uninstall -y opencv-python opencv-python-headless && \
    pip install --no-cache-dir "opencv-python-headless>=4.8.0"

# Pre-download model weights so they're baked into the image (no runtime download).
# - Facenet512: always needed for embeddings
# - Fasnet (MiniFASNet v1 + v2): needed for DeepFace anti_spoofing=True
RUN python - <<'EOF'
import os
os.environ.setdefault("HOME", "/root")
from deepface import DeepFace

# FaceNet512 for identity embeddings
try:
    DeepFace.build_model("Facenet512")
    print("Facenet512 OK")
except Exception as e:
    print(f"Facenet512 warn: {e}")

# Fasnet for anti-spoofing — DeepFace's Fasnet wrapper downloads both
# MiniFASNetV1SE and MiniFASNetV2 checkpoints on first init.
try:
    from deepface.models.spoofing import FasNet
    FasNet.Fasnet()
    print("Fasnet OK")
except Exception as e:
    print(f"Fasnet warn: {e}")
EOF

COPY . .

RUN printf '#!/bin/sh\nexec gunicorn "app:create_app()" --bind "0.0.0.0:${PORT:-8080}" --workers 1 --timeout 300\n' > /start.sh && chmod +x /start.sh
ENTRYPOINT ["/start.sh"]
