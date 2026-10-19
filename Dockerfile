# Face stack v2: InsightFace buffalo_l + Fasnet ONNX on ONNX Runtime CPU.
# No TensorFlow, DeepFace or torch at runtime (those are requirements-dev.txt).
#
# Two stages: the builder installs into a venv and downloads buffalo_l; the
# runtime image copies only the venv and the three model files it uses, so pip
# caches, the model zip and unused models never reach the final image.
# insightface 2.1 is a pure-Python wheel and every other dependency ships
# manylinux wheels, so no compiler (build-essential) is needed in either stage.

FROM python:3.11-slim AS builder
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /build
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir --constraint constraints.txt -r requirements.txt && \
    # insightface depends on opencv-python (needs libGL); swap it for the headless build.
    pip uninstall -y opencv-python opencv-python-headless && \
    pip install --no-cache-dir --constraint constraints.txt "opencv-python-headless>=4.8.0,<4.12" && \
    find /opt/venv -depth -type d -name __pycache__ -exec rm -rf {} + && \
    find /opt/venv -name "*.pyc" -delete
# buffalo_l at build time: FaceAnalysis downloads it to ~/.insightface/models on
# first use; keep only detection, recognition and 3D landmarks.
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition', 'landmark_3d_68'], providers=['CPUExecutionProvider'])" && \
    mkdir -p /models/buffalo_l && \
    cp /root/.insightface/models/buffalo_l/det_10g.onnx \
       /root/.insightface/models/buffalo_l/w600k_r50.onnx \
       /root/.insightface/models/buffalo_l/1k3d68.onnx /models/buffalo_l/ && \
    sha256sum /models/buffalo_l/*.onnx

FROM python:3.11-slim
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /models/buffalo_l /root/.insightface/models/buffalo_l
WORKDIR /app
COPY . .
# Fail the build if the runtime cannot load and run the face models.
RUN python -c "import numpy, cv2, onnxruntime, insightface; print('numpy', numpy.__version__, 'cv2', cv2.__version__, 'ort', onnxruntime.__version__, 'insightface', insightface.__version__)" && \
    python -c "import importlib.util as u; assert u.find_spec('tensorflow') is None and u.find_spec('deepface') is None and u.find_spec('torch') is None, 'legacy stack present'" && \
    python -c "from app.runtime_check import check_face_runtime; check_face_runtime(); from app.services import face_engine; print('warm-up s', round(face_engine.warm_up(), 2))"

RUN printf '#!/bin/sh\nexec gunicorn "app:create_app()" --bind "0.0.0.0:${PORT:-8080}" --workers 1 --timeout 300\n' > /start.sh && chmod +x /start.sh
ENTRYPOINT ["/start.sh"]
