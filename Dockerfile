FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# deepface pulls in opencv-python (non-headless) as a transitive dep.
# Force-replace with headless after all deps are installed.
RUN pip uninstall -y opencv-python opencv-python-headless && \
    pip install --no-cache-dir "opencv-python-headless>=4.8.0"

RUN printf '#!/bin/sh\nexec gunicorn "app:create_app()" --bind "0.0.0.0:${PORT:-8080}" --workers 1 --timeout 300\n' > /start.sh && chmod +x /start.sh
ENTRYPOINT ["/start.sh"]
