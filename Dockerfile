FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python && \
    pip install --no-cache-dir "opencv-python-headless>=4.8.0"

COPY . .

CMD ["sh", "-c", "gunicorn 'app:create_app()' --bind 0.0.0.0:$PORT --workers 1 --timeout 300"]
