FROM python:3.11-slim

# cairosvg needs Cairo at the OS level; libGL and libglib are needed by
# OpenCV even in its headless build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ /app/
COPY frontend/ /frontend/
ENV FRONTEND_DIR=/frontend

EXPOSE 5000

# Serves the API and the PWA. Jobs are executed by a separate worker
# container running `python worker.py` against the same Redis.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
