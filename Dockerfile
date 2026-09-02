FROM python:3.12-slim

# opencv-python needs libGL/libglib even for headless use — a well-known
# Docker gotcha (see README "Bug found & fixed" for the analogous Windows
# opencv issue this project already hit once, for a different reason).
# onnxruntime and TensorFlow both link against libgomp for OpenMP.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Model weight caches and uploaded files are Docker volumes (see
# docker-compose.yml) so a container rebuild doesn't re-download ~350MB of
# ArcFace/mtcnn/insightface weights every time.
ENV DEEPFACE_HOME=/app/storage/deepface \
    INSIGHTFACE_HOME=/app/storage/insightface \
    UPLOAD_ROOT=/app/storage/uploads

EXPOSE 8000

# docker-compose.yml overrides this command for the worker service.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
