# backend/Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    curl \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем wheel файл
COPY wheels/paddlepaddle-3.2.0-cp311-cp311-linux_x86_64.whl /tmp/

# Устанавливаем PaddlePaddle из локального файла
RUN pip install --no-cache-dir /tmp/paddlepaddle-3.2.0-cp311-cp311-linux_x86_64.whl

# Проверка
RUN python -c "import paddle; print('PaddlePaddle version:', paddle.__version__)"

# Устанавливаем PaddleOCR
RUN pip install --no-cache-dir paddleocr

# Предзагрузка моделей
RUN python3 -c "\
from paddleocr import PaddleOCR; \
print('Downloading PaddleOCR models...'); \
ocr = PaddleOCR(lang='ru'); \
print('PaddleOCR models downloaded successfully'); \
" || echo "Warning: PaddleOCR model download failed"

# Копируем requirements и код
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt || true

COPY backend /app/backend
# Копируем обученную модель
COPY trained_model/trained_model /app/models/trained_rec_model

COPY scripts /app/scripts

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
