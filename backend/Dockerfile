# backend/Dockerfile
# Используем официальный Python slim образ (Debian-based -> apt)
FROM python:3.11-slim

# Устанавливаем системные зависимости
# - build-essential, gcc, libpq-dev - для компиляции Python пакетов и PostgreSQL
# - curl - для проверки health endpoints
# - poppler-utils - для pdf2image (работа с PDF)
# - libgl1 - для OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
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
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копируем файл зависимостей и устанавливаем их
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# НОВОЕ: Предзагрузка моделей PaddleOCR во время сборки image
# Это решает проблему с недоступностью китайского CDN у заказчика
RUN python3 -c "\
from paddleocr import PaddleOCR; \
import numpy as np; \
print('Downloading PaddleOCR models...'); \
ocr = PaddleOCR(lang='ru', use_gpu=False, show_log=True); \
print('PaddleOCR models downloaded successfully'); \
img = np.ones((100, 100, 3), dtype=np.uint8) * 255; \
result = ocr.ocr(img, cls=True); \
print('PaddleOCR initialization test passed'); \
" || echo "Warning: PaddleOCR model download failed, will retry at runtime"

# Копируем код приложения
COPY backend /app/backend
COPY scripts /app/scripts

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Открываем порт
EXPOSE 8000

# Команда запуска
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
