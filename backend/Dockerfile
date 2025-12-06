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
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копируем файл зависимостей и устанавливаем их
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Копируем код приложения
COPY backend /app/backend

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Открываем порт
EXPOSE 8000

# Команда запуска
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
