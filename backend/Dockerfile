# backend/Dockerfile
# Используем официальный Python slim образ (Debian-based -> apt)
FROM python:3.11-slim

# Устанавливаем системные зависимости, необходимые для сборки некоторых пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копируем файл зависимостей и устанавливаем их
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Копируем код приложения
COPY backend /app

ENV PYTHONUNBUFFERED=1

# Открываем порт
EXPOSE 8000

# Команда запуска (можно изменить при необходимости)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

