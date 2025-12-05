# Passport OCR Service

Сервис для автоматического распознавания паспортов РФ с использованием компьютерного зрения и OCR.

## Стек технологий
- **Бэкенд:** FastAPI (Python 3.11+)
- **База данных:** PostgreSQL
- **Хранилище:** MinIO (S3-совместимое)
- **OCR:** PaddleOCR / Google Vision API
- **Контейнеризация:** Docker, Docker Compose
- **Фронтенд:** (планируется) React/Vue.js для загрузки и просмотра

## Структура проекта
passport-ocr-service/
├── backend/ # FastAPI приложение
├── frontend/ # Фронтенд (будущее)
├── infra/ # Инфраструктура (миграции БД, конфиги)
├── scripts/ # Вспомогательные скрипты
├── storage/ # Локальное хранилище (для разработки)
└── tests/ # Тесты

text

## Быстрый старт

### Предварительные требования
- Docker и Docker Compose
- Python 3.11+ (для локальной разработки)
- Git

### Установка для разработки
1. Клонируйте репозиторий:
   ```bash
   git clone <repository-url>
   cd passport-ocr-service
Создайте файл .env на основе .env.example:

bash
cp .env.example .env
# Отредактируйте .env при необходимости
Запустите сервисы через Docker Compose:

bash
docker-compose up --build
Откройте в браузере:

FastAPI: http://localhost:8000

FastAPI документация: http://localhost:8000/docs

MinIO Console: http://localhost:9001

MinIO API: http://localhost:9000

API Endpoints (основные)
GET / - Проверка работоспособности

GET /health - Health check

POST /api/v1/upload - Загрузка изображения паспорта (будущее)

GET /api/v1/documents/{id} - Получение информации о документе (будущее)

Разработка
Локальная разработка без Docker
bash
cd backend
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
Проверка окружения
bash
python scripts/check_env.py
Схема базы данных
См. infra/db_schema.md

Лицензия
MIT
