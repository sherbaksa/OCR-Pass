-- Инициализационный скрипт для PostgreSQL
-- Создается автоматически при первом запуске контейнера

-- Устанавливаем расширения
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Создаем схему для приложения (опционально)
CREATE SCHEMA IF NOT EXISTS passport_ocr;

-- Выдаем права пользователю
GRANT ALL PRIVILEGES ON SCHEMA passport_ocr TO "user";
GRANT ALL PRIVILEGES ON DATABASE passport_db TO "user";

-- Логирование
\echo 'PostgreSQL initialized successfully for Passport OCR Service'
