# Схема базы данных для Passport OCR Service

## Таблица `documents`
Хранит информацию о загруженных документах (изображениях паспортов).

```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    original_filename VARCHAR(255) NOT NULL,
    stored_file_path VARCHAR(500) NOT NULL, -- Путь в S3/MinIO
    file_size INTEGER NOT NULL,
    mime_type VARCHAR(100) NOT NULL,
    upload_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'uploaded' CHECK (status IN ('uploaded', 'processing', 'processed', 'error')),
    upload_ip INET,
    user_id UUID -- Для будущей аутентификации
);
CREATE TABLE passport_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Основные поля (серия и номер)
    series VARCHAR(4),
    number VARCHAR(6),
    
    -- Личные данные
    last_name VARCHAR(100),
    first_name VARCHAR(100),
    middle_name VARCHAR(100),
    gender VARCHAR(10) CHECK (gender IN ('male', 'female')),
    birth_date DATE,
    birth_place TEXT,
    
    -- Данные выдачи
    issue_date DATE,
    issue_authority TEXT,
    department_code VARCHAR(7),
    
    -- Адрес регистрации
    registration_address TEXT,
    
    -- Дополнительные метаданные
    confidence_score FLOAT, -- Уверенность OCR (0.0-1.0)
    raw_ocr_text TEXT, -- Полный сырой текст от OCR
    extracted_fields JSONB, -- Все поля в JSON формате
    
    processing_time INTEGER, -- Время обработки в мс
    ocr_engine VARCHAR(50), -- Использованный движок OCR
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE processing_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    log_level VARCHAR(20) CHECK (log_level IN ('info', 'warning', 'error')),
    message TEXT,
    details JSONB
);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_upload_date ON documents(upload_date);
CREATE INDEX idx_passport_data_document_id ON passport_data(document_id);
CREATE INDEX idx_passport_data_series_number ON passport_data(series, number);
CREATE INDEX idx_processing_logs_document_id ON processing_logs(document_id);
CREATE INDEX idx_processing_logs_timestamp ON processing_logs(timestamp);
