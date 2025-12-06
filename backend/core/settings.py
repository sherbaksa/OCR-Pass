from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List


class Settings(BaseSettings):
    """
    Настройки приложения, загружаемые из переменных окружения
    """
    
    # Основные настройки приложения
    app_name: str = Field(default="Passport OCR Service", description="Название приложения")
    debug: bool = Field(default=False, description="Режим отладки")
    api_v1_prefix: str = Field(default="/api/v1", description="Префикс API v1")
    
    # База данных PostgreSQL
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/passport_db",
        description="URL подключения к PostgreSQL"
    )
    
    # MinIO (S3-совместимое хранилище)
    minio_endpoint: str = Field(default="localhost:9000", description="MinIO endpoint")
    minio_access_key: str = Field(default="minioadmin", description="MinIO access key")
    minio_secret_key: str = Field(default="minioadmin", description="MinIO secret key")
    minio_bucket_name: str = Field(default="passport-documents", description="MinIO bucket name")
    minio_secure: bool = Field(default=False, description="Использовать HTTPS для MinIO")
    
    # Внешние API
# Google Vision API настройки
    google_vision_enabled: bool = Field(default=False, description="Включить Google Vision API")
    google_vision_mock_mode: bool = Field(default=False, description="Использовать mock-режим (без реальных API вызовов)")
    google_application_credentials: Optional[str] = Field(
        default=None, 
        description="Путь к JSON-файлу с credentials для Google Vision"
    )
# PaddleOCR настройки
    paddleocr_use_gpu: bool = Field(default=False, description="Использовать GPU для PaddleOCR")
    paddleocr_lang: str = Field(default="ru", description="Язык модели PaddleOCR")
    paddleocr_det_model_dir: Optional[str] = Field(default=None, description="Путь к модели детекции")
    paddleocr_rec_model_dir: Optional[str] = Field(default=None, description="Путь к модели распознавания")
    paddleocr_cls_model_dir: Optional[str] = Field(default=None, description="Путь к модели классификации")

    # Настройки безопасности
    secret_key: str = Field(default="change-me-in-production", description="Секретный ключ для JWT")
    algorithm: str = Field(default="HS256", description="Алгоритм шифрования JWT")
    access_token_expire_minutes: int = Field(default=30, description="Время жизни токена в минутах")
    
    # Настройки загрузки файлов
    max_upload_size_mb: int = Field(default=10, description="Максимальный размер загружаемого файла в МБ")
    allowed_extensions: List[str] = Field(
        default=["jpg", "jpeg", "png", "pdf"],
        description="Разрешенные расширения файлов"
    )
    min_image_resolution: int = Field(default=1000, description="Минимальное разрешение изображения (px)")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Игнорируем лишние поля из .env


# Создаем глобальный экземпляр настроек
settings = Settings()
