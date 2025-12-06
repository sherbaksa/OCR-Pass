"""
Модуль сервисов OCR-Pass
Экспорт основных сервисов для использования в приложении
"""

from backend.services.storage_service import storage_service, StorageService
from backend.services.upload_service import UploadService
from backend.services.preprocessing_service import preprocessing_service, PreprocessingService

__all__ = [
    "storage_service",
    "StorageService",
    "UploadService",
    "preprocessing_service",
    "PreprocessingService",
]
